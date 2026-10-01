"""Shared pieces for the LangChain/LangGraph runners (run_langgraph.py, run_deepagent.py).

What run_mini.py spells out as bash-era prompt rules becomes structure here:
  * SYSTEM_PROMPT keeps only the domain rules; the loop ends when the model
    answers in prose, so there is no COMPLETE_TASK echo, SUMMARY quoting or
    tool_choice="required".
  * ToolCallGuard refuses exact repeats of a VIRA call (the duplicate real API
    call run_mini.py works around) and turns crashes into error results.
  * ModelCallLimitMiddleware caps model calls per task (mini's step_limit).
  * --approve-all puts a human in front of every VIRA call (LangGraph interrupt);
    in real mode, calls that change data on VIRA (score, insights, and every Jeni
    task that isn't a read) always pause.
  * A Toolset picks the tools: VIRA (the sample endpoints) or toolset("jeni") (Jeni's tasks).

Never import run_mini from here: it reads .env and pulls in minisweagent and
litellm on import.
"""
import argparse
import functools
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

import grounding
import jeni_tools
import recruiter_cli
import vira_tools
from terminal import printable


# langsmith reads *_TRACING_V2 before *_TRACING, under both prefixes.
TRACING_VARS = ("LANGSMITH_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


def set_tracing(enabled: bool) -> None:
    """LangSmith tracing ships prompts and tool results off the machine: opt-in only."""
    value = "true" if enabled else "false"
    for name in TRACING_VARS:
        os.environ[name] = value
    try:                                   # langsmith caches its env lookups
        from langsmith.utils import get_env_var
        get_env_var.cache_clear()
    except (ImportError, AttributeError):
        pass


set_tracing(False)  # before any run; runners re-enable it only for --trace

SYSTEM_PROMPT = """\
You operate a recruiter platform through the VIRA tools. Call a tool, read its JSON result,
then decide the next step.

Rules:
- Do ONLY what the user's task asks. Do not add extra steps.
- match_id, app_id, profile_id and job_id are DISTINCT. Never pass one where another is
  expected. If you need an id type you don't have and no tool returns it, do not substitute
  another id: finish as described below and say which id is missing.
- Never invent any field value. Every value must come from either the OUTPUT of a previous
  tool call, or explicit user input. If a required value is available from neither, do not
  guess: finish as described below and say what was missing.
- Never repeat a tool call with the same arguments, and never repeat a call that failed the
  same way. If you cannot make progress, finish.
- Relay what the tools return. Do not write or rewrite content yourself (for example a job
  description), and do not call a tool again just to get a longer or better result.
- "status": "ok" only means the API call was received, NOT that the operation succeeded.
  Always read the result fields for the real outcome. If a result field contains an error or
  a message saying nothing was found/obtained/processed, treat it as a FAILURE even though
  status is "ok", and correct your next call.
- When you are done, reply in plain text without calling a tool. If the task is only
  partially done or cannot be fully completed, start that reply with
  SUMMARY: <what succeeded> | <what failed or is missing> | <why>
- End EVERY reply that has no tool call with a status line as its LAST line, one of:
    STATUS: done                 - the task is finished (succeeded or cannot proceed)
    STATUS: needs_user: <what>   - you must get something from the user to continue
  Use needs_user only when you are genuinely blocked on the user (e.g. a value no tool
  can supply). Otherwise, if more tool calls are needed, make them instead of replying.
"""


@dataclass(frozen=True)
class Toolset:
    """What an agent can call: the tools, which of them only read, and its prompt.

    user_only: arguments whose values must be exactly what the user wrote (ToolCallGuard).
    """
    name: str
    tools: Callable[[], list]
    names: frozenset
    read_only: frozenset
    user_only: frozenset = frozenset()
    prompt: str = SYSTEM_PROMPT


# The four sample AI endpoints (and the assumed match-id lookup), used to compare runtimes.
VIRA = Toolset("vira", vira_tools.langchain_tools, frozenset(vira_tools.NAMES),
               frozenset(vira_tools.READ_ONLY))
TOOLSET_NAMES = ("jeni", "jeni_db", "vira")


@functools.lru_cache(maxsize=None)
def _jeni() -> Toolset:
    names = jeni_tools.names()          # reads the catalog; CatalogMissing without it
    return Toolset("jeni", jeni_tools.langchain_tools, names, jeni_tools.READ_ONLY & names,
                   jeni_tools.USER_ONLY, SYSTEM_PROMPT + jeni_tools.RULES)


def _db(query_tools, context) -> Toolset:
    """The read-only db lookup/validation tools (db_queries.py), context bound at build time."""
    import db_tools
    names = db_tools.names(query_tools)
    return Toolset("db", lambda: db_tools.langchain_tools(query_tools, context),
                   names, names, db_tools.USER_ONLY, SYSTEM_PROMPT + db_tools.RULES)


def _jeni_db(query_tools, context) -> Toolset:
    """jeni (act) + db (resolve/validate) in ONE agent: db tools feed ids to jeni tools.

    A flat union of both tool lists; names are disjoint (verified) so no collision.
    """
    import db_tools
    jt = _jeni()
    dbt = _db(query_tools, context)
    clash = jt.names & dbt.names
    if clash:
        raise ValueError(f"jeni/db tool name collision: {sorted(clash)}")
    return Toolset(
        "jeni_db",
        lambda: jt.tools() + dbt.tools(),
        jt.names | dbt.names,
        jt.read_only | dbt.read_only,
        jt.user_only | dbt.user_only,
        SYSTEM_PROMPT + jeni_tools.RULES + db_tools.RULES,
    )


def toolset(name: str, *, query_tools=None, context=None) -> Toolset:
    """VIRA (the sample endpoints), Jeni's own tasks, or jeni+db combined.

    jeni / jeni_db read the internal catalog (config/README.md) on first use.
    db and jeni_db also need `query_tools` (db_queries.build_db_queries(pool) or
    fake_db_queries(fixtures)) and `context` ({"auth_profile": {"company_id": ...}}),
    supplied by the runner's main() -- never by the model.
    """
    if name == "vira":
        return VIRA
    if name == "jeni":
        return _jeni()
    if name in ("db", "jeni_db"):
        if query_tools is None or context is None:
            raise SystemExit(f"toolset {name!r} needs a db pool/context; the runner must "
                             f"pass query_tools and context (see run_langgraph.py).")
        return _db(query_tools, context) if name == "db" else _jeni_db(query_tools, context)
    raise ValueError(f"unknown toolset {name!r}")


def cli_toolset(args: argparse.Namespace, *, query_tools=None, context=None) -> Toolset:
    """toolset(args.tools) for a runner's main(): a missing catalog is a message, not a traceback."""
    try:
        return toolset(getattr(args, "tools", "vira"),
                       query_tools=query_tools, context=context)
    except jeni_tools.CatalogMissing as exc:
        raise SystemExit(str(exc)) from None


# --- model -------------------------------------------------------------------
def build_chat_model(model_id: str | None = None):
    """CHAT_MODEL (litellm style, as in run_mini.py) -> a LangChain chat model."""
    from langchain.chat_models import init_chat_model
    model_id = model_id or os.environ.get("CHAT_MODEL", "gpt-5-mini")
    provider, sep, name = model_id.partition("/")
    lc_id = f"{provider}:{name}" if sep else model_id     # "openai/gpt-5-mini" -> "openai:gpt-5-mini"
    is_openai = lc_id.startswith("openai:") or (":" not in lc_id and lc_id.startswith(("gpt-", "o1", "o3", "o4")))
    # Chat Completions, like mini's litellm path: nothing is stored server-side
    # (the Responses API stores responses by default).
    kwargs = {"use_responses_api": False} if is_openai else {}
    return init_chat_model(lc_id, **kwargs)


# --- middleware --------------------------------------------------------------
def _normalise(args: dict) -> str:
    return json.dumps({k: v for k, v in sorted(args.items()) if v not in (None, [], "", {})},
                      sort_keys=True, ensure_ascii=False)


class CallLedger:
    """VIRA calls already made, per task thread, shared by an agent and all its subagents.

    A subagent runs in its own message context, so the message-history check alone
    can't see a call the parent (or a sibling subagent) already made.  All of them
    share the task's thread_id, so one ledger keyed by it covers the whole task.
    """

    def __init__(self) -> None:
        self._seen: set[tuple] = set()
        self._lock = threading.Lock()

    def claim(self, key: tuple) -> bool:
        """True the first time `key` is seen; False for every repeat."""
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True


class ToolCallGuard(AgentMiddleware):
    """For the VIRA tools only: refuse repeats and ids the model can't have, never crash the run."""

    def __init__(self, names=frozenset(vira_tools.NAMES), ledger: CallLedger | None = None,
                 user_only=frozenset()):
        super().__init__()
        self.names = set(names)
        self.ledger = ledger
        self.user_only = set(user_only)

    def _repeat_of_earlier_call(self, request) -> bool:
        call = request.tool_call
        key = _normalise(call["args"])
        for msg in request.state.get("messages", []):
            for earlier in getattr(msg, "tool_calls", None) or []:
                if earlier.get("id") == call.get("id"):
                    return False          # everything after this is not earlier
                if earlier["name"] == call["name"] and _normalise(earlier["args"]) == key:
                    return True
        return False

    @staticmethod
    def _result(request, message: str) -> ToolMessage:
        call = request.tool_call
        return ToolMessage(content=json.dumps({"status": "error", "message": message}),
                           tool_call_id=call["id"], name=call["name"], status="error")

    @staticmethod
    def _bad_ids(request) -> tuple[list[str], list[str]]:
        """(wrong-kind ids, invented ids), from grounding's provenance of the call's id args.

        Wrong kind: an id passed as a different kind than it came back as.  Invented: an id
        that is in neither anything the user wrote (any turn of the conversation) nor any
        earlier tool result.  A reviewer's edited args (HumanInTheLoopMiddleware) count as
        user input, like the task.
        """
        messages = request.state.get("messages", [])
        sources = ([("task", m.text) for m in messages if m.type == "human"]
                   + [(f"step {i}", m.text) for i, m in enumerate(messages) if m.type == "tool"])
        edited = (request.state.get("hitl_edited_tool_calls") or {}).get(request.tool_call.get("id"))
        if edited:
            sources.append(("task", json.dumps(edited.get("args", {}))))
        misused, invented = [], []
        for arg, entries in grounding.ground_args(request.tool_call["args"], sources).items():
            if arg not in grounding.ID_KINDS:
                continue
            for p in entries:
                if "misused_as" in p:
                    misused.append(f"{p['value']} is a {p['misused_as']}, not a {arg.removesuffix('s')}")
                elif not p["sources"]:
                    invented.append(f"{p['value']}")
        return misused, invented

    def _not_from_user(self, request) -> list[str]:
        """User-only arguments (e.g. a candidate's email) with a value the user never wrote.

        A reviewer's edited args count as the user's words, as for ids.
        """
        args = request.tool_call["args"] or {}
        fields = sorted(self.user_only & set(args))
        if not fields:
            return []
        messages = request.state.get("messages", [])
        said = " ".join(m.text for m in messages if m.type == "human")
        edited = (request.state.get("hitl_edited_tool_calls") or {}).get(request.tool_call.get("id"))
        if edited:
            said += " " + json.dumps(edited.get("args", {}), ensure_ascii=False)
        said = said.lower()
        bad = []
        for field in fields:
            values = args[field] if isinstance(args[field], list) else [args[field]]
            if any(not isinstance(v, str) or v.strip().lower() not in said
                   for v in values if v not in (None, "")):
                bad.append(field)
        return bad

    def _refusal(self, request) -> ToolMessage | None:
        misused, invented = self._bad_ids(request)
        if misused:
            return self._result(request, "Refused: " + "; ".join(misused) + ". Pass an id only as "
                                "the kind it came back as: call the tool that returns the kind "
                                "you need (profile ids -> get_match_id_from_profile_id), or "
                                "finish and say which id is missing.")
        if invented:
            return self._result(request, f"Refused: {', '.join(invented)} isn't in the task or any "
                                "earlier result. Never invent ids: finish and say which id is missing.")
        not_said = self._not_from_user(request)
        if not_said:     # names only: the value may be personal data
            return self._result(request, f"Refused: {', '.join(not_said)} must be exactly what the "
                                "user wrote, and it isn't. Ask the user for it; never take it from a "
                                "tool result or make it up.")
        repeat = self._repeat_of_earlier_call(request)   # also orders parallel duplicates
        if not repeat and self.ledger is not None:
            info = request.runtime.execution_info
            call = request.tool_call
            repeat = not self.ledger.claim((info.thread_id if info else None, call["name"],
                                            _normalise(call["args"])))
        if repeat:
            return self._result(request, "Refused: identical to an earlier call in this task. "
                                         "Use that result instead of calling again.")
        return None

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self.names:
            return handler(request)
        refused = self._refusal(request)
        if refused:
            return refused
        try:
            return handler(request)
        except GraphBubbleUp:            # interrupts and parent commands must propagate
            raise
        except Exception as exc:
            return self._result(request, f"tool failed ({type(exc).__name__})")

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self.names:
            return await handler(request)
        refused = self._refusal(request)
        if refused:
            return refused
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._result(request, f"tool failed ({type(exc).__name__})")


def middleware(step_limit: int = 12, ledger: CallLedger | None = None,
               toolset: Toolset = VIRA) -> list:
    """Guard + a cap on model calls per run.

    A run is one user turn, or one resume after an approval, so a person starts each one.
    A per-thread cap would end a conversation after `step_limit` calls in total.  Pass one
    shared ledger to an agent and all its subagents.
    """
    return [ToolCallGuard(names=toolset.names, ledger=ledger, user_only=toolset.user_only),
            ModelCallLimitMiddleware(run_limit=step_limit, exit_behavior="end")]


DECISIONS = {"allowed_decisions": ["approve", "edit", "reject"]}


def interrupt_on(approve_all: bool, mode: str | None = None, toolset: Toolset = VIRA) -> dict:
    """Which tools pause for a human.

    Read-only tools (GET-style VIRA calls, db lookups/validations in toolset.read_only)
    NEVER pause: they change nothing, so approving them is just noise.  So:
      * --approve-all: every data-CHANGING tool (all names minus read_only);
      * real mode: the same data-changing set (unsafe to skip on real VIRA);
      * otherwise (mock, no --approve-all): nothing.
    `mode` defaults to the configured one (vira_tools.configure), so a real-mode agent
    can't be built without the write gates.
    """
    if approve_all:
        names = toolset.names - toolset.read_only
    elif (mode or vira_tools.current_mode()) == "real":
        names = toolset.names - toolset.read_only
    else:
        names = set()
    return {name: DECISIONS for name in sorted(names)}


# --- running a task ----------------------------------------------------------
def write_private(path, text: str) -> None:
    """A trace or report: owner-only (0600), with emails and phone numbers scrubbed.

    Traces hold the task text and final answers, which _mask_pii never saw.  A missing
    directory (e.g. traces/) is created owner-only.
    """
    Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):
        try:
            os.fchmod(fd, 0o600)              # an existing file keeps its mode otherwise
        except OSError:
            pass
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(recruiter_cli._scrub_text(text))


class UsageCounter(BaseCallbackHandler):
    """Counts every chat-model call in a run, subagents included."""

    def __init__(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.calls += 1

    def on_llm_end(self, response, **kwargs) -> None:
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)


def _ask_args() -> dict:
    while True:
        try:
            args = json.loads(input("new args as JSON > "))
        except ValueError:
            print("not valid JSON; try again")
            continue
        if isinstance(args, dict):
            return args
        print('expected a JSON object, e.g. {"job_ids": [124]}')


def ask_human(request: dict) -> list[dict]:
    """One decision per pending tool call: approve, reject, or edit its args."""
    decisions = []
    for action in request["action_requests"]:
        print(printable(f"\n[approval] {action['name']}"
                        f"({json.dumps(action['args'], ensure_ascii=False)})"))
        answer = input("approve? [y]es / [n]o / [e]dit args > ").strip().lower()
        if answer.startswith("y"):
            decisions.append({"type": "approve"})
        elif answer.startswith("e"):
            decisions.append({"type": "edit",
                              "edited_action": {"name": action["name"], "args": _ask_args()}})
        else:
            decisions.append({"type": "reject", "message": "The user declined this call."})
    return decisions


def run_task(agent, task: str, *, decide: Callable[[dict], list[dict]] = ask_human,
             callbacks: list | None = None, thread_id: str | None = None) -> dict:
    """Run one user turn, pausing for `decide` at every interrupt.

    A fresh thread by default.  Pass the same `thread_id` again to continue that
    conversation: the agent's checkpointer keeps the earlier turns.
    """
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex},
              "callbacks": callbacks or []}
    out = agent.invoke({"messages": [{"role": "user", "content": task}]}, config, version="v2")
    while out.interrupts:
        if len(out.interrupts) == 1:
            resume: Any = {"decisions": decide(out.interrupts[0].value)}
        else:                            # e.g. parallel subagents, each paused on its own call
            resume = {i.id: {"decisions": decide(i.value)} for i in out.interrupts}
        out = agent.invoke(Command(resume=resume), config, version="v2")
    return out.value


def show(messages, start: int = 0) -> None:
    """Trajectory printout in the style of run_mini.show(); control characters removed.

    Prints messages[start:], numbered as in the whole conversation.
    """
    for i, m in enumerate(messages[start:], start):
        if m.type == "human":
            print(printable(f"[{i}] user      : {m.text}"))
        elif m.type == "ai":
            for tc in m.tool_calls:
                print(printable(f"[{i}] assistant → call: {tc['name']}"
                                f"({json.dumps(tc['args'], ensure_ascii=False)})"))
            if m.text:
                print(printable(f"[{i}] assistant : {m.text}"))
        elif m.type == "tool":
            print(printable(f"[{i}] tool      → {m.text}"))


def final_text(result: dict) -> str:
    return result["messages"][-1].text if result.get("messages") else ""


# --- CLI ---------------------------------------------------------------------
def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", choices=["real", "mock"], default="mock",
                   help="mock (default): local fake VIRA; real: call VIRA at $VIRA_BASE_URL")
    p.add_argument("--task", help="run this one task and exit (default: interactive prompt)")
    p.add_argument("--tools", choices=TOOLSET_NAMES, default="jeni_db",
                   help="vira (default): the sample AI endpoints; jeni: Jeni's own tasks "
                        "(needs config/jeni_tasks.json); jeni_db: jeni PLUS the read-only db "
                        "lookup/validation tools, so ids can be resolved/checked before acting")
    p.add_argument("--dsn", default=os.environ.get("TRON_POSTGRES_DSN"),
                   help="Postgres DSN for jeni_db/db (default: $TRON_POSTGRES_DSN). If unset, "
                        "jeni_db uses in-memory fake db queries so it runs offline.")
    p.add_argument("--company-id", type=int, default=5143,
                   help="authenticated company_id injected into db queries (tenant scope); "
                        "never taken from the model")
    p.add_argument("--approve-all", action="store_true",
                   help="pause for human approval before every VIRA tool call "
                        "(in real mode, score/insights always pause)")
    p.add_argument("--step-limit", type=int, default=12,
                   help="max model calls per task (run_mini.py uses 12)")
    p.add_argument("--trace", action="store_true",
                   help="allow LangSmith tracing if LANGSMITH_* is configured (off by default)")
    p.add_argument("--trace-json", metavar="PATH",
                   help="write each task's trace (steps + grounding) here, for the "
                        "visualisation page; owner-only, emails/phones scrubbed, e.g. "
                        "traces/run.json (gitignored)")
    return p


def setup(args: argparse.Namespace) -> None:
    set_tracing(args.trace)
    vira_tools.configure(args.mode)


def _trace_run(label: str, task: str, result: dict, counter: UsageCounter,
               seconds: float, tools=grounding.VIRA_TOOLS) -> dict:
    """One run in compare_agents.py's --json format, so the page reads both."""
    steps = grounding.trace_from_messages(result["messages"], task, tools)
    return {"runner": label, "task": task, "repeat": 1, "passed": None, "error": "",
            "model_calls": counter.calls, "input_tokens": counter.input_tokens,
            "output_tokens": counter.output_tokens, "seconds": round(seconds, 1), "cost": None,
            "audit": [], "steps": steps, "grounding": grounding.summary(steps),
            "final": final_text(result)}


def _report(result: dict, counter: "UsageCounter", t0: float, *, task: str,
            after, trace, label, tools) -> dict:
    """Shared post-run output for the sync and async runners."""
    messages = result["messages"]
    turn = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)
    print("\n=== trajectory ===")
    show(messages, turn)                 # this turn only; earlier turns were printed already
    if after:
        after(result)
    print(f"\n(model calls: {counter.calls}, tokens in/out: "
          f"{counter.input_tokens}/{counter.output_tokens})")
    if trace is not None:
        trace["runs"].append(_trace_run(label, task, result, counter, time.monotonic() - t0,
                                        tools))
        trace["tasks"][task] = {"text": task, "check": ""}
        write_private(trace["path"], json.dumps({k: v for k, v in trace.items() if k != "path"},
                                                ensure_ascii=False, indent=1))
        print(f"(trace written to {trace['path']})")
    return result


def run_and_show(agent, task: str, after: Callable[[dict], None] | None = None,
                 trace: dict | None = None, label: str = "",
                 tools=grounding.VIRA_TOOLS, thread_id: str | None = None) -> dict:
    counter = UsageCounter()
    t0 = time.monotonic()
    result = run_task(agent, task, callbacks=[counter], thread_id=thread_id)
    return _report(result, counter, t0, task=task, after=after, trace=trace,
                   label=label, tools=tools)


# --- async runners: needed when the toolset has async tools (db_tools) --------
async def arun_task(agent, task: str, *, decide: Callable[[dict], list[dict]] = ask_human,
                    callbacks: list | None = None, thread_id: str | None = None) -> dict:
    """Async twin of run_task: drives the graph with ainvoke so coroutine tools (db) run.

    Must be called on the SAME event loop that created the asyncpg pool, or asyncpg
    raises 'attached to a different loop'.  run_langgraph.py's async main() ensures this.
    """
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex},
              "callbacks": callbacks or []}
    out = await agent.ainvoke({"messages": [{"role": "user", "content": task}]}, config,
                              version="v2")
    while out.interrupts:
        if len(out.interrupts) == 1:
            resume: Any = {"decisions": decide(out.interrupts[0].value)}
        else:
            resume = {i.id: {"decisions": decide(i.value)} for i in out.interrupts}
        out = await agent.ainvoke(Command(resume=resume), config, version="v2")
    return out.value


async def arun_and_show(agent, task: str, after: Callable[[dict], None] | None = None,
                        trace: dict | None = None, label: str = "",
                        tools=grounding.VIRA_TOOLS, thread_id: str | None = None) -> dict:
    counter = UsageCounter()
    t0 = time.monotonic()
    result = await arun_task(agent, task, callbacks=[counter], thread_id=thread_id)
    return _report(result, counter, t0, task=task, after=after, trace=trace,
                   label=label, tools=tools)


def repl(label: str, agent, args: argparse.Namespace,
         after: Callable[[dict], None] | None = None) -> None:
    tools = cli_toolset(args).names
    trace = None
    if getattr(args, "trace_json", None):
        trace = {"path": args.trace_json, "model": os.environ.get("CHAT_MODEL", "gpt-5-mini"),
                 "mode": args.mode, "repeat": 1, "tasks": {}, "runs": [],
                 "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.task:
        run_and_show(agent, args.task, after, trace, label, tools)
        return
    print(f"Recruiter agent ({label}, mode={args.mode}). Type a task, 'new' to start a new "
          f"conversation, or 'quit'.")
    thread_id = uuid.uuid4().hex         # one conversation: each reply sees the earlier turns
    while True:
        try:
            task = input("\ninput task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not task or task.lower() in {"quit", "exit"}:
            print("bye")
            return
        if task.lower() == "new":
            thread_id = uuid.uuid4().hex
            print("(new conversation)")
            continue
        try:
            run_and_show(agent, task, after, trace, label, tools, thread_id)
        except Exception as exc:
            print(printable(f"[error] {type(exc).__name__}: {exc}"))


def _last_ai_text(messages) -> str:
    for m in reversed(messages):
        if m.type == "ai" and not m.tool_calls and m.text:
            return m.text
    return ""


def read_status(messages) -> tuple[str, str]:
    """(verdict, detail) from the agent's last prose reply.

    verdict: "done" | "needs_user" | "continue".  Tolerant of a forgetful model:
    a reply with no STATUS line is treated as "done" (it stopped calling tools), which
    is the safe default -- the next user message then starts a fresh task.
    """
    text = _last_ai_text(messages)
    if not text:
        return "continue", ""          # still mid-loop (last message was a tool call)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("STATUS:"):
            body = line[len("STATUS:"):].strip()
            if body.startswith("needs_user"):
                return "needs_user", body.split(":", 1)[1].strip() if ":" in body else ""
            return "done", ""
    return "done", ""                  # prose, no STATUS: assume the task ended


def extract_record(messages, read_only: frozenset) -> dict:
    """A compact, deterministic record of one finished task -- no LLM.

    Walks the thread in order and reports, for each value used in a data-CHANGING call,
    where it came from (user text vs an earlier tool result), plus the actions taken and
    the final outcome.  Read-only resolver/validator calls (find_*, validate_*) are not
    stored as actions; they only serve as provenance for the ids the actions used.
    """
    instruction = next((m.text for m in messages if m.type == "human" and m.text), "")
    # provenance sources, mirroring ToolCallGuard._bad_ids: user text + each tool result
    sources = ([("task", m.text) for m in messages if m.type == "human"]
               + [(f"step {i}", m.text) for i, m in enumerate(messages) if m.type == "tool"])
    actions, resolved, seen_vals = [], [], set()
    for m in messages:
        if m.type != "ai":
            continue
        for tc in m.tool_calls:
            if tc["name"] in read_only:
                continue                               # resolver read: not an action
            actions.append({"tool": tc["name"], "args": tc.get("args", {})})
            for arg, entries in grounding.ground_args(tc.get("args", {}), sources).items():
                for p in entries:
                    key = (arg, json.dumps(p["value"], default=str))
                    if key in seen_vals:
                        continue
                    seen_vals.add(key)
                    src = p["sources"]
                    origin = ("unknown" if not src
                              else "user" if any(s == "task" for s in src)
                              else "system (looked up)")
                    resolved.append({"field": arg, "value": p["value"], "from": origin})
    return {"instruction": instruction, "resolved": resolved, "actions": actions,
            "outcome": _last_ai_text(messages)}


def memory_digest(records: list[dict], *, keep_last: int = 5, max_age_sec: int = 1800,
                  hard_cap: int = 15) -> str:
    """A short text block of recent finished tasks, injected into a new task as user text.

    A record is kept if it is newer than `max_age_sec` (default 30 min) OR among the last
    `keep_last` (default 5) -- so a quiet session still shows the last few, and a busy one
    keeps everything recent, up to `hard_cap` (newest win) so the context can't blow up.

    Because ToolCallGuard grounds ids against the user's words, any id named here becomes
    usable in the new task (e.g. 'add sql to that same job 501') without being refused.
    """
    if not records:
        return ""
    now = time.time()
    n = len(records)
    kept = [r for i, r in enumerate(records)
            if (now - r.get("ts", 0) <= max_age_sec) or i >= n - keep_last]
    kept = kept[-hard_cap:]                              # ceiling; keep the newest
    kept = [r for r in kept if r.get("actions")]         # a task that did nothing isn't memory
    if not kept:
        return ""

    def _clean(text: str) -> str:
        # strip control chars (e.g. a pasted "\r") and collapse whitespace
        return " ".join((text or "").split())

    def _tag(origin: str) -> str:
        return "from user" if origin == "user" else "looked up" if "system" in origin else origin

    lines = ["[earlier in this session - for reference]"]
    for i, r in enumerate(kept, 1):
        did = ", ".join(a["tool"] for a in r["actions"]) or "(none)"
        # group resolved values by (field, origin) so 8 app_ids become one line, not eight
        groups: dict[tuple, list] = {}
        for v in r.get("resolved", []):
            groups.setdefault((v["field"], _tag(v["from"])), []).append(v["value"])
        val_parts = [f"{field} = {', '.join(str(x) for x in vals)} ({tag})"
                     for (field, tag), vals in groups.items()]
        lines.append(f"{i}. Instruction: {_clean(r['instruction'])}")
        lines.append(f"   Did: {did}")
        if val_parts:
            lines.append(f"   Values: {'; '.join(val_parts)}")
    return "\n".join(lines)


async def arepl(label: str, agent, args: argparse.Namespace, tools,
                after: Callable[[dict], None] | None = None) -> None:
    """Async twin of repl for toolsets with async tools (db/jeni_db).

    `tools` is the Toolset (so read_only is available for the finished-task record); a bare
    name set is also accepted (then no record filtering / memory).
    """
    names = getattr(tools, "names", tools)              # grounding wants the name set
    read_only = getattr(tools, "read_only", frozenset())
    trace = None
    if getattr(args, "trace_json", None):
        trace = {"path": args.trace_json, "model": os.environ.get("CHAT_MODEL", "gpt-5-mini"),
                 "mode": args.mode, "repeat": 1, "tasks": {}, "runs": [],
                 "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.task:
        await arun_and_show(agent, args.task, after, trace, label, names)
        return
    print(f"Recruiter agent ({label}, mode={args.mode}). Type a task, 'new' to start a new "
          f"conversation, or 'quit'.")
    records: list[dict] = []                            # finished-task memory, this session
    thread_id = uuid.uuid4().hex
    awaiting = False                                    # are we mid-task, waiting on the user?
    instruction = ""                                    # the task's ORIGINATING user text
    while True:
        try:
            user = input("\ninput task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not user or user.lower() in {"quit", "exit"}:
            print("bye")
            return
        if user.lower() == "new":
            thread_id = uuid.uuid4().hex
            awaiting = False
            print("(new conversation)")
            continue
        # A fresh task (not a reply to a pending question) carries the memory digest in,
        # as user text, so past ids are both visible to the model and grounded for the guard.
        # Keep the ORIGINATING instruction separately, so the record stores what the user
        # actually typed -- not the injected digest (which would nest, digest-in-digest).
        task = user
        if not awaiting:
            instruction = user                         # first turn of this task
            digest = memory_digest(records)
            if digest:
                task = f"{digest}\n\n{user}"
        try:
            result = await arun_and_show(agent, task, after, trace, label, names,
                                         thread_id)
        except Exception as exc:
            print(printable(f"[error] {type(exc).__name__}: {exc}"))
            continue
        verdict, detail = read_status(result.get("messages", []))
        if verdict == "needs_user":
            awaiting = True                            # keep the thread; next input continues it
        else:                                          # done (or treated as done)
            record = extract_record(result.get("messages", []), read_only)
            record["instruction"] = instruction        # the real instruction, never the digest
            record["ts"] = time.time()                 # completion time, for memory_digest's age window
            records.append(record)
            thread_id = uuid.uuid4().hex               # refresh: next task starts clean
            awaiting = False
