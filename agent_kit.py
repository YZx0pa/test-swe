"""Shared pieces for the LangChain/LangGraph runners (run_langgraph.py, run_deepagent.py).

What run_mini.py spells out as bash-era prompt rules becomes structure here:
  * SYSTEM_PROMPT keeps only the domain rules; the loop ends when the model
    answers in prose, so there is no COMPLETE_TASK echo, SUMMARY quoting or
    tool_choice="required".
  * ToolCallGuard refuses exact repeats of a VIRA call (the duplicate real API
    call run_mini.py works around) and turns crashes into error results.
  * ModelCallLimitMiddleware caps model calls per task (mini's step_limit).
  * --approve-all puts a human in front of every VIRA call (LangGraph interrupt);
    in real mode, calls that change data on VIRA (score, insights) always pause.

Never import run_mini from here: it reads .env and pulls in minisweagent and
litellm on import.
"""
import argparse
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

import grounding
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
"""


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

    def __init__(self, names=frozenset(vira_tools.NAMES), ledger: CallLedger | None = None):
        super().__init__()
        self.names = set(names)
        self.ledger = ledger

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
        that is in neither the task nor any earlier tool result.  A reviewer's edited args
        (HumanInTheLoopMiddleware) count as user input, like the task.
        """
        messages = request.state.get("messages", [])
        task = next((m.text for m in messages if m.type == "human"), "")
        sources = [("task", task)] + [(f"step {i}", m.text) for i, m in enumerate(messages)
                                      if m.type == "tool"]
        edited = (request.state.get("hitl_edited_tool_calls") or {}).get(request.tool_call.get("id"))
        if edited:
            sources.append(("task", json.dumps(edited.get("args", {}))))
        misused, invented = [], []
        for arg, entries in grounding.ground_args(request.tool_call["args"], sources).items():
            if arg not in grounding.ID_KINDS:
                continue
            for p in entries:
                if "misused_as" in p:
                    misused.append(f"{p['value']} is a {p['misused_as']}, not a {arg[:-1]}")
                elif not p["sources"]:
                    invented.append(f"{p['value']}")
        return misused, invented

    def _refusal(self, request) -> ToolMessage | None:
        misused, invented = self._bad_ids(request)
        if misused:
            return self._result(request, "Refused: " + "; ".join(misused) + ". No tool converts "
                                "between id kinds: finish and say which id is missing.")
        if invented:
            return self._result(request, f"Refused: {', '.join(invented)} isn't in the task or any "
                                "earlier result. Never invent ids: finish and say which id is missing.")
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


def middleware(step_limit: int = 12, ledger: CallLedger | None = None) -> list:
    """Guard + a per-thread cap on model calls (each task runs on a fresh thread).

    Pass one shared ledger to an agent and all its subagents.
    """
    return [ToolCallGuard(ledger=ledger),
            ModelCallLimitMiddleware(thread_limit=step_limit, exit_behavior="end")]


DECISIONS = {"allowed_decisions": ["approve", "edit", "reject"]}


def interrupt_on(approve_all: bool, mode: str | None = None) -> dict:
    """Which tools pause for a human: every one with --approve-all; in real mode, always
    the ones that trigger calculations on VIRA (not in READ_ONLY).  `mode` defaults to
    the configured one (vira_tools.configure), so a real-mode agent can't be built
    without them."""
    if approve_all:
        names = vira_tools.NAMES
    elif (mode or vira_tools.current_mode()) == "real":
        names = vira_tools.NAMES - vira_tools.READ_ONLY
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
             callbacks: list | None = None) -> dict:
    """Run one task on a fresh thread, pausing for `decide` at every interrupt."""
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "callbacks": callbacks or []}
    out = agent.invoke({"messages": [{"role": "user", "content": task}]}, config, version="v2")
    while out.interrupts:
        if len(out.interrupts) == 1:
            resume: Any = {"decisions": decide(out.interrupts[0].value)}
        else:                            # e.g. parallel subagents, each paused on its own call
            resume = {i.id: {"decisions": decide(i.value)} for i in out.interrupts}
        out = agent.invoke(Command(resume=resume), config, version="v2")
    return out.value


def show(messages) -> None:
    """Trajectory printout in the style of run_mini.show(); control characters removed."""
    for i, m in enumerate(messages):
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
               seconds: float) -> dict:
    """One run in compare_agents.py's --json format, so the page reads both."""
    steps = grounding.trace_from_messages(result["messages"], task)
    return {"runner": label, "task": task, "repeat": 1, "passed": None, "error": "",
            "model_calls": counter.calls, "input_tokens": counter.input_tokens,
            "output_tokens": counter.output_tokens, "seconds": round(seconds, 1), "cost": None,
            "audit": [], "steps": steps, "grounding": grounding.summary(steps),
            "final": final_text(result)}


def run_and_show(agent, task: str, after: Callable[[dict], None] | None = None,
                 trace: dict | None = None, label: str = "") -> dict:
    counter = UsageCounter()
    t0 = time.monotonic()
    result = run_task(agent, task, callbacks=[counter])
    print("\n=== trajectory ===")
    show(result["messages"])
    if after:
        after(result)
    print(f"\n(model calls: {counter.calls}, tokens in/out: "
          f"{counter.input_tokens}/{counter.output_tokens})")
    if trace is not None:
        trace["runs"].append(_trace_run(label, task, result, counter, time.monotonic() - t0))
        trace["tasks"][task] = {"text": task, "check": ""}
        write_private(trace["path"], json.dumps({k: v for k, v in trace.items() if k != "path"},
                                                ensure_ascii=False, indent=1))
        print(f"(trace written to {trace['path']})")
    return result


def repl(label: str, agent, args: argparse.Namespace,
         after: Callable[[dict], None] | None = None) -> None:
    trace = None
    if getattr(args, "trace_json", None):
        trace = {"path": args.trace_json, "model": os.environ.get("CHAT_MODEL", "gpt-5-mini"),
                 "mode": args.mode, "repeat": 1, "tasks": {}, "runs": [],
                 "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.task:
        run_and_show(agent, args.task, after, trace, label)
        return
    print(f"Recruiter agent ({label}, mode={args.mode}). Type a task, or 'quit'.")
    while True:
        try:
            task = input("\ninput task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not task or task.lower() in {"quit", "exit"}:
            print("bye")
            return
        try:
            run_and_show(agent, task, after, trace, label)
        except Exception as exc:
            print(printable(f"[error] {type(exc).__name__}: {exc}"))
