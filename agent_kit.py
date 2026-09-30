"""Shared pieces for the LangChain/LangGraph runners (run_langgraph.py, run_deepagent.py).

What run_mini.py spells out as bash-era prompt rules becomes structure here:
  * SYSTEM_PROMPT keeps only the domain rules; the loop ends when the model
    answers in prose, so there is no COMPLETE_TASK echo, SUMMARY quoting or
    tool_choice="required".
  * ToolCallGuard refuses exact repeats of a VIRA call (the duplicate real API
    call run_mini.py works around) and turns crashes into error results.
  * ModelCallLimitMiddleware caps model calls per task (mini's step_limit).
  * --approve-all puts a human in front of every call that changes data (LangGraph interrupt);
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
import re

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
- A person may approve, edit or reject a call before it runs. An edit or a rejection is their
  decision: report what ran as the outcome, not as a failure, and don't redo what they removed
  or rejected, or offer to, with the same tool or another.
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


def _failed(message) -> bool:
    """A tool result that reports a failure: an error status, in the message or its JSON."""
    if getattr(message, "status", None) == "error":
        return True
    try:
        return json.loads(message.text).get("status") == "error"
    except (ValueError, AttributeError):
        return False


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
                 user_only=frozenset(), read_only=frozenset()):
        super().__init__()
        self.names = set(names)
        self.ledger = ledger
        self.user_only = set(user_only)
        self.read_only = set(read_only)

    def _repeat_of_earlier_call(self, request) -> bool:
        """An identical call earlier in the conversation.  It counts as new again once a
        write ran after it, if it is a read (the write may have changed what it reads) or if
        it failed (the write may have fixed why): "publish" fails on a closed job, "reopen
        it, then publish" retries it.  A call that failed or was refused also counts as new
        once the user has written since.  A write that succeeded is never repeated."""
        call = request.tool_call
        key = _normalise(call["args"])
        is_read = call["name"] in self.read_only
        messages = request.state.get("messages", [])
        failed = {m.tool_call_id for m in messages if m.type == "tool" and _failed(m)}
        repeat = earlier_failed = False
        for msg in messages:
            if msg.type == "human" and repeat and earlier_failed:
                repeat = False
            for earlier in getattr(msg, "tool_calls", None) or []:
                if earlier.get("id") == call.get("id"):
                    return repeat         # everything after this is not earlier
                if earlier["name"] == call["name"] and _normalise(earlier["args"]) == key:
                    repeat, earlier_failed = True, earlier.get("id") in failed
                elif repeat and (is_read or earlier_failed) and earlier["name"] not in self.read_only:
                    repeat = False
        return repeat

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
        if request.tool_call["name"] in overruled(request.state.get("messages", [])):
            return self._result(request, f"Refused: the reviewer already edited or rejected "
                                f"{request.tool_call['name']} in this request, and that decision "
                                "stands. Report what ran; don't redo it or offer to. The user "
                                "will ask if they want more.")
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
    return [ToolCallGuard(names=toolset.names, ledger=ledger, user_only=toolset.user_only,
                          read_only=toolset.read_only),
            ModelCallLimitMiddleware(run_limit=step_limit, exit_behavior="end")]


DECISIONS = {"allowed_decisions": ["approve", "edit", "reject"]}
# How HumanInTheLoopMiddleware marks a reviewer's decision at the start of a tool result
# (tests check these against the middleware's own messages).
REVIEWER_EDITED = "Note: a human reviewer replaced this tool call before it ran."
REVIEWER_REJECTED = "User rejected the tool call for "


def overruled(messages) -> set[str]:
    """Tools whose call a reviewer edited or rejected since the user last wrote.

    The decision stands until the user writes again: until then a new call to the same
    tool gets no approval card (interrupt_on's `when`) and ToolCallGuard refuses it, so the
    agent can't redo what the reviewer removed.
    """
    names = set()
    for m in reversed(messages):
        if m.type == "human":
            break
        if m.type == "tool" and m.text.startswith((REVIEWER_EDITED, REVIEWER_REJECTED)):
            names.add(m.name)
    return names


def _ask_unless_overruled(request) -> bool:
    return request.tool_call["name"] not in overruled(request.state.get("messages", []))

# Tools that ALWAYS pause for a human, in any mode and even without --approve-all, because
# acting on the wrong ones is costly/irreversible (bulk actions on candidates, ownership, etc.).
# Only names that exist in the active toolset are gated, so this is safe for vira/jeni/jeni_db.
ALWAYS_CONFIRM = frozenset({
    "shortlist_multiple_application",
    "reject_multiple_application",
    "transfer_job_ownership",
    "share_application",
})


def interrupt_on(approve_all: bool, mode: str | None = None, toolset: Toolset = VIRA) -> dict:
    """Which tools pause for a human.

    Read-only tools (GET-style VIRA calls, db lookups/validations in toolset.read_only)
    NEVER pause: they change nothing, so approving them is just noise.  So:
      * --approve-all: every data-CHANGING tool (all names minus read_only);
      * real mode: the same data-changing set (unsafe to skip on real VIRA);
      * otherwise (mock, no --approve-all): nothing.
    ON TOP of that, tools in ALWAYS_CONFIRM always pause (any mode), if present in the
    toolset -- high-stakes actions a human should see every time.
    `mode` defaults to the configured one (vira_tools.configure), so a real-mode agent
    can't be built without the write gates.  No card for a tool the reviewer already
    overruled (overruled()).
    """
    if approve_all:
        names = toolset.names - toolset.read_only
    elif (mode or vira_tools.current_mode()) == "real":
        names = toolset.names - toolset.read_only
    else:
        names = set()
    names |= (ALWAYS_CONFIRM & toolset.names)      # always-confirm, whatever the mode
    return {name: {**DECISIONS, "when": _ask_unless_overruled} for name in sorted(names)}


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


_INTERPRET_PROMPT = (
    "Interpret the user's latest reply about a pending action. Return ONLY one JSON object, "
    "one of these shapes:\n"
    '  {"decision":"approve"}\n'
    '  {"decision":"reject"}\n'
    '  {"decision":"update","args":{...}}\n'
    '  {"decision":"clarify","message":"..."}\n'
    "Context you are given: `action`, `original_args` (the ONLY allowed values), "
    "`current_args` (the proposed selection so far), and `latest_user_reply`.\n"
    "Rules:\n"
    "- approve: accept current_args unchanged (e.g. 'yes', 'looks good'). If nothing changes, approve.\n"
    "- reject: cancel the action (e.g. 'no', 'cancel').\n"
    "- update: return the COMPLETE replacement args (same arg names).\n"
    "- clarify: ask only if the instruction is unclear, or names a value not in original_args.\n"
    "- List values are ORDERED SETS: never repeat a value.\n"
    "- 'add <ids>' adds values not already selected; 'remove/drop/except <ids>' removes values.\n"
    "- 'only/keep/select/first N/last N/top N' REPLACES the list (first/last/top use "
    "original_args order).\n"
    "- Every value must come from original_args; never invent one.\n"
    "- Do NOT perform the action. The script handles confirmation and execution.")


def _confirm_payload(action: dict, original_args: dict, current_args: dict, reply: str) -> str:
    """The user-message payload for the confirm model: action, allowed set, current, reply."""
    return json.dumps({"action": action, "original_args": original_args,
                       "current_args": current_args, "latest_user_reply": reply},
                      ensure_ascii=False)


CONFIRM_USAGE = None        # UsageCounter for the CURRENT confirmation (set per [approval])
CONFIRM_TURN = None         # UsageCounter summing all confirmations in the whole turn


def _interpret(history: list, model) -> dict | None:
    """Send the confirmation conversation to the model; return its decision dict or None.

    `history` is the running message list for THIS confirmation only (system prompt, the
    original call, and each instruction/result since) -- so the model has the original ids
    and the changes so far in context, and validates against them itself (no code guard).

    On failure prints a short reason (set JENI_DEBUG=0 to silence) and returns None."""
    if model is None:
        if os.environ.get("JENI_DEBUG", "1") != "0":
            print("  [interpret] no confirmation model (build failed or CHAT_MODEL unset)")
        return None
    cfg = {"callbacks": [CONFIRM_USAGE]} if CONFIRM_USAGE is not None else {}
    try:
        reply = model.invoke(history, config=cfg)
        # LangChain chat reply -> text. .text is a property; fall back to .content.
        raw = getattr(reply, "text", "") or ""
        if not raw:
            content = getattr(reply, "content", reply)
            if isinstance(content, list):      # content parts -> join the text pieces
                content = "".join(p.get("text", "") if isinstance(p, dict) else str(p)
                                  for p in content)
            raw = str(content)
        raw = raw.strip()
        brace = raw[raw.find("{"): raw.rfind("}") + 1] if "{" in raw else ""
        out = json.loads(brace) if brace else None
        if isinstance(out, dict) and "decision" in out:
            return out
        if os.environ.get("JENI_DEBUG", "1") != "0":
            print(f"  [interpret] reply had no usable decision; got: {raw[:200]!r}")
        return None
    except Exception as exc:
        if os.environ.get("JENI_DEBUG", "1") != "0":
            print(f"  [interpret] {type(exc).__name__}: {exc}")
        return None


def ask_human(request: dict, model=None) -> list[dict]:
    """One decision per pending tool call: approve, reject, or edit its args.

    Fast paths (no model): press Enter or 'y' to approve, 'n' to reject.  Any other text is
    a free-text instruction sent -- with this confirmation's history -- to the interpreter
    model, which applies it to the current proposal (add/remove/select) or asks back if a
    value isn't in the original call.  Each change is shown; nothing runs until Enter/y.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    global CONFIRM_USAGE, CONFIRM_TURN
    model = model or confirm_chat_model()
    decisions = []
    for action in request["action_requests"]:
        base = {"name": action["name"], "args": action.get("args", {})}
        proposal = dict(base["args"])
        # history for THIS confirmation only: system rules + the original call
        history = [SystemMessage(_INTERPRET_PROMPT)]   # per-turn payloads carry the args + reply
        conf = UsageCounter()          # tokens/calls for THIS confirmation
        CONFIRM_USAGE = conf           # _interpret counts into the current confirmation
        calls = 0                      # interpreter calls so far, this confirmation (cap 5)
        MAX_CALLS = 5
        print(printable(f"\n[approval] {base['name']}"
                        f"({json.dumps(proposal, ensure_ascii=False)})"))
        while True:
            answer = input("approve? [Enter/y]es  [n]o  or say what to change > ").strip()
            low = answer.lower()
            if low in {"", "y", "yes"}:
                decisions.append({"type": "approve"} if proposal == base["args"] else
                                 {"type": "edit",
                                  "edited_action": {"name": base["name"], "args": proposal}})
                break
            if low in {"n", "no"}:
                decisions.append({"type": "reject", "message": "The user declined this call."})
                break
            if calls >= MAX_CALLS:     # cap reached without a yes/no -> bail to the main agent
                print(f"  reached {MAX_CALLS} attempts without a decision; abandoning this call.")
                decisions.append({"type": "reject",
                                  "message": f"Approval abandoned: the change could not be "
                                             f"confirmed after {MAX_CALLS} attempts."})
                break
            # free text -> interpreter, with the running history as context
            history.append(HumanMessage(_confirm_payload(
                base["name"], base["args"], proposal, answer)))
            out = _interpret(history, model)
            calls += 1
            if out is None:
                print("  (couldn't reach the interpreter) -- press Enter to run, 'n' to reject.")
                history.pop()
                continue
            history.append(AIMessage(json.dumps(out, ensure_ascii=False)))
            decision = out.get("decision")
            if decision == "reject":
                decisions.append({"type": "reject", "message": "The user declined this call."})
                break
            if decision == "approve":
                decisions.append({"type": "approve"} if proposal == base["args"] else
                                 {"type": "edit",
                                  "edited_action": {"name": base["name"], "args": proposal}})
                break
            if decision == "clarify":                   # the model needs the user to clarify
                print(printable(f"  {out.get('message', 'please clarify')}"))
                continue
            if decision == "update" and isinstance(out.get("args"), dict):
                # safety net: drop duplicates (order-preserving) and any value not in the
                # ORIGINAL call -- the model should already do this, but guarantee it.
                cleaned = {}
                for k, v in out["args"].items():
                    if isinstance(v, list):
                        orig = base["args"].get(k)
                        allowed = list(orig) if isinstance(orig, list) else None
                        seen, out_list = set(), []
                        for x in v:
                            if x in seen:
                                continue
                            if allowed is not None and x not in allowed:
                                continue          # not offered in the original call -> drop
                            seen.add(x)
                            out_list.append(x)
                        cleaned[k] = out_list
                    else:
                        cleaned[k] = v
                proposal = cleaned
                print(printable(f"  updated -> {base['name']}"
                                f"({json.dumps(proposal, ensure_ascii=False)})"))
                print("  press Enter to run this, 'n' to reject, or say another change.")
                continue
            print("  couldn't apply that -- press Enter to run, 'n' to reject, or rephrase.")
        if conf.calls:                 # per-confirmation usage line, when this [approval] closes
            print(f"  (confirmation {_confirm_model_id()}: {conf.calls} call(s), "
                  f"tokens in/out: {conf.input_tokens}/{conf.output_tokens})")
            if CONFIRM_TURN is not None:          # add into the whole-turn total
                CONFIRM_TURN.calls += conf.calls
                CONFIRM_TURN.input_tokens += conf.input_tokens
                CONFIRM_TURN.output_tokens += conf.output_tokens
    return decisions


CONFIRM_MODEL = os.environ.get("CONFIRM_MODEL", "gpt-4o-mini")  # small model for confirmations


@functools.lru_cache(maxsize=1)
def confirm_chat_model():
    """The dedicated chat model for interrupt confirmations (no tools). Small/cheap by
    default (CONFIRM_MODEL, e.g. gpt-5-mini or gpt-4o-mini), separate from the agent model."""
    try:
        return build_chat_model(CONFIRM_MODEL)
    except Exception:
        return None


def _confirm_model_id() -> str:
    """The confirmation model's id, for the usage line."""
    return CONFIRM_MODEL


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
TOOLS_HELP = {"jeni": "Jeni's own tasks (needs config/jeni_tasks.json)",
              "jeni_db": "jeni plus read-only db lookups, so ids are resolved and checked "
                         "before acting",
              "vira": "the four sample AI endpoints"}


def parser(description: str, *, toolsets: tuple = TOOLSET_NAMES,
           default_tools: str = "jeni_db") -> argparse.ArgumentParser:
    """The runners' shared flags.  A runner without async tools passes toolsets without jeni_db."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", choices=["real", "mock"], default="mock",
                   help="mock (default): local fake VIRA; real: call VIRA at $VIRA_BASE_URL")
    p.add_argument("--task", help="run this one task and exit (default: interactive prompt)")
    p.add_argument("--tools", choices=toolsets, default=default_tools,
                   help="; ".join(f"{name}{' (default)' if name == default_tools else ''}: "
                                  f"{TOOLS_HELP[name]}" for name in toolsets))
    if "jeni_db" in toolsets:
        p.add_argument("--dsn", default=os.environ.get("TRON_POSTGRES_DSN"),
                       help="Postgres DSN for jeni_db in --mode real (default: "
                            "$TRON_POSTGRES_DSN). In mock mode the lookups answer from mock "
                            "VIRA's synthetic data instead, so their ids match its tasks.")
        p.add_argument("--company-id", type=int, default=5143,
                       help="authenticated company_id injected into db queries (tenant scope); "
                            "never taken from the model")
    p.add_argument("--approve-all", action="store_true",
                   help="pause for human approval before every tool call that changes data "
                        "(in real mode, score/insights always pause; reads never do)")
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
    if CONFIRM_TURN is not None and CONFIRM_TURN.calls:
        print(f"(confirmation model {_confirm_model_id()} total: {CONFIRM_TURN.calls} call(s), "
              f"tokens in/out: {CONFIRM_TURN.input_tokens}/{CONFIRM_TURN.output_tokens})")
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
    global CONFIRM_USAGE, CONFIRM_TURN
    CONFIRM_USAGE = UsageCounter()        # current confirmation (reset again per [approval])
    CONFIRM_TURN = UsageCounter()         # fresh per turn; sums all confirmations this turn
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
    global CONFIRM_USAGE, CONFIRM_TURN
    CONFIRM_USAGE = UsageCounter()        # current confirmation (reset again per [approval])
    CONFIRM_TURN = UsageCounter()         # fresh per turn; sums all confirmations this turn
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


_EXECUTED_RE = re.compile(r"[Ee]xecuted instead:[^\n]*?with arguments\s*(\{[^\n]*\})")


def _result_ids(tool_msg_text: str) -> list:
    """The ids a read-only resolver returned, for the Resolve trail.  Handles the common
    shapes: a top-level list of ints under 'applications'/'valid_values', or selection
    'candidates' of {..._id: N}.  Best-effort: returns [] if nothing id-like is found."""
    text = tool_msg_text or ""
    try:
        data = json.loads(text[: text.rfind("}") + 1])
    except (ValueError, AttributeError):
        return []
    if not isinstance(data, dict):
        return []
    res = data.get("result", data)
    if not isinstance(res, dict):
        res = data
    for key in ("applications", "valid_values"):
        v = res.get(key) if isinstance(res, dict) else None
        if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
            return v
    sel = res.get("selection") if isinstance(res, dict) else None
    if isinstance(sel, dict) and isinstance(sel.get("candidates"), list):
        ids = [c[k] for c in sel["candidates"] if isinstance(c, dict)
               for k in c if k.endswith("_id") and isinstance(c[k], int)]
        if ids:
            return ids
    return []


def _executed_args(tool_msg_text: str) -> dict | None:
    """If a tool result carries the HITL 'Executed instead: ... with arguments {...}' note,
    return those executed args; else None.  This is how an interrupt EDIT is recovered: the
    agent's own message still shows its ORIGINAL call, but the edited call is what ran."""
    if not tool_msg_text or "xecuted instead" not in tool_msg_text:
        return None
    m = _EXECUTED_RE.search(tool_msg_text)
    if not m:
        return None
    try:
        args = json.loads(m.group(1))
        return args if isinstance(args, dict) else None
    except ValueError:
        return None


def extract_record(messages, read_only: frozenset) -> dict:
    """A compact, deterministic record of one finished task -- no LLM.

    Reports, for each data-CHANGING call, the args that ACTUALLY RAN (an interrupt edit
    replaces the agent's proposed args, recovered from the tool result's 'Executed instead'
    note), where each value came from, and the final outcome.  Read-only resolver/validator
    calls are not stored as actions; they only provide provenance for the ids used.
    """
    instruction = next((m.text for m in messages if m.type == "human" and m.text), "")
    sources = ([("task", m.text) for m in messages if m.type == "human"]
               + [(f"step {i}", m.text) for i, m in enumerate(messages) if m.type == "tool"])
    # pair each ai tool call with the tool message that follows it, to find an edit note
    resolve, actions, resolved, seen_vals = [], [], [], set()
    for i, m in enumerate(messages):
        if m.type != "ai":
            continue
        for tc in m.tool_calls:
            nxt = next((messages[j] for j in range(i + 1, len(messages))
                        if messages[j].type == "tool"), None)
            if tc["name"] in read_only:
                # a lookup/validation step: record HOW things were found (and what it returned)
                resolve.append({"tool": tc["name"], "args": tc.get("args", {}),
                                "found": _result_ids(nxt.text) if nxt is not None else []})
                continue
            args = tc.get("args", {})
            edited = False
            if nxt is not None:                   # HITL edit -> the executed args, not the proposed
                executed = _executed_args(nxt.text)
                if executed is not None:
                    args = executed
                    edited = True                 # a human chose these at the approval step
            actions.append({"tool": tc["name"], "args": args})
            for arg, entries in grounding.ground_args(args, sources).items():
                for p in entries:
                    key = (arg, json.dumps(p["value"], default=str))
                    if key in seen_vals:
                        continue
                    seen_vals.add(key)
                    src = p["sources"]
                    if edited:
                        origin = "user (reviewer-selected)"   # chosen by the human at approval
                    elif not src:
                        origin = "unknown"
                    elif any(s == "task" for s in src):
                        origin = "user"
                    else:
                        origin = "system (looked up)"
                    resolved.append({"field": arg, "value": p["value"], "from": origin})
    return {"instruction": instruction, "resolve": resolve, "resolved": resolved,
            "actions": actions, "outcome": _last_ai_text(messages)}


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
        return ("from user" if origin == "user"
                else "reviewer-selected" if "reviewer" in origin
                else "looked up" if "system" in origin else origin)

    lines = ["[earlier in this session - for reference]"]
    for i, r in enumerate(kept, 1):
        did = ", ".join(a["tool"] for a in r["actions"]) or "(none)"
        resolve_parts = []
        for a in r.get("resolve", []):
            found = a.get("found") or []
            if found:
                shown = ", ".join(str(x) for x in found[:12]) + ("…" if len(found) > 12 else "")
                resolve_parts.append(f"{a['tool']} -> [{shown}]")
            else:
                resolve_parts.append(a["tool"])
        resolve = "; ".join(resolve_parts)
        # group resolved values by (field, origin) so 8 app_ids become one line, not eight
        groups: dict[tuple, list] = {}
        for v in r.get("resolved", []):
            groups.setdefault((v["field"], _tag(v["from"])), []).append(v["value"])
        val_parts = [f"{field} = {', '.join(str(x) for x in vals)} ({tag})"
                     for (field, tag), vals in groups.items()]
        lines.append(f"{i}. Instruction: {_clean(r['instruction'])}")
        if resolve:
            lines.append(f"   Resolve: {resolve}")
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