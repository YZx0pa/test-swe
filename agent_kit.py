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
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
import re

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command
from pydantic import BaseModel, Field

import grounding
import jeni_tools
import pii_vault
import recruiter_cli
# Re-exported for existing runners/tests; implementation lives in task_memory.py.
from task_memory import (TaskMemoryMiddleware, active_task_messages as _active_task_messages,
                         extract_record, memory_digest, packed_task_messages)
import vira_tools
from terminal import printable


STAGE_LOG = logging.getLogger("jeni.stages")


def stage(event: str, **details) -> None:
    """Write a compact, non-PII execution-stage record for developers."""
    values = " ".join(f"{key}={value}" for key, value in sorted(details.items())
                      if value is not None)
    STAGE_LOG.info("%s%s", event, f" {values}" if values else "")


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
- A resolver result with `selection_policy: choose_many` is a list for the user to choose from,
  not permission to act on every item. When the user asks to list applicants, show the list and
  stop. Only shortlist selected application ids, or all applicants when the user explicitly says
  "shortlist all".
- If a tool returns `user_action_required: true`, explain its message and ask the user to correct
  that field. Do not silently retry a write using a previous or guessed value.
- When you are done, return the terminal response schema. Put the user-facing explanation in
  `message`. Set `status` to `done` when the task is finished (including when it cannot
  proceed), or `needs_user` only when you are genuinely blocked on the user. Otherwise, if
  more tool calls are needed, make them instead of returning a terminal response.
"""


class TerminalResponse(BaseModel):
    """The only allowed final result once the agent has stopped calling tools."""

    message: str = Field(description="Concise, user-facing outcome or request for information.")
    status: Literal["done", "needs_user"] = Field(
        description="done when this task is complete; needs_user only when user input is required."
    )


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


def _without_tools(toolset: Toolset, disabled: tuple[str, ...] | list[str]) -> Toolset:
    """Development-only filter that keeps the exposed tool list and its metadata aligned."""
    disabled_names = frozenset(disabled)
    if not disabled_names:
        return toolset

    exposed = {tool.name for tool in toolset.tools()}
    unknown = disabled_names - exposed
    if unknown:
        raise ValueError(f"cannot disable tool(s) not exposed by {toolset.name}: {sorted(unknown)}")

    return Toolset(
        toolset.name,
        lambda: [tool for tool in toolset.tools() if tool.name not in disabled_names],
        toolset.names - disabled_names,
        toolset.read_only - disabled_names,
        toolset.user_only - disabled_names,
        toolset.prompt,
    )


def toolset(name: str, *, query_tools=None, context=None,
            disabled_tools: tuple[str, ...] | list[str] = ()) -> Toolset:
    """VIRA (the sample endpoints), Jeni's own tasks, or jeni+db combined.

    jeni / jeni_db read the internal catalog (config/README.md) on first use.
    db and jeni_db also need `query_tools` (db_queries.build_db_queries(pool) or
    fake_db_queries(fixtures)) and `context` ({"auth_profile": {"company_id": ...}}),
    supplied by the runner's main() -- never by the model.
    """
    if name == "vira":
        base = VIRA
    elif name == "jeni":
        base = _jeni()
    elif name in ("db", "jeni_db"):
        if query_tools is None or context is None:
            raise SystemExit(f"toolset {name!r} needs a db pool/context; the runner must "
                             f"pass query_tools and context (see run_langgraph.py).")
        base = _db(query_tools, context) if name == "db" else _jeni_db(query_tools, context)
    else:
        raise ValueError(f"unknown toolset {name!r}")
    return _without_tools(base, disabled_tools)


def cli_toolset(args: argparse.Namespace, *, query_tools=None, context=None) -> Toolset:
    """toolset(args.tools) for a runner's main(): a missing catalog is a message, not a traceback."""
    try:
        return toolset(getattr(args, "tools", "vira"),
                       query_tools=query_tools, context=context,
                       disabled_tools=getattr(args, "disable_tool", ()))
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
    TaskMemoryMiddleware additionally contributes the current task boundary when
    one UI thread holds several completed tasks.
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
        messages = _active_task_messages(request.state)
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
        messages = _active_task_messages(request.state)
        sources = ([("task", m.text) for m in messages if m.type == "human"]
                   + [(f"step {i}", m.text) for i, m in enumerate(messages) if m.type == "tool"])
        compact = memory_digest(request.state.get("task_memory_records", []))
        if compact:
            sources.append(("earlier completed task", compact))
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
        messages = _active_task_messages(request.state)
        said = " ".join(m.text for m in messages if m.type == "human")
        edited = (request.state.get("hitl_edited_tool_calls") or {}).get(request.tool_call.get("id"))
        if edited:
            said += " " + json.dumps(edited.get("args", {}), ensure_ascii=False)
        said = said.lower()
        bad = []
        for field in fields:
            values = args[field] if isinstance(args[field], list) else [args[field]]
            if any(not isinstance(v, str) or (v.strip().lower() not in said
                                               and not pii_vault.VAULT.allowed(field, v))
                   for v in values if v not in (None, "")):
                bad.append(field)
        return bad

    def _refusal(self, request) -> ToolMessage | None:
        name = request.tool_call["name"]
        # A high-impact action must always return to the reviewer for its next
        # proposed execution.  An earlier edit/rejection must never bypass that
        # card or turn into an automatic retry.
        if name not in ALWAYS_CONFIRM and name in overruled(_active_task_messages(request.state)):
            stage("TOOL_GUARD_REFUSED", tool=name, reason="reviewer_decision_stands")
            return self._result(request, f"Refused: the reviewer already edited or rejected "
                                f"{name} in this request, and that decision stands. Report what "
                                "ran; don't redo it or offer to. The user will ask if they want more.")
        misused, invented = self._bad_ids(request)
        stage("GROUNDING_CHECK", tool=name, result="refused" if (misused or invented) else "passed",
              misused=len(misused), invented=len(invented))
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
            task_start = request.state.get("task_memory_active_start", 0)
            repeat = not self.ledger.claim((info.thread_id if info else None, task_start, call["name"],
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
            stage("TOOL_CALL", tool=request.tool_call["name"], mode="sync")
            result = handler(request)
            stage("TOOL_RESULT", tool=request.tool_call["name"], status=getattr(result, "status", None))
            return result
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
            stage("TOOL_CALL", tool=request.tool_call["name"], mode="async")
            result = await handler(request)
            stage("TOOL_RESULT", tool=request.tool_call["name"], status=getattr(result, "status", None))
            return result
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return self._result(request, f"tool failed ({type(exc).__name__})")


def middleware(step_limit: int = 12, ledger: CallLedger | None = None,
               toolset: Toolset = VIRA, execution_validation: AgentMiddleware | None = None) -> list:
    """Execution validation, guard, and a cap on model calls per run.

    A run is one user turn, or one resume after an approval, so a person starts each one.
    A per-thread cap would end a conversation after `step_limit` calls in total.  Pass one
    shared ledger to an agent and all its subagents.  ``execution_validation`` is the
    post-approval entity DB validator for Jeni; it must run before ToolCallGuard so a
    reviewer-edited payload is checked before grounding/provenance and VIRA execution.
    """
    checks = [execution_validation] if execution_validation is not None else []
    return [*checks, ToolCallGuard(names=toolset.names, ledger=ledger, user_only=toolset.user_only,
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
    name = request.tool_call["name"]
    required = name in ALWAYS_CONFIRM or name not in overruled(_active_task_messages(request.state))
    stage("HUMAN_CHECK", tool=name, required=required,
          policy="always_confirm" if name in ALWAYS_CONFIRM else "normal")
    return required


def _latest_validation_reason(state, tool_name: str) -> str | None:
    """A safe explanation for the next approval card after validation blocked a call."""
    for message in reversed(_active_task_messages(state)):
        if message.type != "tool" or message.name != tool_name:
            continue
        text = message.text.rsplit("Tool response:", 1)[-1]
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue
        if payload.get("status") in {"validation_error", "validation_unavailable"}:
            return str(payload.get("message") or "The previous attempt did not pass validation.")
    return None


def _describe(tool_call, state, runtime) -> str:
    """An approval card's text: what the call would do, in plain words (jeni_tools.summary),
    instead of the middleware's "Tool: … Args: {…}"."""
    description = jeni_tools.summary(tool_call["name"], tool_call.get("args") or {})
    reason = _latest_validation_reason(state, tool_call["name"])
    return description + (f" Previous attempt was not run: {reason}" if reason else "")

# Tools that ALWAYS pause for a human, in any mode and even without --approve-all, because
# acting on the wrong ones is costly/irreversible (bulk actions on candidates, ownership, etc.).
# Every newly proposed execution of one of these tools needs its own approval card.
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
    can't be built without the write gates.  A prior reviewer edit/rejection suppresses
    a normal tool's repeat, while ALWAYS_CONFIRM tools always show another card.
    """
    if approve_all:
        names = toolset.names - toolset.read_only
    elif (mode or vira_tools.current_mode()) == "real":
        names = toolset.names - toolset.read_only
    else:
        names = set()
    names |= (ALWAYS_CONFIRM & toolset.names)      # always-confirm, whatever the mode
    return {name: {**DECISIONS, "when": _ask_unless_overruled, "description": _describe}
            for name in sorted(names)}


def unattended(request: dict) -> list[dict]:
    """Decisions when nobody is at the card (compare_agents): approve on mock VIRA, where nothing
    real changes, so a task that needs a card (a shortlist always does) can finish; reject in any
    other mode."""
    if vira_tools.current_mode() == "mock":
        return [{"type": "approve"} for _ in request["action_requests"]]
    return [{"type": "reject", "message": "Nobody is here to approve this."}
            for _ in request["action_requests"]]


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
        stage("AGENT_ANALYSING", model_call=self.calls)

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
    """Terminal approval fallback: approve, reject, or explicitly replace arguments.

    Edits use the same HITL ``edited_action`` decision as the demo UI.  The terminal accepts
    a complete JSON object only; it intentionally does not interpret free-text changes.
    """
    decisions = []
    for action in request["action_requests"]:
        base = {"name": action["name"], "args": action.get("args", {})}
        print(printable(f"\n[approval] {base['name']}"
                        f"({json.dumps(base['args'], ensure_ascii=False)})"))
        while True:
            low = input("approve? [Enter/y]es  [e]dit JSON  [n]o > ").strip().lower()
            if low in {"", "y", "yes"}:
                decisions.append({"type": "approve"})
                break
            if low in {"e", "edit"}:
                args = _ask_args()
                decisions.append({"type": "edit",
                                  "edited_action": {"name": base["name"], "args": args}})
                break
            if low in {"n", "no"}:
                decisions.append({"type": "reject", "message": "The user declined this call."})
                break
            print("  enter y to approve, e to replace all arguments as JSON, or n to reject.")
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


def terminal_response_data(result: dict) -> dict | None:
    """Normalise create_agent's Pydantic/dict structured terminal result."""
    response = result.get("structured_response")
    if isinstance(response, BaseModel):
        response = response.model_dump()
    if not isinstance(response, dict):
        return None
    message, status = response.get("message"), response.get("status")
    if isinstance(message, str) and status in {"done", "needs_user"}:
        return {"message": message, "status": status}
    return None


def show(messages, start: int = 0, *, terminal: dict | None = None) -> None:
    """Trajectory printout in the style of run_mini.show(); control characters removed.

    Prints messages[start:], numbered as in the whole conversation.
    """
    for i, m in enumerate(messages[start:], start):
        if m.type == "human":
            print(printable(f"[{i}] user      : {m.text}"))
        elif m.type == "ai":
            for tc in m.tool_calls:
                if tc["name"] == TerminalResponse.__name__:
                    continue                    # LangChain's internal structured-output tool
                print(printable(f"[{i}] assistant → call: {tc['name']}"
                                f"({json.dumps(tc['args'], ensure_ascii=False)})"))
            if m.text and not terminal:
                print(printable(f"[{i}] assistant : {m.text}"))
        elif m.type == "tool":
            print(printable(f"[{i}] tool      → {m.text}"))
    if terminal:
        print(printable("[final] assistant : " + terminal["message"]))


def final_text(result: dict) -> str:
    terminal = terminal_response_data(result)
    if terminal:
        return terminal["message"]
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
    p.add_argument("--mode", choices=["real", "mock"], default="real",
                   help="mock (default): local fake VIRA; real: call VIRA at $VIRA_BASE_URL")
    p.add_argument("--task", help="run this one task and exit (default: interactive prompt)")
    p.add_argument("--tools", choices=toolsets, default=default_tools,
                   help="; ".join(f"{name}{' (default)' if name == default_tools else ''}: "
                                  f"{TOOLS_HELP[name]}" for name in toolsets))
    p.add_argument("--disable-tool", action="append", default=[], metavar="TOOL",
                   help="development only: omit one exposed tool; repeat for more than one")
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
    p.add_argument("--no-task-memory", dest="task_memory", action="store_false", default=True,
                   help="do not inject compact context from completed tasks")
    p.add_argument("--trace", action="store_true",
                   help="allow LangSmith tracing if LANGSMITH_* is configured (off by default)")
    p.add_argument("--stage-log", metavar="PATH",
                   default=os.environ.get("JENI_STAGE_LOG", "logs/jeni_agent.log"),
                   help="write compact execution-stage records here (default: logs/jeni_agent.log)")
    p.add_argument("--trace-json", metavar="PATH",
                   help="write each task's trace (steps + grounding) here, for the "
                        "visualisation page; owner-only, emails/phones scrubbed, e.g. "
                        "traces/run.json (gitignored)")
    return p


def setup(args: argparse.Namespace) -> None:
    set_tracing(args.trace)
    vira_tools.configure(args.mode)
    log_path = getattr(args, "stage_log", None)
    if log_path:
        target = Path(log_path)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        STAGE_LOG.handlers.clear()
        STAGE_LOG.addHandler(handler)
        STAGE_LOG.setLevel(logging.INFO)
        STAGE_LOG.propagate = False
        stage("RUN_CONFIGURED", mode=args.mode, tools=getattr(args, "tools", None), log=target)


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
    latest_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=0)
    terminal = terminal_response_data(result)

    # Show the same temporary context that TaskMemoryMiddleware supplied to
    # the model for this task: compact completed-task records, then the latest
    # user input and this task's live messages.  Do not display or mutate the
    # full checkpoint history here.
    prior_records = list(result.get("task_memory_records", []))
    if (terminal and terminal["status"] == "done"
            and result.get("task_memory_active_start") == len(messages)):
        # after_agent has just stored this task.  It was not available before
        # this task started, so exclude it from this task's displayed context.
        prior_records = prior_records[:-1]
    # `needs_user` keeps the same task open across later human replies, so the
    # display must start from task_memory_active_start—not the latest reply.
    # After `done`, after_agent advances active_start; use the preserved start
    # of that just-completed task instead.
    if terminal and terminal["status"] == "done":
        display_start = result.get("task_memory_last_completed_start", latest_human)
    else:
        display_start = result.get("task_memory_active_start", latest_human)
    try:
        display_start = max(0, min(int(display_start), len(messages)))
    except (TypeError, ValueError):
        display_start = latest_human
    display_messages = packed_task_messages(messages, prior_records, display_start)
    print("\n=== trajectory ===")
    show(display_messages, terminal=terminal)
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


async def arepl(label: str, agent, args: argparse.Namespace, tools,
                after: Callable[[dict], None] | None = None) -> None:
    """Async twin of repl for toolsets with async tools (db/jeni_db)."""
    names = getattr(tools, "names", tools)              # grounding wants the name set
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
    thread_id = uuid.uuid4().hex
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
            print("(new conversation)")
            continue
        try:
            await arun_and_show(agent, user, after, trace, label, names,
                                thread_id)
        except Exception as exc:
            print(printable(f"[error] {type(exc).__name__}: {exc}"))
            continue
