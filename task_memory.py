"""Completed-task memory for the Jeni LangGraph agent.

The module owns deterministic task-record extraction and the middleware that
keeps full checkpoints for UI/audit while giving the model compact context from
completed tasks.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

import grounding


STAGE_LOG = logging.getLogger("jeni.stages")


def _stage(event: str, **details) -> None:
    values = " ".join(f"{key}={value}" for key, value in sorted(details.items())
                      if value is not None)
    STAGE_LOG.info("%s%s", event, f" {values}" if values else "")


def active_task_messages(state: dict) -> list:
    """Messages from the current task, excluding prior completed tasks."""
    messages = state.get("messages", [])
    start = state.get("task_memory_active_start", 0)
    try:
        start = max(0, min(int(start), len(messages)))
    except (TypeError, ValueError):
        start = 0
    return messages[start:]


def packed_task_messages(messages: list, records: list[dict], task_start: int) -> list:
    """Return the compact message list used for one task's model context.

    This is deliberately pure: the middleware uses it for ``request.override``
    and the CLI can use the same function to show an accurate debug trajectory
    without writing the compact digest into the checkpoint.
    """
    active = messages[max(0, task_start):]
    digest = memory_digest(records)
    if not digest or not active:
        return active
    if active[0].type == "human":
        packed = f"{digest}\n\nUser latest input:\n{active[0].text}"
        return [active[0].model_copy(update={"content": packed})] + active[1:]
    return [HumanMessage(content=f"{digest}\n\nUser latest input follows.")] + active


def last_ai_text(messages) -> str:
    for message in reversed(messages):
        if message.type == "ai" and not message.tool_calls and message.text:
            return message.text
    return ""


_EXECUTED_RE = re.compile(r"[Ee]xecuted instead:[^\n]*?with arguments\s*(\{[^\n]*\})")


def _tool_json(text: str) -> dict:
    # HITL edits prepend a human-review note and then ``Tool response:``.  The
    # latter is the real tool payload; never confuse the edited-arguments JSON
    # in the note with the action outcome.
    payload = (text or "").rsplit("Tool response:", 1)[-1]
    start, end = payload.find("{"), payload.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(payload[start:end + 1])
    except (ValueError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def _action_result(text: str) -> dict:
    """Normalise Jeni's request state separately from VIRA's real task state."""
    result = _tool_json(text)
    request_status = str(result.get("request_status", "")).lower()
    group_status = str(result.get("group_status", "")).lower()
    subtasks = result.get("subtasks") if isinstance(result.get("subtasks"), list) else []
    failed = [sub for sub in subtasks if isinstance(sub, dict)
              and str(sub.get("status", "")).lower() == "failed"]
    reason = next((sub.get("failed_reason") for sub in failed if sub.get("failed_reason")), None)
    if request_status != "ok":
        state = "failed"
        reason = reason or result.get("message")
    elif group_status in {"queued", "pending", "processing", "in_progress", "running"}:
        state = "queued"
    elif failed:
        state = "failed"
    elif subtasks and all(str(sub.get("status", "")).lower() == "completed"
                          for sub in subtasks if isinstance(sub, dict)):
        state = "completed"
    else:
        state = "unknown"
    return {"status": state, "request_status": request_status or None,
            "group_status": group_status or None, "failure_reason": reason}


def _resolver_result(text: str) -> dict:
    """Compact, structural result of a read-only resolver call."""
    data = _tool_json(text)
    result = data.get("result", data)
    if not isinstance(result, dict):
        return {}
    while isinstance(result.get("result"), dict):
        result = result["result"]

    values = dict(result.get("resolved_fields") or {})
    for key, value in result.items():
        if key not in {"status", "message", "resolved_fields", "selection", "candidates",
                       "invalid_values", "valid_values", "applications", "result"} and (
                       key.endswith("_id") or key.endswith("_ids")):
            values[key] = value
    selection = result.get("selection")
    if isinstance(selection, dict) and isinstance(selection.get("candidates"), list):
        ids = [row[key] for row in selection["candidates"] if isinstance(row, dict)
               for key in row if key.endswith("_id") and isinstance(row[key], int)]
        if ids:
            values[selection.get("target_field", "candidate_ids")] = ids
    for key in ("applications", "valid_values"):
        value = result.get(key)
        if isinstance(value, list) and value and all(isinstance(item, int) for item in value):
            values.setdefault("app_ids" if key == "applications" else key, value)
    if result.get("status") == "ambiguous" and isinstance(result.get("candidates"), list):
        candidates = [row for row in result["candidates"][:10] if isinstance(row, dict)]
        if candidates:
            values["candidates"] = candidates
    return {"status": result.get("status"), "values": values}


def _input_origins(args: dict, prior_messages: list) -> dict:
    sources = ([("task", message.text) for message in prior_messages if message.type == "human"]
               + [(f"step {index}", message.text) for index, message in enumerate(prior_messages)
                  if message.type == "tool"])
    origins = {}
    for field, entries in grounding.ground_args(args, sources).items():
        found = {source for entry in entries for source in entry.get("sources", [])}
        if "task" in found:
            origins[field] = "user-provided"
        elif found:
            origins[field] = "looked up"
    return origins


def _executed_args(text: str) -> dict | None:
    match = _EXECUTED_RE.search(text or "")
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def extract_record(messages, read_only: frozenset) -> dict:
    """Build one chronological, LLM-free record for a completed task."""
    instruction = next((message.text for message in messages
                        if message.type == "human" and message.text), "")
    sources = ([("task", message.text) for message in messages if message.type == "human"]
               + [(f"step {index}", message.text) for index, message in enumerate(messages)
                  if message.type == "tool"])
    events, actions = [], []
    for index, message in enumerate(messages):
        if message.type != "ai":
            continue
        for call in message.tool_calls:
            result = next((candidate for candidate in messages[index + 1:]
                           if candidate.type == "tool"
                           and candidate.tool_call_id == call.get("id")), None)
            args = call.get("args", {})
            if call["name"] in read_only:
                events.append({"kind": "resolve", "tool": call["name"], "args": args,
                               "input_origins": _input_origins(args, messages[:index]),
                               "result": _resolver_result(result.text) if result else {}})
                continue
            executed = _executed_args(result.text) if result else None
            if executed is not None:
                args = executed
                events.append({"kind": "selection", "tool": call["name"], "args": args,
                               "from": "reviewer-selected"})
            actions.append({"tool": call["name"], "args": args})
            events.append({"kind": "action", "tool": call["name"], "args": args,
                           "result": _action_result(result.text) if result else {"status": "unknown"}})

    # ``resolved`` is retained for compatibility with existing trace callers.
    resolved, seen = [], set()
    for action in actions:
        for field, entries in grounding.ground_args(action["args"], sources).items():
            for entry in entries:
                key = (field, json.dumps(entry["value"], default=str))
                if key in seen:
                    continue
                seen.add(key)
                origin = "user" if "task" in entry["sources"] else (
                    "system (looked up)" if entry["sources"] else "unknown")
                resolved.append({"field": field, "value": entry["value"], "from": origin})
    return {"instruction": instruction, "events": events, "resolved": resolved,
            "actions": actions, "outcome": last_ai_text(messages)}


def memory_digest(records: list[dict], *, keep_last: int = 5, max_age_sec: int = 1800,
                  hard_cap: int = 15) -> str:
    """Render recent completed task records for the next model call."""
    now = time.time()
    kept = [record for index, record in enumerate(records)
            if now - record.get("ts", 0) <= max_age_sec or index >= len(records) - keep_last]
    kept = [record for record in kept[-hard_cap:] if record.get("actions")]
    if not kept:
        return ""

    def short(value, limit=12):
        if isinstance(value, list):
            return "[" + ", ".join(str(item) for item in value[:limit]) + (
                "…" if len(value) > limit else "") + "]"
        return str(value)

    def args(values, origins):
        return ", ".join(f"{field}={short(value)}" +
                         (f" ({origins[field]})" if field in origins else "")
                         for field, value in values.items())

    lines = ["[earlier in this session - for reference]"]
    for task_no, record in enumerate(kept, 1):
        lines.append(f"{task_no}. Instruction: {' '.join(record['instruction'].split())}")
        lines.append("   Timeline:")
        for seq, event in enumerate(record.get("events", []), 1):
            call = f"{event.get('tool')}({args(event.get('args', {}), event.get('input_origins', {}))})"
            if event["kind"] == "resolve":
                result = event.get("result") or {}
                output = ", ".join(f"{key}={short(value)}"
                                   for key, value in (result.get("values") or {}).items())
                text = f"Resolve: {call} -> {output or result.get('status', 'no result')}"
            elif event["kind"] == "selection":
                text = f"Selection: {call} ({event.get('from', 'user-selected')})"
            else:
                result = event.get("result") or {}
                suffix = f" -> status={result.get('status', 'unknown')}"
                if result.get("failure_reason"):
                    suffix += f"; failure_reason={short(result['failure_reason'])}"
                elif result.get("status") == "unknown" and result.get("raw_status"):
                    suffix += f" (tool status={result['raw_status']})"
                text = f"Action: {call}{suffix}"
            lines.append(f"   {seq}. {text}")
    return "\n".join(lines)


class TaskMemoryState(TypedDict, total=False):
    task_memory_records: list[dict]
    task_memory_active_start: int
    task_memory_last_completed_start: int
    task_memory_waiting: bool


class TaskMemoryMiddleware(AgentMiddleware):
    """Preserve full checkpoints but compact completed tasks for model context."""

    state_schema = TaskMemoryState

    def __init__(self, read_only: frozenset, *, record_cap: int = 15):
        super().__init__()
        self.read_only, self.record_cap = read_only, record_cap

    @staticmethod
    def _terminal_response(state):
        """Read create_agent(response_format=TerminalResponse)'s parsed result."""
        response = state.get("structured_response")
        if isinstance(response, dict):
            status, message = response.get("status"), response.get("message")
        else:
            status, message = getattr(response, "status", None), getattr(response, "message", None)
        if status in {"done", "needs_user"} and isinstance(message, str) and message.strip():
            return status, message
        return None, None

    def _model_messages(self, state):
        active = active_task_messages(state)
        records = state.get("task_memory_records", [])
        if not active:
            _stage("MEMORY_PACK", result="skipped", reason="no_active_messages", records=len(records))
            return None
        digest = memory_digest(records)
        if not digest and state.get("task_memory_active_start", 0) in (None, 0):
            _stage("MEMORY_PACK", result="skipped", reason="no_prior_tasks", records=len(records))
            return None
        _stage("MEMORY_PACK", result="injected" if digest else "active_only", records=len(records),
               packed_chars=len(digest), active_messages=len(active))
        if not digest:
            return active

        # Keep one user message for the common new-task case.  The explicit
        # boundary prevents the current request being interpreted as history.
        return packed_task_messages(state.get("messages", []), records,
                                    state.get("task_memory_active_start", 0))

    def wrap_model_call(self, request, handler):
        messages = self._model_messages(request.state)
        return handler(request.override(messages=messages)) if messages else handler(request)

    async def awrap_model_call(self, request, handler):
        messages = self._model_messages(request.state)
        return await handler(request.override(messages=messages) if messages else request)

    def after_model(self, state, runtime):
        """Retain a concise progress signal for developer logs during the tool loop."""
        latest = (state.get("messages") or [None])[-1]
        if getattr(latest, "tool_calls", None):
            _stage("AGENT_RETURN", outcome="tool_calls")
        return None

    def after_agent(self, state, runtime):
        """Record only after the agent has produced the validated terminal response."""
        status, message = self._terminal_response(state)
        if not status:
            _stage("AGENT_RETURN", outcome="missing_structured_response")
            return None
        _stage("AGENT_RETURN", status=status)
        _stage("MEMORY_RECORD", status=status, phase="terminal")
        update = {"task_memory_waiting": status == "needs_user"}
        if status == "done":
            active_start = state.get("task_memory_active_start", 0)
            try:
                active_start = max(0, min(int(active_start), len(state.get("messages", []))))
            except (TypeError, ValueError):
                active_start = 0
            record = extract_record(active_task_messages(state), self.read_only)
            record.update(outcome=message, ts=time.time())
            records = (state.get("task_memory_records", []) + [record])[-self.record_cap:]
            update.update(task_memory_records=records,
                          task_memory_active_start=len(state.get("messages", [])),
                          task_memory_last_completed_start=active_start,
                          task_memory_waiting=False)
            _stage("MEMORY_RECORD", status="stored", records=len(records),
                   actions=len(record["actions"]), events=len(record["events"]))
        else:
            _stage("MEMORY_RECORD", status="deferred", reason="needs_user")
        return update
