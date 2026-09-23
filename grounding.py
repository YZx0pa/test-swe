"""Traces and grounding: where does every value an agent uses come from?

A trace is one run in order: the model's tool calls, what each returned, and
the final answer.  Every VIRA tool argument, and every id/score in the final
answer, is looked up in the user's task and in the tool results that came
before it:

    sources ["task"]            the value was in the user's request
    sources ["step 2", ...]     it appeared in step 2's tool result: the agent
                                used what it observed ("chained")
    sources ["default"]         a documented default (lang=en, nothing asked)
    sources []                  UNGROUNDED: nothing before it contains it

Used by compare_agents.py (--json) and the runners (--trace-json) to feed the
visualisation page.  Pure functions: no environment, no I/O.
"""
import json
import re
import shlex
from typing import Any

VIRA_TOOLS = {"find_talents", "generate_jd", "score_candidates", "candidate_insights"}
LANGUAGES = {"ar": "arabic", "en": "english", "fr": "french", "de": "german",
             "es": "spanish", "hi": "hindi", "zh": "chinese", "ur": "urdu"}
_NUMBER = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)([%+])?(?![\d])")
_INT_FLAGS = {"job_ids", "profile_ids", "app_ids", "match_ids"}
_LIST_FLAGS = {"skills", "job_function", "industry", "other_requirements"}


# --- provenance ----------------------------------------------------------------
def _numbers(text: str) -> tuple[set[int], set[float]]:
    ints, floats = set(), set()
    for m in _NUMBER.finditer(text):
        raw = m.group(1)
        if "." in raw:
            floats.add(float(raw))
        else:
            ints.add(int(raw))
    return ints, floats


def _where(value: Any, sources: list[tuple[str, str]], *, arg: str = "") -> list[str]:
    if arg == "lang" and isinstance(value, str):
        code = value.lower()
        name = LANGUAGES.get(code, code)
        found = [label for label, text in sources
                 if name in text.lower() or re.search(rf"\b{re.escape(code)}\b", text.lower())]
        if not found and code == "en":
            return ["default"]
        return found
    if isinstance(value, bool) or value is None:
        return ["default"]
    if isinstance(value, int):
        return [label for label, text in sources if value in _numbers(text)[0]]
    if isinstance(value, float):
        return [label for label, text in sources if value in _numbers(text)[1]]
    needle = str(value).strip().lower()
    return [label for label, text in sources if needle and needle in text.lower()]


def ground_args(args: dict, sources: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """{arg: [{"value", "sources"}]} for every non-empty leaf value of a VIRA call."""
    out = {}
    for arg, value in (args or {}).items():
        values = value if isinstance(value, list) else [value]
        leaves = [v for v in values if v not in (None, "", 0) or arg == "lang"]
        if leaves:
            out[arg] = [{"value": v, "sources": _where(v, sources, arg=arg)} for v in leaves]
    return out


def ground_answer(text: str, sources: list[tuple[str, str]]) -> list[dict]:
    """Every id/score/percentage in the final answer, with where it came from."""
    found, seen = [], set()
    for m in _NUMBER.finditer(text or ""):
        raw, suffix = m.group(1), m.group(2) or ""
        if len(raw) == 1 and not suffix:          # list numbering, small counts
            continue
        if raw + suffix in seen:
            continue
        seen.add(raw + suffix)
        value: Any = float(raw) if "." in raw else int(raw)
        where = _where(value, sources)
        if not where and suffix == "%":           # "95%" relayed from a 0.95 score
            where = _where(round(value / 100, 4), sources)
        found.append({"value": raw + suffix, "sources": where})
    return found


# --- traces --------------------------------------------------------------------
def _answer_step(i: int, text: str, sources) -> dict:
    return {"i": i, "kind": "answer", "text": text, "numbers": ground_answer(text, sources)}


def trace_from_messages(messages, task: str) -> list[dict]:
    """Steps of a LangChain/LangGraph run (create_agent or deepagents main agent)."""
    steps, sources, answer = [], [("task", task)], ""
    for m in messages:
        if m.type == "ai":
            if m.text and m.tool_calls:
                steps.append({"i": len(steps) + 1, "kind": "note", "text": m.text})
            for tc in m.tool_calls:
                step = {"i": len(steps) + 1, "kind": "call", "tool": tc["name"], "args": tc["args"]}
                if tc["name"] in VIRA_TOOLS:
                    step["provenance"] = ground_args(tc["args"], sources)
                steps.append(step)
            if not m.tool_calls:
                answer = m.text
        elif m.type == "tool":
            text = m.text
            step = {"i": len(steps) + 1, "kind": "result", "tool": m.name,
                    "status": getattr(m, "status", "success"), "content": text,
                    "refused": text.startswith('{"status": "error", "message": "Refused')}
            steps.append(step)
            sources.append((f"step {step['i']}", text))
    steps.append(_answer_step(len(steps) + 1, answer, sources))
    return steps


def parse_cli(command: str) -> tuple[str, dict] | tuple[str, str] | None:
    """`python3 recruiter_cli.py --mode mock find-talents --job-ids 1,2` -> ("find_talents", {...})."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if tokens[:1] == ["echo"]:
        return "echo", " ".join(tokens[1:])
    if tokens[:2] != ["python3", "recruiter_cli.py"]:
        return None
    sub, args, rest, i = None, {}, tokens[2:], 0
    while i < len(rest):
        tok = rest[i]
        if tok.startswith("--"):
            name, raw = tok[2:].replace("-", "_"), rest[i + 1] if i + 1 < len(rest) else ""
            i += 2
            if name in ("mode", "confirmed"):
                continue
            if name in _INT_FLAGS:
                parts = [p.strip() for p in raw.split(",") if p.strip()]
                args[name] = [int(p) if p.isdigit() else p for p in parts]
            elif name in _LIST_FLAGS:
                args[name] = [p.strip() for p in raw.split(",") if p.strip()]
            elif name == "job_id":
                args[name] = int(raw) if raw.isdigit() else raw
            else:
                args[name] = raw
        else:
            sub = sub or tok
            i += 1
    return (sub or "").replace("-", "_"), args


def trace_from_mini(messages: list[dict], task: str) -> list[dict]:
    """Steps of a mini-swe-agent run.  Off-policy commands and their output stay hidden."""
    steps, sources, answer, pending = [], [("task", task)], "", []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            if (m.get("content") or "").strip():
                steps.append({"i": len(steps) + 1, "kind": "note", "text": m["content"].strip()})
            for tc in m.get("tool_calls") or []:
                command = json.loads(tc["function"]["arguments"]).get("command", "")
                parsed = parse_cli(command)
                if parsed and parsed[0] == "echo":
                    text = parsed[1]
                    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in text:
                        answer = text if "SUMMARY" in text or not answer else answer
                        steps.append({"i": len(steps) + 1, "kind": "echo", "text": text})
                    pending.append(None)
                elif parsed and parsed[0] in VIRA_TOOLS:
                    steps.append({"i": len(steps) + 1, "kind": "call", "tool": parsed[0],
                                  "args": parsed[1], "command": command,
                                  "provenance": ground_args(parsed[1], sources)})
                    pending.append(parsed[0])
                else:
                    steps.append({"i": len(steps) + 1, "kind": "call", "tool": "bash",
                                  "args": {}, "off_policy": True,
                                  "command": "<off-policy command: not shown>"})
                    pending.append("hidden")
        elif role == "tool" and pending:
            tool = pending.pop(0)
            if tool is None:
                continue
            if tool == "hidden":
                steps.append({"i": len(steps) + 1, "kind": "result", "tool": "bash",
                              "status": "hidden", "content": "<not shown>"})
                continue
            try:
                observation = json.loads(m.get("content", ""))
                text, code = observation.get("output", ""), observation.get("returncode", 0)
            except (TypeError, ValueError):
                text, code = str(m.get("content", "")), 0
            step = {"i": len(steps) + 1, "kind": "result", "tool": tool,
                    "status": "success" if code == 0 else "error", "content": text.strip()}
            steps.append(step)
            if code == 0:
                sources.append((f"step {step['i']}", text))
    steps.append(_answer_step(len(steps) + 1, answer, sources))
    return steps


def summary(steps: list[dict]) -> dict:
    """Counts for the grounding badges: all args/numbers grounded? how many chained?"""
    args = [p for s in steps for vals in (s.get("provenance") or {}).values() for p in vals]
    numbers = [n for s in steps if s["kind"] == "answer" for n in s["numbers"]]
    return {
        "args": len(args),
        "args_grounded": sum(bool(p["sources"]) for p in args),
        "args_chained": sum(any(src.startswith("step") for src in p["sources"]) for p in args),
        "numbers": len(numbers),
        "numbers_grounded": sum(bool(n["sources"]) for n in numbers),
        "ungrounded": [f"{p['value']}" for p in args if not p["sources"]]
                      + [n["value"] for n in numbers if not n["sources"]],
    }
