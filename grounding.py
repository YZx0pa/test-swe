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
# An id argument may only take ids that appeared as the same kind of id.
ID_KINDS = {"job_ids": {"job_id", "job_ids"}, "profile_ids": {"profile_id", "profile_ids"},
            "app_ids": {"app_id", "app_ids"}, "match_ids": {"match_id", "match_ids"}}
_ALL_ID_KEYS = set().union(*ID_KINDS.values())
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


def _id_keys(text: str) -> dict[int, set[str]]:
    """int -> the JSON keys it sits under in a tool result (for the id-kind check)."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    found: dict[int, set[str]] = {}

    def walk(node, key=None):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, int) and not isinstance(node, bool) and key:
            found.setdefault(node, set()).add(key)

    walk(data)
    return found


def _misused_as(arg: str, value: Any, where: list[str], sources: list[tuple[str, str]]) -> str | None:
    """The id kind `value` really was, if it's passed as a different kind (e.g. profile_id)."""
    if arg not in ID_KINDS or not isinstance(value, int) or "task" in where:
        return None
    kinds = set()
    for label, text in sources:
        if label in where:
            kinds |= _id_keys(text).get(value, set()) & _ALL_ID_KEYS
    if kinds and not kinds & ID_KINDS[arg]:
        return sorted(kinds)[0]
    return None


def ground_args(args: dict, sources: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """{arg: [{"value", "sources", "misused_as"?}]} for every non-empty leaf value of a VIRA call."""
    out = {}
    for arg, value in (args or {}).items():
        values = value if isinstance(value, list) else [value]
        leaves = [v for v in values if v not in (None, "", 0) or arg == "lang"]
        entries = []
        for v in leaves:
            entry = {"value": v, "sources": _where(v, sources, arg=arg)}
            kind = _misused_as(arg, v, entry["sources"], sources)
            if kind:
                entry["misused_as"] = kind       # a real value, passed as the wrong kind of id
            entries.append(entry)
        if entries:
            out[arg] = entries
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


_OBSERVATION = re.compile(r"^<returncode>(-?\d+)</returncode>\s*<output>\n?(.*?)</output>\s*$", re.S)


def mini_observation(content: str) -> tuple[str, int | None]:
    """mini's tool message -> (command output, return code).

    Handles mini.yaml's JSON observation template and the models' default
    `<returncode>N</returncode><output>...</output>` one (what run_mini.py uses).
    The code is None when the content was already plain output.
    """
    content = content or ""
    try:
        obs = json.loads(content)
        if isinstance(obs, dict) and "returncode" in obs and "output" in obs:
            return str(obs["output"]).strip(), int(obs["returncode"])
    except (TypeError, ValueError):
        pass
    m = _OBSERVATION.match(content)
    if m:
        return m.group(2).strip(), int(m.group(1))
    return content.strip(), None


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
            text, code = mini_observation(m.get("content", ""))
            code = 0 if code is None else code
            step = {"i": len(steps) + 1, "kind": "result", "tool": tool,
                    "status": "success" if code == 0 else "error", "content": text}
            steps.append(step)
            if code == 0:
                sources.append((f"step {step['i']}", text))
    steps.append(_answer_step(len(steps) + 1, answer, sources))
    return steps


def reground(steps: list[dict], task: str, *, mini: bool = False) -> list[dict]:
    """Recompute provenance on a stored trace (e.g. one recorded by an older checker)."""
    sources, out = [("task", task)], []
    for st in steps:
        st = dict(st)
        if st["kind"] == "call" and st.get("tool") in VIRA_TOOLS:
            st["provenance"] = ground_args(st.get("args") or {}, sources)
        elif st["kind"] == "answer":
            st["numbers"] = ground_answer(st.get("text", ""), sources)
        elif st["kind"] == "result" and st.get("status") != "hidden":
            if mini:                         # older traces kept mini's <returncode> wrapper
                text, code = mini_observation(st.get("content", ""))
                st["content"] = text
                if code is not None:
                    st["status"] = "success" if code == 0 else "error"
            if not mini or st.get("status") == "success":
                sources.append((f"step {st['i']}", st.get("content", "")))
        out.append(st)
    return out


def summary(steps: list[dict]) -> dict:
    """Counts for the grounding badges: all args/numbers grounded? how many chained?"""
    args = [(arg, p) for s in steps for arg, vals in (s.get("provenance") or {}).items()
            for p in vals]
    numbers = [n for s in steps if s["kind"] == "answer" for n in s["numbers"]]
    # Pair each call with its result: a wrong-kind id the guard refused never reached VIRA.
    refused_calls, pending = set(), {}
    for s in steps:
        if s["kind"] == "call":
            pending.setdefault(s.get("tool"), []).append(s["i"])
        elif s["kind"] == "result" and pending.get(s.get("tool")):
            call_i = pending[s["tool"]].pop(0)
            if s.get("refused"):
                refused_calls.add(call_i)
    wrong_kind = [(s["i"], f"{p['value']} ({p['misused_as']} as {arg})") for s in steps
                  for arg, vals in (s.get("provenance") or {}).items() for p in vals
                  if "misused_as" in p]
    return {
        "args": len(args),
        "args_grounded": sum(bool(p["sources"]) and "misused_as" not in p for _, p in args),
        "args_chained": sum(any(src.startswith("step") for src in p["sources"])
                            and "misused_as" not in p for _, p in args),
        "numbers": len(numbers),
        "numbers_grounded": sum(bool(n["sources"]) for n in numbers),
        "ungrounded": [f"{p['value']}" for _, p in args if not p["sources"]]
                      + [n["value"] for n in numbers if not n["sources"]],
        "misused": [text for i, text in wrong_kind if i not in refused_calls],
        "misused_blocked": [text for i, text in wrong_kind if i in refused_calls],
    }
