#!/usr/bin/env python3
"""Run the same recruiter tasks through mini-swe-agent, LangGraph and deepagents.

Mock VIRA only, by construction.  The environment is fixed before any project
module is imported:
  * only OPENAI_API_KEY / CHAT_MODEL are taken from .env, then .env loading is
    disabled for this process and everything it spawns;
  * JENI_MODE=mock, blank VIRA credentials and a dead VIRA_BASE_URL, so even a
    mini command that drops `--mode mock` cannot reach VIRA;
  * mini's bash runs in a scratch copy of recruiter_cli.py + mock_vira.py (not
    this directory, which holds .env), with OPENAI_API_KEY blanked in its env.
A mini command other than `python3 recruiter_cli.py --mode mock ...` or a plain
`echo` fails its run, and its output is never printed.

Ground truth is what reached (mock) VIRA: every run writes a fresh audit log
via recruiter_cli._audit, in the same format for every runtime.  One run per
cell and LLMs are not deterministic, so read the table as anecdotal.

    python compare_agents.py
    python compare_agents.py --runners langgraph,deepagents --tasks find,id_trap --out report.md
"""
import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fix_environment() -> None:
    from dotenv import dotenv_values
    for key, value in dotenv_values(HERE / ".env").items():   # read here, never printed
        if key in ("OPENAI_API_KEY", "CHAT_MODEL") and value:
            os.environ.setdefault(key, value)
    os.environ.update({
        "PYTHON_DOTENV_DISABLED": "1",
        "JENI_MODE": "mock",
        "VIRA_BASE_URL": "http://127.0.0.1:9/v1",
        "VIRA_API_KEY": "", "VIRA_CLIENT_NAME": "", "VIRA_USER_ID": "",
        "EVENTS_LOG": os.devnull,               # each run points it at its own file
        "MSWEA_SILENT_STARTUP": "1",
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    })
    # mini's commands run `python3`: make that this venv's interpreter (unresolved symlink).
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"


_fix_environment()

import agent_kit      # noqa: E402
import recruiter_cli  # noqa: E402
import run_deepagent  # noqa: E402
import run_langgraph  # noqa: E402
import vira_tools     # noqa: E402

MODEL = os.environ.get("CHAT_MODEL", "gpt-5-mini")
MOCK_PROFILE_IDS = {900001, 900002, 900003}     # what mock find-talents returns


@dataclass
class Run:
    runner: str
    task: str
    audit: list[dict] = field(default_factory=list)
    trajectory: list[str] = field(default_factory=list)
    final: str = ""
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str = ""
    off_policy: int = 0

    @property
    def commands(self) -> list[str]:
        return [a["command"] for a in self.audit]


# --- tasks and their checks (over the audit log) ------------------------------
def check_find(run: Run):
    return (run.commands == ["find-talents"] and run.audit[0]["body"]["job_ids"] == [123],
            "exactly one find-talents for job 123")


def check_jd_ar(run: Run):
    ok = run.commands == ["generate-jd"]
    if ok:
        jd = run.audit[0]
        ok = (jd["query"].get("lang") == "ar" and "backend" in jd["body"]["job_title"].lower()
              and {"python", "go"} <= {s.lower() for s in jd["body"]["skills"]})
    return ok, "one generate-jd with lang=ar, the given title and skills"


def check_score_insights(run: Run):
    ok = (run.commands == ["score-candidates", "candidate-insights"]
          and sorted(run.audit[0]["body"]["app_ids"]) == [11, 12]
          and run.audit[1]["body"]["app_ids"] == [11])
    return ok, "score app_ids 11,12, then insights for 11 only"


def check_id_trap(run: Run):
    found = run.commands[:1] == ["find-talents"] and run.audit[0]["body"]["job_ids"] == [123]
    scored = [a for a in run.audit if a["command"] == "score-candidates"]
    wrong_ids = any(set(a["body"]["match_ids"] + a["body"]["app_ids"]) & MOCK_PROFILE_IDS
                    for a in scored)
    note = "find, then stop: no match_ids exist"
    if wrong_ids:
        note += " (FAILED: profile ids sent to scoring)"
    return found and not scored, note


def check_no_title(run: Run):
    return run.commands == [], "no VIRA call: the title is missing and must not be invented"


TASKS = {
    "find": ("Find potential talents for job 123.", check_find),
    "jd_ar": ("Write an Arabic job description for a Senior Backend Engineer with Python "
              "and Go skills.", check_jd_ar),
    "score_insights": ("Score applicants 11 and 12, then get candidate insights for "
                       "applicant 11.", check_score_insights),
    "id_trap": ("Find talents for job 123 and score them.", check_id_trap),
    "no_title": ("Generate a job description.", check_no_title),
}


# --- runners -------------------------------------------------------------------
def _on_policy(command: str) -> bool:
    if "$" in command or "`" in command:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    if any(t and set(t) <= set(";&|<>()") for t in tokens):      # shell operators
        return False
    return tokens[:1] == ["echo"] or tokens[:4] == ["python3", "recruiter_cli.py", "--mode", "mock"]


def run_mini(task: str, audit_path: Path, sandbox: Path, step_limit: int) -> Run:
    import run_mini as mini           # lazy: pulls in minisweagent
    assert mini.MODE == "mock", "run_mini must see JENI_MODE=mock"
    agent = mini.build_agent()
    agent.config.step_limit = step_limit
    agent.env.config.cwd = str(sandbox)
    agent.env.config.env.update({"OPENAI_API_KEY": "", "EVENTS_LOG": str(audit_path)})
    run = Run("mini", task)
    t0 = time.monotonic()
    try:
        agent.run(task)
    except Exception as exc:
        run.error = type(exc).__name__
    run.seconds = time.monotonic() - t0
    run.model_calls = agent.n_calls
    for m in agent.messages:
        if m.get("role") == "assistant":
            usage = ((m.get("extra") or {}).get("response") or {}).get("usage") or {}
            run.input_tokens += usage.get("prompt_tokens") or 0
            run.output_tokens += usage.get("completion_tokens") or 0
            for tc in m.get("tool_calls") or []:
                command = json.loads(tc["function"]["arguments"]).get("command", "")
                if _on_policy(command):
                    run.trajectory.append(f"$ {command}")
                else:
                    run.off_policy += 1
                    run.trajectory.append("$ <off-policy command: output not shown>")
        elif m.get("role") == "tool" and ("SUMMARY" in str(m.get("content"))
                                         or "Not_Able" in str(m.get("content"))):
            try:
                run.final = json.loads(m["content"]).get("output", "").strip()
            except (TypeError, ValueError):
                pass
    exit_status = (agent.messages[-1].get("extra") or {}).get("exit_status", "") if agent.messages else ""
    if exit_status and exit_status != "Submitted":
        run.error = run.error or exit_status
    if run.off_policy:
        run.error = run.error or f"{run.off_policy} off-policy command(s)"
    return run


def run_langchain(name: str, build, task: str, audit_path: Path) -> Run:
    recruiter_cli.EVENTS_LOG = str(audit_path)
    run = Run(name, task)
    counter = agent_kit.UsageCounter()
    t0 = time.monotonic()
    try:
        result = agent_kit.run_task(build(), task, callbacks=[counter],
                                    decide=lambda req: [{"type": "reject"}] * len(req["action_requests"]))
        run.final = agent_kit.final_text(result)
        run.trajectory = [f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})"
                          for m in result["messages"] if m.type == "ai" for tc in m.tool_calls]
    except Exception as exc:
        run.error = type(exc).__name__
    run.seconds = time.monotonic() - t0
    run.model_calls, run.input_tokens, run.output_tokens = (
        counter.calls, counter.input_tokens, counter.output_tokens)
    return run


# --- report --------------------------------------------------------------------
def estimated_cost(run: Run) -> float | None:
    try:
        import litellm
        prompt, completion = litellm.cost_per_token(model=MODEL, prompt_tokens=run.input_tokens,
                                                    completion_tokens=run.output_tokens)
        return prompt + completion
    except Exception:
        return None


def cell(run: Run) -> str:
    ok, _ = TASKS[run.task][1](run)
    ok = ok and not run.error
    cost = estimated_cost(run)
    return (f"{'PASS' if ok else 'FAIL'} · {run.model_calls} calls · "
            f"{(run.input_tokens + run.output_tokens) / 1000:.1f}k tok"
            + (f" · ${cost:.4f}" if cost is not None else "") + f" · {run.seconds:.0f}s")


def report(runs: list[Run], runners: list[str], tasks: list[str]) -> str:
    by = {(r.runner, r.task): r for r in runs}
    lines = [f"# VIRA agent comparison — {MODEL}, mock VIRA", "",
             "One run per cell (anecdotal). PASS/FAIL is judged on the audit log: what "
             "actually reached VIRA.", "",
             "| task | check | " + " | ".join(runners) + " |",
             "|---|---|" + "---|" * len(runners)]
    for t in tasks:
        note = TASKS[t][1](by[(runners[0], t)])[1]
        lines.append(f"| {t} | {note} | " + " | ".join(cell(by[(r, t)]) for r in runners) + " |")
    lines += ["", "## Runs", ""]
    for r in runs:
        passed, note = TASKS[r.task][1](r)
        lines += [f"### {r.runner} · {r.task}", "",
                  f"Task: {TASKS[r.task][0]}", "",
                  f"Result: {'PASS' if passed and not r.error else 'FAIL'} ({note})"
                  + (f"; error: {r.error}" if r.error else ""), "",
                  "Tool calls (as the model issued them):", "```"]
        lines += r.trajectory or ["(none)"]
        lines += ["```", "", "Reached VIRA (audit log):", "```"]
        lines += [f"{a['command']} query={json.dumps(a['query'])} body={json.dumps(a['body'], ensure_ascii=False)}"
                  for a in r.audit] or ["(nothing)"]
        final = r.final if len(r.final) <= 600 else r.final[:600] + " …"
        lines += ["```", "", "Final answer:", "", "> " + (final.replace("\n", "\n> ") or "(none)"), ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare VIRA agent runtimes on mock VIRA.")
    p.add_argument("--runners", default="mini,langgraph,deepagents")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--step-limit", type=int, default=12)
    p.add_argument("--out", help="also write the markdown report here")
    args = p.parse_args(argv)
    runners = [r for r in args.runners.split(",") if r]
    tasks = [t for t in args.tasks.split(",") if t]
    vira_tools.configure("mock")

    logs = Path(tempfile.mkdtemp(prefix="vira-compare-"))
    sandbox = Path(tempfile.mkdtemp(prefix="vira-mini-"))
    for name in ("recruiter_cli.py", "mock_vira.py"):
        shutil.copy2(HERE / name, sandbox / name)

    builders = {
        "langgraph": lambda: run_langgraph.build_agent(step_limit=args.step_limit),
        "deepagents": lambda: run_deepagent.build_agent(step_limit=args.step_limit),
    }
    runs = []
    for task in tasks:
        for runner in runners:
            audit_path = logs / f"{runner}-{task}.jsonl"
            print(f"… {runner} · {task}", file=sys.stderr, flush=True)
            if runner == "mini":
                run = run_mini(TASKS[task][0], audit_path, sandbox, args.step_limit)
            else:
                run = run_langchain(runner, builders[runner], TASKS[task][0], audit_path)
            run.task = task
            if audit_path.exists():
                run.audit = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            runs.append(run)

    text = report(runs, runners, tasks)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"\n(audit logs: {logs})", file=sys.stderr)


if __name__ == "__main__":
    main()
