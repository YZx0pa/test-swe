#!/usr/bin/env python3
"""Run the same recruiter tasks through mini-swe-agent, LangGraph and deepagents.

Mock VIRA only, by construction.  The environment is fixed before any project
module is imported:
  * only OPENAI_API_KEY / CHAT_MODEL (and JENI_TASKS_FILE, where Jeni's internal task
    catalog lives) are taken from .env, then .env loading is
    disabled for this process and everything it spawns;
  * JENI_MODE=mock, blank VIRA credentials and a dead VIRA_BASE_URL;
  * mini's "bash" is mini_env.RecruiterEnvironment, which runs only
    `python3 recruiter_cli.py …` (always with --mode mock) and `echo`, without a
    shell and with a minimal environment.
Any other mini command is refused before it runs, still fails its run, and its
text is never printed.

Ground truth is what reached (mock) VIRA: every run writes a fresh audit log
via recruiter_cli._audit, in the same format for every runtime.  One run per
cell and LLMs are not deterministic, so read the table as anecdotal.

    python compare_agents.py
    python compare_agents.py --model gpt-4o-mini --repeat 3 --json traces/runs.json
    python compare_agents.py --runners langgraph,deepagents --tasks find,id_trap --out traces/report.md
    python compare_agents.py --suite jeni --repeat 3 --out traces/jeni.md   # Jeni's own tasks (jeni_eval.py)

--out and --json files are written owner-only with emails and phone numbers
scrubbed (agent_kit.write_private); traces/ is gitignored.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fix_environment() -> None:
    from dotenv import dotenv_values
    for key, value in dotenv_values(HERE / ".env").items():   # read here, never printed
        if key in ("OPENAI_API_KEY", "CHAT_MODEL", "JENI_TASKS_FILE") and value:
            os.environ.setdefault(key, value)
    os.environ.update({
        "PYTHON_DOTENV_DISABLED": "1",
        "JENI_MODE": "mock",
        "VIRA_BASE_URL": "http://127.0.0.1:9/v1",
        "VIRA_API_KEY": "", "VIRA_CLIENT_NAME": "", "VIRA_USER_ID": "",
        "EVENTS_LOG": os.devnull,               # each run points it at its own file
        "MSWEA_SILENT_STARTUP": "1",
        "LANGSMITH_TRACING_V2": "false", "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false", "LANGCHAIN_TRACING": "false",
    })


_fix_environment()

import agent_kit      # noqa: E402
import grounding      # noqa: E402
import jeni_eval      # noqa: E402
import jeni_tools     # noqa: E402
import mini_policy    # noqa: E402
import recruiter_cli  # noqa: E402
import run_deepagent  # noqa: E402
import run_langgraph  # noqa: E402
import vira_tools     # noqa: E402
from mock_vira import MockVira  # noqa: E402

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
    repeat: int = 1
    steps: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return TASKS[self.task][1](self)[0] and not self.error

    @property
    def grounding(self) -> dict:
        return grounding.summary(self.steps)

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


def check_top_pick(run: Run):
    ids = [11, 12, 13]
    scores = MockVira.call("candidate_score_calculation", {}, {"app_ids": ids})["result"]["scores"]
    best = max(scores, key=lambda s: s["composite_score"])["app_id"]
    ok = (run.commands == ["score-candidates", "candidate-insights"]
          and sorted(run.audit[0]["body"]["app_ids"]) == ids
          and run.audit[1]["body"]["app_ids"] == [best])
    return ok, f"score 11,12,13, then insights only for the top scorer ({best})"


def check_id_trap(run: Run):
    profiles = sorted(MOCK_PROFILE_IDS)
    matches = MockVira.call("get_match_id_from_profile_id", {},
                            {"job_id": 123, "profile_ids": profiles})["result"]["matches"]
    ok = run.commands == ["find-talents", "get-match-id-from-profile-id", "score-candidates"]
    if ok:
        find, lookup, score = run.audit
        ok = (find["body"]["job_ids"] == [123]
              and lookup["body"]["job_id"] == 123
              and sorted(lookup["body"]["profile_ids"]) == profiles
              and sorted(score["body"]["match_ids"]) == sorted(m["match_id"] for m in matches)
              and not score["body"]["app_ids"])
    wrong_ids = any(set(a["body"]["match_ids"] + a["body"]["app_ids"]) & MOCK_PROFILE_IDS
                    for a in run.audit if a["command"] == "score-candidates")
    note = "find, look up match ids, then score those match ids"
    if wrong_ids:
        note += " (FAILED: profile ids sent to scoring)"
    return ok, note


def check_no_title(run: Run):
    return run.commands == [], "no VIRA call: the title is missing and must not be invented"


VIRA_TASKS = {
    "find": ("Find potential talents for job 123.", check_find),
    "jd_ar": ("Write an Arabic job description for a Senior Backend Engineer with Python "
              "and Go skills.", check_jd_ar),
    "score_insights": ("Score applicants 11 and 12, then get candidate insights for "
                       "applicant 11.", check_score_insights),
    "top_pick": ("Score applicants 11, 12 and 13, then get candidate insights only for the "
                 "one with the highest composite score.", check_top_pick),
    "id_trap": ("Find talents for job 123 and score them.", check_id_trap),
    "no_title": ("Generate a job description.", check_no_title),
}
# --suite picks the tasks and the tools: the sample endpoints, or Jeni's own tasks.
SUITES = {"vira": VIRA_TASKS, "jeni": jeni_eval.TASKS}     # toolset: agent_kit.toolset(suite)
TASKS = VIRA_TASKS


# --- runners -------------------------------------------------------------------
def _on_policy(command: str) -> bool:
    """What mini_env would run (it refuses the rest): the same parser, the same mode."""
    return mini_policy.parse(command, "mock")[0] != "refused"


def run_mini(task: str, audit_path: Path, step_limit: int) -> Run:
    import run_mini as mini           # lazy: pulls in minisweagent
    assert mini.MODE == "mock", "run_mini must see JENI_MODE=mock"
    agent = mini.build_agent()
    agent.config.step_limit = step_limit
    agent.env.config.env.update({"EVENTS_LOG": str(audit_path)})
    run = Run("mini", task)
    t0 = time.monotonic()
    try:
        agent.run(task)
    except Exception as exc:
        run.error = type(exc).__name__
    run.seconds = time.monotonic() - t0
    run.model_calls = agent.n_calls
    pending = []                      # per tool call: was it on policy?
    for m in agent.messages:
        if m.get("role") == "assistant":
            usage = ((m.get("extra") or {}).get("response") or {}).get("usage") or {}
            run.input_tokens += usage.get("prompt_tokens") or 0
            run.output_tokens += usage.get("completion_tokens") or 0
            for tc in m.get("tool_calls") or []:
                command = json.loads(tc["function"]["arguments"]).get("command", "")
                pending.append(_on_policy(command))
                if pending[-1]:
                    run.trajectory.append(f"$ {command}")
                else:
                    run.off_policy += 1
                    run.trajectory.append("$ <off-policy command: output not shown>")
        elif m.get("role") == "tool" and pending and not pending.pop(0):
            continue                  # a refused command's output is never read
        elif m.get("role") == "tool" and ("SUMMARY" in str(m.get("content"))
                                         or "Not_Able" in str(m.get("content"))):
            try:
                run.final = json.loads(m["content"]).get("output", "").strip()
            except (TypeError, ValueError):
                pass
    run.steps = grounding.trace_from_mini(agent.messages, task)
    exit_status = (agent.messages[-1].get("extra") or {}).get("exit_status", "") if agent.messages else ""
    if exit_status and exit_status != "Submitted":
        run.error = run.error or exit_status
    if run.off_policy:
        run.error = run.error or f"{run.off_policy} off-policy command(s)"
    return run


def run_langchain(name: str, build, task: str, audit_path: Path,
                  tools=grounding.VIRA_TOOLS) -> Run:
    recruiter_cli.EVENTS_LOG = str(audit_path)
    run = Run(name, task)
    counter = agent_kit.UsageCounter()
    t0 = time.monotonic()
    try:
        result = agent_kit.run_task(build(), task, callbacks=[counter],
                                    decide=lambda req: [{"type": "reject"}] * len(req["action_requests"]))
        run.final = agent_kit.final_text(result)
        run.steps = grounding.trace_from_messages(result["messages"], task, tools)
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


def _median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else 0


def cell(group: list[Run]) -> str:
    """One table cell: pass rate over the repeats, then medians."""
    passed = sum(r.passed for r in group)
    costs = [c for c in (estimated_cost(r) for r in group) if c is not None]
    tokens = _median([r.input_tokens + r.output_tokens for r in group])
    verdict = ("PASS" if passed == len(group) else "FAIL") if len(group) == 1 else f"{passed}/{len(group)}"
    return (f"{verdict} · {_median([r.model_calls for r in group])} calls · {tokens / 1000:.1f}k tok"
            + (f" · ${sum(costs) / len(costs):.4f}" if costs else "")
            + f" · {_median([r.seconds for r in group]):.0f}s")


def details(r: Run) -> list[str]:
    passed, note = TASKS[r.task][1](r)
    g = r.grounding
    lines = [f"### {r.runner} · {r.task} · run {r.repeat}", "",
             f"Task: {TASKS[r.task][0]}", "",
             f"Result: {'PASS' if r.passed else 'FAIL'} ({note})"
             + (f"; error: {r.error}" if r.error else ""), "",
             f"Grounding: {g['args_grounded']}/{g['args']} tool args traced "
             f"({g['args_chained']} chained from earlier results), "
             f"{g['numbers_grounded']}/{g['numbers']} answer numbers traced"
             + (f"; UNGROUNDED: {', '.join(g['ungrounded'])}" if g["ungrounded"] else "")
             + (f"; invented ids refused by the guard: {', '.join(g['ungrounded_blocked'])}"
                if g["ungrounded_blocked"] else "")
             + (f"; WRONG ID KIND: {', '.join(g['misused'])}" if g["misused"] else "")
             + (f"; refused by the guard: {', '.join(g['misused_blocked'])}" if g["misused_blocked"] else ""), "",
             "Tool calls (as the model issued them):", "```"]
    lines += r.trajectory or ["(none)"]
    lines += ["```", "", "Reached VIRA (audit log):", "```"]
    lines += [f"{a['command']} query={json.dumps(a['query'])} "
              f"body={json.dumps(a['body'], ensure_ascii=False)}" for a in r.audit] or ["(nothing)"]
    final = r.final if len(r.final) <= 600 else r.final[:600] + " …"
    return lines + ["```", "", "Final answer:", "", "> " + (final.replace("\n", "\n> ") or "(none)"), ""]


def report(runs: list[Run], runners: list[str], tasks: list[str], repeat: int,
           suite: str = "vira") -> str:
    groups = {(runner, t): [r for r in runs if r.runner == runner and r.task == t]
              for runner in runners for t in tasks}
    lines = [f"# VIRA agent comparison ({suite} tasks) — {MODEL}, mock VIRA", "",
             f"{repeat} run(s) per cell; medians shown. PASS/FAIL is judged on the audit log: "
             "what actually reached VIRA.", "",
             "| task | check | " + " | ".join(runners) + " |",
             "|---|---|" + "---|" * len(runners)]
    for t in tasks:
        note = TASKS[t][1](groups[(runners[0], t)][0])[1]
        lines.append(f"| {t} | {note} | " + " | ".join(cell(groups[(r, t)]) for r in runners) + " |")
    totals = []
    for runner in runners:
        mine = [r for r in runs if r.runner == runner]
        g = [r.grounding for r in mine]
        costs = [c for c in (estimated_cost(r) for r in mine) if c is not None]
        totals.append(f"{sum(r.passed for r in mine)}/{len(mine)} pass · "
                      f"~${sum(costs):.3f} · ungrounded values: {sum(len(x['ungrounded']) for x in g)} "
                      f"(+{sum(len(x['ungrounded_blocked']) for x in g)} refused) · "
                      f"wrong-kind ids reaching VIRA: {sum(len(x['misused']) for x in g)} "
                      f"(+{sum(len(x['misused_blocked']) for x in g)} refused)")
    lines.append("| **total** | | " + " | ".join(totals) + " |")
    lines += ["", "## Runs (every failure, plus run 1 of each cell)", ""]
    for r in runs:
        if not r.passed or r.repeat == 1:
            lines += details(r)
    return "\n".join(lines)


def to_json(runs: list[Run], repeat: int, suite: str = "vira") -> dict:
    return {
        "model": MODEL, "mode": "mock", "repeat": repeat, "suite": suite,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": {k: {"text": v[0], "check": v[1](next(r for r in runs if r.task == k))[1]}
                  for k, v in TASKS.items() if any(r.task == k for r in runs)},
        "runs": [{
            "runner": r.runner, "task": r.task, "repeat": r.repeat, "passed": r.passed,
            "error": r.error, "model_calls": r.model_calls, "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens, "seconds": round(r.seconds, 1),
            "cost": estimated_cost(r), "audit": r.audit, "steps": r.steps,
            "grounding": r.grounding, "final": r.final,
        } for r in runs],
    }


def main(argv=None):
    global MODEL, TASKS
    p = argparse.ArgumentParser(description="Compare VIRA agent runtimes on mock VIRA.")
    p.add_argument("--suite", choices=sorted(SUITES), default="vira",
                   help="vira: the sample endpoints; jeni: Jeni's own tasks (LangGraph and "
                        "deepagents only)")
    p.add_argument("--runners", help="default: mini,langgraph,deepagents (vira) or "
                                     "langgraph,deepagents (jeni)")
    p.add_argument("--tasks", help="default: every task in the suite")
    p.add_argument("--model", default=MODEL, help="CHAT_MODEL for every runner (litellm-style id)")
    p.add_argument("--repeat", type=int, default=1, help="runs per cell (LLMs are not deterministic)")
    p.add_argument("--step-limit", type=int, default=12)
    p.add_argument("--out", help="also write the markdown report here")
    p.add_argument("--json", help="write every run's trace (steps, grounding, audit) here")
    args = p.parse_args(argv)
    MODEL = os.environ["CHAT_MODEL"] = args.model
    TASKS = SUITES[args.suite]
    try:
        toolset = agent_kit.toolset(args.suite)       # jeni: reads the internal catalog
    except jeni_tools.CatalogMissing as exc:
        p.error(str(exc))
    runners = [r for r in (args.runners or ("mini,langgraph,deepagents" if args.suite == "vira"
                                            else "langgraph,deepagents")).split(",") if r]
    tasks = [t for t in (args.tasks or ",".join(TASKS)).split(",") if t]
    if args.suite == "jeni" and "mini" in runners:
        p.error("mini runs recruiter_cli's subcommands, which don't cover Jeni's tasks")
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        p.error(f"unknown task(s) for --suite {args.suite}: {', '.join(unknown)}")
    vira_tools.configure("mock")

    logs = Path(tempfile.mkdtemp(prefix="vira-compare-"))

    builders = {
        "langgraph": lambda: run_langgraph.build_agent(step_limit=args.step_limit, toolset=toolset),
        "deepagents": lambda: run_deepagent.build_agent(step_limit=args.step_limit, toolset=toolset),
    }
    runs = []
    for task in tasks:
        for runner in runners:
            for n in range(1, args.repeat + 1):
                audit_path = logs / f"{runner}-{task}-{n}.jsonl"
                print(f"… {MODEL} · {runner} · {task} · run {n}", file=sys.stderr, flush=True)
                if runner == "mini":
                    run = run_mini(TASKS[task][0], audit_path, args.step_limit)
                else:
                    run = run_langchain(runner, builders[runner], TASKS[task][0], audit_path,
                                        toolset.names)
                run.task, run.repeat = task, n
                if audit_path.exists():
                    run.audit = [json.loads(line) for line in
                                 audit_path.read_text(encoding="utf-8").splitlines()]
                runs.append(run)

    text = report(runs, runners, tasks, args.repeat, args.suite)
    print(text)
    if args.out:
        agent_kit.write_private(args.out, text + "\n")
    if args.json:
        agent_kit.write_private(args.json, json.dumps(to_json(runs, args.repeat, args.suite),
                                                      ensure_ascii=False, indent=1))
    print(f"\n(audit logs: {logs})", file=sys.stderr)


if __name__ == "__main__":
    main()
