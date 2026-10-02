#!/usr/bin/env python3
"""Check the real VIRA engine end to end: read-only task groups, then (--writes) a write test.

    python vira_check.py              # send the checks, wait up to 30 s, report what ran
    python vira_check.py --wait 60
    python vira_check.py --writes     # then the write test, if the engine is running groups

It sends three task groups through recruiter_cli.execute (real mode: VIRA_ACTUAL_LOCATION with
VIRA_XRTOKEN, audited) and reads each one back (vira_results): through the engine's info call
when VIRA_RESULT_LOCATION is set, from its tables on TRON_POSTGRES_DSN otherwise:

  runs        v2's own payload for search_users: does a group run at all?
  engine      the engine's sample vocabulary: sub_task_get_job_description for a title (text only).
  failure     two tasks in one group.  The first has a sub-task that must fail (no job title)
              followed by one that would succeed alone; the second is a user search.  It shows
              whether a failed sub-task stops the rest of its task (chained) and whether a
              failed task stops the next task (independent).

Without --writes, nothing here creates or changes a job, an application or a user.  It prints
statuses and failure reasons of these groups only, never their contents, the URL or the token.

--writes changes staging, on a job of its own only, and only after a read-only group has just
run (otherwise its writes would sit queued and run later with nobody watching):
  1. the agent, in real mode, creates "Jeni v2 test <time>" (Python, 1-2 years) and adds Kafka to
     it, using the id create_job returns: cards are approved for exactly those two calls on that
     title, and any other card is rejected;
  2. directly: make it private, check Kafka is on it, remove Kafka, edit its description, check;
  3. close it (the cleanup), whatever happened before.
No LinkedIn, no shares, no candidates, no collaborators, no ownership change.
Exit status: 0 everything ran, 1 something is still queued or failed to send, 2 not set up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from langchain_core.callbacks import BaseCallbackHandler

import recruiter_cli as vira       # loads .env by path, as the runners do
import jeni_tools
import vira_results

# The agent's part of the write test: the only cards it may get approved.
WRITE_CARDS = ("create_job", "add_job_skills")


def _group(name: str, session: str, tasks: list) -> dict:
    return {"agent_session_uuid": session, "task_group_name": name, "tasks": tasks}


def _setup_job(*subs) -> dict:
    return {"task_name": "task_setup_job", "sub_tasks": [
        {"sub_task_name": name, "fields": fields} for name, fields in subs]}


def checks(stamp: str) -> list[tuple[str, dict]]:
    """Each group has a session of its own, as jeni_tools.session_uuid gives every submission."""
    new_session = jeni_tools.session_uuid
    search = jeni_tools.payload(jeni_tools.catalog()["search_users"], {"search_key": "a", "limit": 3},
                                new_session(f"vira-check-{stamp}"))
    title = [{"field_name": "job_title", "field_value": "Software Engineer"}]
    return [
        ("runs", {**search, "task_group_name": f"Jeni v2 check {stamp} - runs"}),
        ("engine", _group(f"Jeni v2 check {stamp} - engine", new_session(f"vira-check-{stamp}"),
                          [_setup_job(("sub_task_get_job_description", title))])),
        ("failure", _group(f"Jeni v2 check {stamp} - failure", new_session(f"vira-check-{stamp}"), [
            _setup_job(("sub_task_get_job_description", []), ("sub_task_get_job_skills", title)),
            search["tasks"][0]])),
    ]


def statuses(group: dict | None) -> list[tuple[str, list[tuple[str, str, str | None]]]]:
    """[(task key, [(sub-task key, status, failed reason)])] of a read-back group."""
    if not group:
        return []
    return [(t["agentTaskKey"], [(s["agentSubTaskKey"], s["agentSubTaskStatus"], s.get("failedReason"))
                                 for s in t["subTasks"]])
            for t in group["result"]["tasks"]]


def verdicts(group: dict | None) -> list[str]:
    """What the "failure" group shows about chaining and independence, once it has run."""
    tasks = statuses(group)
    if len(tasks) < 2 or not vira_results.finished(group):
        return ["not run yet: no verdict"]
    first, second = tasks[0][1], tasks[1][1]
    out = []
    if first[0][1] == "failed":
        later = [s for _, s, _ in first[1:]]
        out.append("sub-tasks are chained: after a failed sub-task, the rest of its task failed"
                   if later and all(s == "failed" for s in later)
                   else f"sub-tasks are NOT chained: after a failed sub-task the rest were {later}")
    else:
        out.append(f"the sub-task meant to fail didn't ({first[0][1]}): no verdict on chaining")
    done = all(s == "completed" for _, s, _ in second)
    out.append("tasks are independent: the next task still ran" if done
               else f"the next task didn't complete ({[s for _, s, _ in second]})")
    return out


def _replies(result: dict) -> list[dict]:
    """The sub-task results in a projected Jeni task result (jeni_tools.project)."""
    subs = result.get("subtasks") if isinstance(result, dict) else None
    return [s["result"] for s in subs or [] if isinstance(s, dict) and isinstance(s.get("result"), dict)]


def succeeded(result: dict) -> bool:
    """Sent, read back, and every sub-task completed (the group reads "completed" either way)."""
    subs = result.get("subtasks") if isinstance(result, dict) else None
    return (result.get("request_status") == "ok" and bool(subs)
            and all(isinstance(s, dict) and s.get("status") == "completed" for s in subs))


def outcome(result: dict) -> str:
    """A step's outcome for the printout: statuses and failure reasons only."""
    subs = [s.get("status") for s in result.get("subtasks") or [] if isinstance(s, dict)]
    reasons = [s.get("failed_reason") for s in result.get("subtasks") or [] if isinstance(s, dict)]
    reason = next((r for r in reasons if r), None) or (result.get("message") if not succeeded(result) else None)
    return (f"{result.get('group_status') or result.get('request_status')} {subs}"
            + (f" ({vira._scrub_text(str(reason))[:100]})" if reason else ""))


def job_id(result: dict) -> int | None:
    """The job's id in a projected Jeni task result (VIRA's job record has jobId)."""
    value = next((r["jobId"] for r in _replies(result) if "jobId" in r), None)
    return value if isinstance(value, int) and value > 0 else None


def has_skill(result: dict, skill: str) -> bool:
    for reply in _replies(result):
        names = [str(s.get("name", s.get("skillName", "")) if isinstance(s, dict) else s).lower()
                 for s in reply.get("skills") or []]
        if skill.lower() in names or skill.lower() in str(reply.get("skillsText", "")).lower():
            return True
    return False


def _as_dict(output) -> dict:
    """A tool's output as the callback sees it (a dict, a ToolMessage or JSON text) as a dict."""
    if isinstance(output, dict):
        return output
    text = getattr(output, "content", output)
    try:
        value = json.loads(text) if isinstance(text, str) else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class CreatedJobs(BaseCallbackHandler):
    """The jobId each create_job call returned, so the next card can be checked against it."""

    def __init__(self):
        self.names, self.ids = {}, []

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.names[run_id] = (serialized or {}).get("name") or kwargs.get("name")

    def on_tool_end(self, output, *, run_id, **kwargs):
        if self.names.pop(run_id, None) == "create_job" and job_id(_as_dict(output)):
            self.ids.append(job_id(_as_dict(output)))


def planned_decisions(title: str, created: CreatedJobs, cards: list):
    """At an approval card: approve only the planned calls (create the test job, then add Kafka to
    the job that call returned) and reject anything else."""
    def decide(request):
        out = []
        for action in request["action_requests"]:
            name, args = action["name"], action.get("args") or {}
            ok = (name == "create_job" and not created.ids
                  and str(args.get("job_title", "")).strip().lower() == title.lower()) or (
                name == "add_job_skills" and str(args.get("job_id")) in map(str, created.ids)
                and [str(x).lower() for x in args.get("skills") or []] == ["kafka"])
            cards.append((name, ok))
            out.append({"type": "approve"} if ok else
                       {"type": "reject", "message": "Not part of the write test: do only what was asked."})
        return out
    return decide


def run_writes(stamp: str, out=print, *, model=None, mode: str = "real") -> bool:
    """The write test (see the module docstring).  True if every step ran as planned.
    model and mode are for the offline test (a scripted model on the mock)."""
    import agent_kit
    import run_langgraph
    import vira_tools
    vira_tools.configure(mode)
    title, thread, cards, created_jobs = f"Jeni v2 test {stamp}", f"vira-check-writes-{stamp}", [], CreatedJobs()
    agent = run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), gate_writes=True, model=model)
    result = agent_kit.run_task(
        agent, f"Create a job titled '{title}' needing Python with 1 to 2 years of experience, "
               f"then add Kafka to it.", decide=planned_decisions(title, created_jobs, cards),
        callbacks=[created_jobs], thread_id=thread)
    created = created_jobs.ids[0] if created_jobs.ids else None
    added_to = [c["args"].get("job_id") for m in result["messages"] for c in getattr(m, "tool_calls", None) or []
                if c["name"] == "add_job_skills"]
    out(f"agent: cards {cards}")
    out(f"agent: create_job -> job {created}; add_job_skills on {added_to}")
    chained = created is not None and str(created) in map(str, added_to)
    out("  -> " + ("steps chain on real results: the skill went to the job create_job returned" if chained
                   else "the agent didn't chain on the new job's id"))
    if created is None:
        out(f"create_job was approved but no job id came back: if it runs later, close '{title}' by hand"
            if ("create_job", True) in cards else "no test job was created: nothing to clean up")
        return False
    ok = chained and [(name, True) for name in WRITE_CARDS] == cards     # those two cards, nothing else
    steps = [("make_job_private", {"job_id": created}, succeeded),
             ("get_single_job_details", {"job_id": created}, lambda r: succeeded(r) and has_skill(r, "Kafka")),
             ("remove_job_skills", {"job_id": created, "skills": ["Kafka"]}, succeeded),
             ("edit_job", {"job_id": created, "job_description": "A test job made by Jeni v2's end-to-end "
                                                                 "check. Safe to delete."}, succeeded),
             ("get_single_job_details", {"job_id": created}, lambda r: succeeded(r) and not has_skill(r, "Kafka"))]
    try:                    # each call its own VIRA session (jeni_tools.session_uuid)
        for name, args, expect in steps:
            r = jeni_tools.run(name, args)
            good = expect(r)
            ok = ok and good
            out(f"{name}: {outcome(r)} {'as expected' if good else 'NOT as expected'}")
    finally:
        r = jeni_tools.run("make_job_closed", {"job_id": created, "reason_for_closure": "Jeni v2 test, done"})
        out(f"make_job_closed (cleanup): {outcome(r)}")
        ok = ok and succeeded(r)
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check the real VIRA engine: read-only groups, then "
                                             "optionally a write test.")
    p.add_argument("--wait", type=float, default=30, help="seconds to wait for each group (default 30)")
    p.add_argument("--writes", action="store_true",
                   help="then the write test on a job of its own (only if the read-only groups ran)")
    args = p.parse_args(argv)
    via = vira_results.source() or ("api" if os.environ.get("VIRA_RESULT_LOCATION") else "db")
    needed = ("VIRA_ACTUAL_LOCATION", "VIRA_XRTOKEN",
              "VIRA_RESULT_LOCATION" if via == "api" else "TRON_POSTGRES_DSN")
    missing = [n for n in needed if not os.environ.get(n)]
    if missing:
        print(f"not set up: {', '.join(missing)} missing from .env")
        return 2
    problem = vira_results.api_problem() if via == "api" else None
    if problem:
        print(f"not set up: {problem}")
        return 2
    print(f"reading results back through the engine's {'info call' if via == 'api' else 'tables'}")
    stamp, ok = time.strftime("%Y%m%d-%H%M%S"), True
    sent = []
    for label, body in checks(stamp):
        try:
            reply = vira.execute(f"check-{label}", vira.TASK_GROUP_PATH, {}, body, mode="real")
        except Exception as exc:
            print(f"{label:8} not sent: {type(exc).__name__}")
            ok = False
            continue
        group_uuid = (reply.get("result") or {}).get("agentTaskGroupUuid") if reply.get("status") == "ok" else None
        if not group_uuid:
            print(f"{label:8} refused: http {reply.get('http_status')} {(reply.get('result') or {}).get('message')}")
            ok = False
            continue
        print(f"{label:8} accepted (http {reply.get('http_status')})")
        sent.append((label, group_uuid))
    for label, group_uuid in sent:
        group = vira_results.wait_for(group_uuid, args.wait, via=via)
        state = "ran" if vira_results.finished(group) else f"still queued after {args.wait:g}s"
        print(f"\n{label}: {state}")
        for task, subs in statuses(group):
            print(f"  {task}")
            for sub, status, reason in subs:
                print(f"    {sub}: {status}" + (f"  ({vira._scrub_text(str(reason))[:100]})" if reason else ""))
        if label == "failure":
            for line in verdicts(group):
                print(f"  -> {line}")
        ok = ok and vira_results.finished(group)
    if not ok:
        print("\nVIRA hasn't run everything: the engine fix is still needed (docs/agent-frameworks.md §15).")
        if args.writes:
            print("Write test not sent: its writes would sit queued and run later with nobody watching.")
        return 1
    if args.writes:
        print(f"\nwrite test (staging, a job of its own; reads back through the {via})")
        os.environ.setdefault("VIRA_RESULT_SOURCE", via)
        return 0 if run_writes(stamp) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
