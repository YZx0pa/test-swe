#!/usr/bin/env python3
"""Check the real VIRA engine end to end, with read-only task groups only.

    python vira_check.py              # send the checks, wait up to 30 s, report what ran
    python vira_check.py --wait 60

It sends three task groups through recruiter_cli.execute (real mode: VIRA_ACTUAL_LOCATION with
VIRA_XRTOKEN, audited) and reads each one back (vira_results): through the engine's info call
when VIRA_RESULT_LOCATION is set, from its tables on TRON_POSTGRES_DSN otherwise:

  runs        v2's own payload for search_users: does a group run at all?
  engine      the engine's sample vocabulary: sub_task_get_job_description for a title (text only).
  failure     two tasks in one group.  The first has a sub-task that must fail (no job title)
              followed by one that would succeed alone; the second is a user search.  It shows
              whether a failed sub-task stops the rest of its task (chained) and whether a
              failed task stops the next task (independent).

Nothing here creates or changes a job, an application or a user.  It prints statuses and
failure reasons of these groups only, never their contents, the URL or the token.
Exit status: 0 everything ran, 1 something is still queued or failed to send, 2 not set up.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import recruiter_cli as vira       # loads .env by path, as the runners do
import jeni_tools
import vira_results


def _group(name: str, session: str, tasks: list) -> dict:
    return {"agent_session_uuid": session, "task_group_name": name, "tasks": tasks}


def _setup_job(*subs) -> dict:
    return {"task_name": "task_setup_job", "sub_tasks": [
        {"sub_task_name": name, "fields": fields} for name, fields in subs]}


def checks(stamp: str) -> list[tuple[str, dict]]:
    session = jeni_tools.session_uuid(f"vira-check-{stamp}")
    search = jeni_tools.payload(jeni_tools.catalog()["search_users"], {"search_key": "a", "limit": 3},
                                session)
    title = [{"field_name": "job_title", "field_value": "Software Engineer"}]
    return [
        ("runs", {**search, "task_group_name": f"Jeni v2 check {stamp} - runs"}),
        ("engine", _group(f"Jeni v2 check {stamp} - engine", session,
                          [_setup_job(("sub_task_get_job_description", title))])),
        ("failure", _group(f"Jeni v2 check {stamp} - failure", session, [
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check the real VIRA engine with read-only task groups.")
    p.add_argument("--wait", type=float, default=30, help="seconds to wait for each group (default 30)")
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
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
