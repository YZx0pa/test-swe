"""Jeni requests for compare_agents.py --suite jeni, each with a check over the audit log.

A check reads what reached (mock) VIRA, not what the model said.  Writes must match
exactly; reads are free, since looking something up first is fine.  Pure functions: a
run is anything with .audit (audit-log entries) and .commands.
"""
from __future__ import annotations

import jeni_tools
import mock_jeni


def fields(entry: dict) -> dict:
    """{field_name: field_value} of an audit entry's one-task group."""
    [task] = entry["body"]["tasks"]
    return {f["field_name"]: f.get("field_value") for f in task["sub_tasks"][0]["fields"]}


def writes(run) -> list[dict]:
    return [a for a in run.audit if a["command"].replace("-", "_") not in jeni_tools.READ_ONLY]


def _commands(entries: list[dict]) -> list[str]:
    return [a["command"] for a in entries]


def _read(run, command: str, **expected) -> bool:
    return any(a["command"] == command and all(fields(a).get(k) == v for k, v in expected.items())
               for a in run.audit)


def _lower(values) -> set[str]:
    return {str(v).strip().lower() for v in values or []}


def check_job_details(run):
    ok = run.commands == ["get-single-job-details"] and fields(run.audit[0])["job_id"] == 7001
    return ok, "one get_single_job_details for job 7001"


def check_create_then_skill(run):
    done = writes(run)
    ok = _commands(done) == ["create-job", "add-job-skills"]
    if ok:
        create, add = fields(done[0]), fields(done[1])
        title = str(create.get("job_title") or "")
        ok = ("data engineer" in title.lower() and {"python", "sql"} <= _lower(create.get("skills"))
              and (create.get("min_exp"), create.get("max_exp")) == (3, 5)
              and add["job_id"] == mock_jeni.created_job_id(title)
              and "spark" in _lower(add["skills"]))
    return ok, "create the job, then add Spark using the new job's id"


def check_assign(run):
    done = writes(run)
    ok = (_commands(done) == ["add-job-collaborators"] and _read(run, "search-users")
          and fields(done[0]) == {"job_id": 7001, "user_ids": [802], "role_id": 5})
    return ok, "search for Bob, then add user 802 as a team member (role 5), no transfer"


def top_applicants(job_id: int, n: int) -> list[int]:
    ranked = sorted((a for a, (j, _, _) in mock_jeni.APPLICATIONS.items() if j == job_id),
                    key=lambda a: -mock_jeni.APPLICATIONS[a][1])
    return sorted(ranked[:n])


def check_shortlist(run):
    done = writes(run)
    best = top_applicants(7001, 2)
    ok = (_commands(done) == ["shortlist-multiple-application"]
          and _read(run, "get-applications", job_id=7001)
          and sorted(fields(done[0])["app_ids"]) == best)
    return ok, f"read job 7001's applications, then shortlist the top two ({best[0]}, {best[1]})"


def check_add_candidate(run):
    done = writes(run)
    ok = _commands(done) == ["create-application-to-job"]
    if ok:
        f = fields(done[0])        # name and email are masked in the audit log
        ok = f["job_id"] == 7001 and f.get("candidate_name") and f.get("candidate_email")
    return bool(ok), "one create_application_to_job on job 7001 with the name and email given"


def check_share_no_email(run):
    return writes(run) == [], "no share: the recipient's email wasn't given, so ask for it"


def check_unsupported(run):
    return writes(run) == [], "nothing changed: sharing a job isn't supported"


TASKS = {
    "job_details": ("Show me the details of job 7001.", check_job_details),
    "create_then_skill": ("Create a Data Engineer job needing Python and SQL with 3 to 5 years of "
                          "experience, then add Spark as a skill to it.", check_create_then_skill),
    "assign_team_member": ("Assign job 7001 to Bob as a team member.", check_assign),
    "shortlist_top2": ("Shortlist the two applicants with the highest match scores for job 7001.",
                       check_shortlist),
    "add_candidate": ("Add Maya Lim (maya.lim@example.com) as a candidate to job 7001.",
                      check_add_candidate),
    "share_no_email": ("Share the CV of applicant 5102 with the hiring manager.",
                       check_share_no_email),
    "unsupported": ("Share job 7001 on Facebook.", check_unsupported),
}
