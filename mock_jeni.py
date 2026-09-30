"""Mock of VIRA's task-group API for Jeni's tasks (the assumed endpoint in jeni_tools.PATH).

Jeni v1 hands VIRA a task group (one or more tasks, each with sub-tasks and their fields)
and reads back the group with a status per task and sub-task.  This answers in that shape
(a real reply looks like raw_data/response.js): the sub-task's own result is in
agentSubTaskResponse, and a failure has a failedReason.

Stateless and deterministic, so parallel calls and repeated runs agree: a created job's id
comes from its title, and a change (closing a job, adding skills) is reported back but not
remembered.  All data is synthetic.
"""
from __future__ import annotations

import json
import uuid
import zlib
from typing import Any, Callable, Dict

NOTE = "SYNTHETIC mock response"
_NS = uuid.UUID("6f1c2a52-6a0e-4c43-9a57-000000000000")

JOBS = {
    7001: {"jobId": 7001, "jobName": "Senior Backend Engineer", "status": "open", "isPrivate": True,
           "skills": ["Python", "Go", "PostgreSQL"], "minExp": 5, "maxExp": 8,
           "countryName": "Singapore", "vacancy": 2},
    7002: {"jobId": 7002, "jobName": "Data Analyst", "status": "closed", "isPrivate": False,
           "skills": ["SQL", "Tableau"], "minExp": 2, "maxExp": 4,
           "countryName": "Malaysia", "vacancy": 1},
    7003: {"jobId": 7003, "jobName": "Product Designer", "status": "open", "isPrivate": False,
           "skills": ["Figma", "User research"], "minExp": 3, "maxExp": 6,
           "countryName": "Singapore", "vacancy": 1},
    686412: {"jobId": 686412, "jobName": "Data Scientist", "status": "open", "isPrivate": False,
             "skills": [], "minExp": None, "maxExp": None,
             "countryName": "Singapore", "vacancy": 1},
}
USERS = [
    {"userId": 801, "firstName": "Alice", "lastName": "Johnson", "email": "alice.johnson@example.com"},
    {"userId": 802, "firstName": "Bob", "lastName": "Tan", "email": "bob.tan@example.com"},
    {"userId": 803, "firstName": "Priya", "lastName": "Nair", "email": "priya.nair@example.com"},
]
APPLICATIONS = {                       # appId -> (jobId, matchScore, stage)
    5101: (7001, 0.72, "applied"), 5102: (7001, 0.91, "applied"),
    5103: (7001, 0.85, "applied"), 5104: (7001, 0.64, "applied"),
    5201: (7002, 0.88, "shortlisted"), 5202: (7002, 0.59, "applied"),
    674320: (686412, 0.90, "applied"), 674321: (686412, 0.88, "applied"),
    674322: (686412, 0.86, "applied"), 674323: (686412, 0.84, "applied"),
    674324: (686412, 0.82, "applied"), 674325: (686412, 0.80, "applied"),
    674346: (686412, 0.78, "applied"), 674347: (686412, 0.76, "applied"),
}
SUGGESTED = {7001: [(910001, 0.93), (910002, 0.89), (910003, 0.81)],
             7002: [(920001, 0.86), (920002, 0.78)]}
SELF_SOURCED = {7001: [(930001, 0.77), (930002, 0.70)], 7002: [(940001, 0.74)]}


def db_fixtures(company_id: int) -> Dict:
    """These jobs, users and applications as db_queries.fake_db_queries fixtures, so in mock
    mode the db lookups (--tools jeni_db) return only ids this mock knows."""
    return {"company_id": company_id,
            "jobs": [{"jobId": j["jobId"], "jobName": j["jobName"], "company_id": company_id}
                     for j in JOBS.values()],
            "users": [{"userId": u["userId"], "firstname": u["firstName"],
                       "lastname": u["lastName"], "email": u["email"], "company_id": company_id}
                      for u in USERS],
            "applications": [{"app_id": app_id, "job_id": job_id}
                             for app_id, (job_id, _, _) in APPLICATIONS.items()]}


def created_job_id(title: str) -> int:
    """The id create_job gives a job with this title (7100-7999)."""
    return 7100 + zlib.crc32(title.strip().lower().encode()) % 900


def cloned_job_id(job_id: int) -> int:
    return 8000 + job_id % 1000


def created_app_id(email: str) -> int:
    return 6000 + zlib.crc32(email.strip().lower().encode()) % 1000


def _job(job_id: Any) -> Dict | None:
    if job_id in JOBS:
        return JOBS[job_id]
    if isinstance(job_id, int) and (7100 <= job_id < 8000 or 8000 <= job_id < 9000):
        return {"jobId": job_id, "jobName": "(new job)", "status": "open", "isPrivate": True,
                "skills": [], "minExp": None, "maxExp": None, "countryName": None, "vacancy": 1}
    return None


def _application(app_id: int) -> Dict:
    job_id, score, stage = APPLICATIONS[app_id]
    return {"appId": app_id, "jobId": job_id, "candidateName": f"Mock Candidate {app_id}",
            "candidateEmail": f"candidate{app_id}@example.com", "matchScore": score, "stage": stage}


# --- one handler per task: fields -> (agentSubTaskResponse, failedReason) ------------
Result = tuple[Dict | None, str | None]


def _no_job(job_id) -> Result:
    return None, f"Job {job_id} not found"


def _create_job(f) -> Result:
    job_id = created_job_id(f["job_title"])
    return {"jobId": job_id, "jobName": f["job_title"], "skills": f.get("skills") or [],
            "minExp": f.get("min_exp"), "maxExp": f.get("max_exp"), "status": "open",
            "isPrivate": True, "message": "Job created successfully"}, None


def _edit_job(f) -> Result:
    if not _job(f["job_id"]):
        return _no_job(f["job_id"])
    changed = sorted(k for k, v in f.items() if k != "job_id" and v is not None)
    return {"jobId": f["job_id"], "updatedFields": changed, "message": "Job updated"}, None


def _skills(add: bool) -> Callable[[Dict], Result]:
    def handler(f) -> Result:
        job = _job(f["job_id"])
        if not job:
            return _no_job(f["job_id"])
        current = list(job["skills"])
        given = [s for s in f["skills"] if isinstance(s, str)]
        if add:
            skills = current + [s for s in given if s.lower() not in {c.lower() for c in current}]
        else:
            skills = [c for c in current if c.lower() not in {s.lower() for s in given}]
        return {"jobId": f["job_id"], "skills": skills}, None
    return handler


def _clone_job(f) -> Result:
    if not _job(f["job_id"]):
        return _no_job(f["job_id"])
    return {"jobId": cloned_job_id(f["job_id"]), "clonedFromJobId": f["job_id"],
            "message": "Job cloned"}, None


def _transfer_job_ownership(f) -> Result:
    if not _job(f["job_id"]):
        return _no_job(f["job_id"])
    email = str(f["new_owner_user_email"]).lower()
    user = next((u for u in USERS if u["email"] == email), None)
    if not user:
        return None, "No active user with that email in your company"
    return {"jobId": f["job_id"], "newOwnerUserId": user["userId"],
            "message": "Job ownership transferred"}, None


def _add_job_collaborators(f) -> Result:
    if not _job(f["job_id"]):
        return _no_job(f["job_id"])
    known = {u["userId"] for u in USERS}
    passed = [{"error": False, "jobId": f["job_id"], "record": None, "userId": uid,
               "message": "This user successfully added as a collaborator of this job!"}
              for uid in f["user_ids"] if uid in known]
    failed = [{"error": True, "jobId": f["job_id"], "record": None, "userId": uid,
               "message": "User not found in your company"}
              for uid in f["user_ids"] if uid not in known]
    return {"failedArr": failed, "passedArr": passed, "existingArr": []}, None


def _publish_job_to_linkedin(f) -> Result:
    job = _job(f["job_id"])
    if not job:
        return _no_job(f["job_id"])
    if job["status"] != "open" or job["isPrivate"]:
        return None, "Job must be open and public before publishing to LinkedIn"
    return {"jobId": f["job_id"], "message": "Job published to LinkedIn"}, None


def _visibility(f) -> Result:
    if not _job(f["job_id"]):
        return _no_job(f["job_id"])
    return {"jobId": f["job_id"], "isPrivate": bool(f["is_private"])}, None


def _status(status: str) -> Callable[[Dict], Result]:
    def handler(f) -> Result:
        if not _job(f["job_id"]):
            return _no_job(f["job_id"])
        out = {"jobId": f["job_id"], "status": status}
        if f.get("reason_for_closure"):
            out["reasonForClosure"] = f["reason_for_closure"]
        return out, None
    return handler


def _per_application(stage: str) -> Callable[[Dict], Result]:
    def handler(f) -> Result:
        # User-driven action: the user chooses which applications to act on, so accept the
        # ids they give rather than gating on the fixture.  (A real backend still checks the
        # ids belong to the recruiter's company; the db validate_app_ids tool covers that.)
        passed = [{"error": False, "appId": a, "stage": stage, "message": f"Application {stage}"}
                  for a in f["app_ids"]]
        return {"failedArr": [], "passedArr": passed, "existingArr": []}, None
    return handler


def _share_application(f) -> Result:
    emails = [e for e in f["emails"] if isinstance(e, str)]
    passed = [{"error": False, "appId": a, "sharedWithCount": len(emails)}
              for a in f["app_ids"] if a in APPLICATIONS]
    failed = [{"error": True, "appId": a, "message": "Application not found"}
              for a in f["app_ids"] if a not in APPLICATIONS]
    return {"failedArr": failed, "passedArr": passed, "existingArr": []}, None


def _page(f, rows: list) -> list:
    skip, limit = f.get("skip") or 0, f.get("limit") or 10
    return rows[skip:skip + limit]


def _get_applications(f) -> Result:
    key = str(f.get("search_key") or "").lower()
    rows = [_application(a) for a in APPLICATIONS
            if f.get("job_id") in (None, APPLICATIONS[a][0])]
    rows = [r for r in rows if key in r["candidateName"].lower() or key in r["candidateEmail"]]
    return {"applications": _page(f, rows), "total": len(rows)}, None


def _get_single_application_details(f) -> Result:
    if f["app_id"] not in APPLICATIONS:
        return None, f"Application {f['app_id']} not found"
    return {**_application(f["app_id"]), "phone": f"+65 8000 {f['app_id']}",
            "skills": ["Python", "SQL"], "experienceYears": 6}, None


def _create_application_to_job(f) -> Result:
    if not _job(f["job_id"]):
        return _no_job(f["job_id"])
    return {"appId": created_app_id(str(f["candidate_email"])), "jobId": f["job_id"],
            "message": "Application created"}, None


def _get_single_job_details(f) -> Result:
    job = _job(f["job_id"])
    return (dict(job), None) if job else _no_job(f["job_id"])


def _search_users(f) -> Result:
    key = str(f.get("search_key") or "").lower()
    rows = [u for u in USERS
            if key in f"{u['firstName']} {u['lastName']}".lower() or key in u["email"]]
    return {"users": _page(f, rows), "total": len(rows)}, None


def _candidates(pool: Dict[int, list]) -> Callable[[Dict], Result]:
    def handler(f) -> Result:
        if not _job(f["job_id"]):
            return _no_job(f["job_id"])
        return {"jobId": f["job_id"], "candidates": [{"profileId": p, "matchScore": s}
                                                      for p, s in pool.get(f["job_id"], [])]}, None
    return handler


HANDLERS: Dict[str, Callable[[Dict], Result]] = {
    "task_create_job": _create_job,
    "task_edit_job": _edit_job,
    "task_add_job_skills": _skills(add=True),
    "task_remove_job_skills": _skills(add=False),
    "task_clone_job": _clone_job,
    "task_transfer_job_ownership": _transfer_job_ownership,
    "task_add_job_collaborators": _add_job_collaborators,
    "task_publish_job_to_linkedin": _publish_job_to_linkedin,
    "task_make_job_private": _visibility,
    "task_make_job_public": _visibility,
    "task_make_job_closed": _status("closed"),
    "task_make_job_open": _status("open"),
    "task_reject_multiple_application": _per_application("rejected"),
    "task_shortlist_multiple_application": _per_application("shortlisted"),
    "task_share_application": _share_application,
    "task_get_applications": _get_applications,
    "task_get_single_application_details": _get_single_application_details,
    "task_create_application_to_job": _create_application_to_job,
    "task_get_single_job_details": _get_single_job_details,
    "task_search_users": _search_users,
    "task_get_suggested_candidates_for_a_job": _candidates(SUGGESTED),
    "task_get_self_sourcing_candidates_for_a_job": _candidates(SELF_SOURCED),
}


# --- the task group ------------------------------------------------------------------
def _run_sub_task(task_name: str, sub: Dict) -> tuple[str, Dict | None, str | None]:
    fields = {f.get("field_name"): f.get("field_value") for f in sub.get("fields") or []}
    missing = [f["field_name"] for f in sub.get("fields") or []
               if f.get("mandatory") and f.get("field_value") in (None, "", [])]
    handler = HANDLERS.get(task_name)
    if handler is None:
        return "failed", None, f"Unknown task {task_name}"
    if missing:
        return "failed", None, f"Missing mandatory field(s): {', '.join(missing)}"
    response, failed_reason = handler(fields)
    return ("failed" if failed_reason else "completed"), response, failed_reason


def run_task_group(body: Dict) -> Dict:
    tasks = body.get("tasks") or []
    if not tasks:
        return {"status": "error", "http_status": 400, "result": {"message": "tasks is required"}}
    seed = json.dumps(body, sort_keys=True, default=str)
    group_id = 1000 + zlib.crc32(seed.encode()) % 9000
    stamp = "2026-09-30T00:00:00.000Z"
    out_tasks, statuses = [], []
    for n, task in enumerate(tasks, 1):
        subs = []
        for m, sub in enumerate(task.get("sub_tasks") or [], 1):
            status, response, reason = _run_sub_task(task.get("task_name", ""), sub)
            statuses.append(status)
            subs.append({
                "agentSubTaskId": str(group_id * 100 + n * 10 + m),
                "agentSubTaskUuid": str(uuid.uuid5(_NS, f"{seed}/{n}/{m}")),
                "agentTaskId": str(group_id * 10 + n),
                "agentSubTaskKey": sub.get("sub_task_name"),
                "agentSubTaskName": sub.get("sub_task_name"),
                "agentSubTaskDescription": sub.get("description"),
                "agentSubTaskNo": str(m), "agentSubTaskStatus": status,
                "createdAt": stamp, "updatedAt": stamp,
                "agentSubTaskResponse": response, "failedReason": reason})
        out_tasks.append({
            "agentTaskId": str(group_id * 10 + n),
            "agentTaskUuid": str(uuid.uuid5(_NS, f"{seed}/{n}")),
            "agentTaskGroupId": str(group_id),
            "agentTaskKey": task.get("task_name"), "agentTaskName": task.get("task_name"),
            "agentTaskDescription": task.get("description"), "agentTaskNo": str(n),
            "agentTaskStatus": "failed" if any(s["agentSubTaskStatus"] == "failed" for s in subs)
                               else "completed",
            "createdAt": stamp, "updatedAt": stamp, "subTasks": subs})
    name = body.get("task_group_name") or "Jeni task group"
    return {"status": "ok", "http_status": 200, "result": {
        "agentTaskGroupUuid": str(uuid.uuid5(_NS, seed)),
        "agentTaskGroupId": str(group_id),
        "agentTaskGroupKey": name.lower(), "agentTaskGroupName": name,
        "agentTaskGroupStatus": "failed" if "failed" in statuses else "completed",
        "agentSessionUuid": body.get("agent_session_uuid"),
        "agentTaskGroupDescription": None, "agentTaskGroupNo": "1",
        "createdAt": stamp, "updatedAt": stamp,
        "creatorName": "Mock Recruiter",
        "tasks": out_tasks, "_note": NOTE}}
