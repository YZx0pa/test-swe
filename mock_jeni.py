"""Mock of VIRA's task-group API for Jeni's tasks (the assumed endpoint in jeni_tools.PATH).

Jeni v1 hands VIRA a task group (one or more tasks, each with sub-tasks and their fields)
and reads back the group with a status per task and sub-task.  This answers in that shape
(a real reply looks like raw_data/response.js): the sub-task's own result is in
agentSubTaskResponse, and a failure has a failedReason.

Stateless and deterministic by default, so parallel calls and repeated runs agree: every task
group starts from the fixtures below, a created job's id comes from its title, and a change
(closing a job, adding skills) is reported back but not remembered.  The demo
(demo/jeni_graph.py) calls remember_changes() instead: one State then lives across calls, so
a job made public can be published, and a created job is found by title.  All data is
synthetic.
"""
from __future__ import annotations

import datetime
import json
import threading
import time
import uuid
import zlib
from typing import Any, Callable, Dict

NOTE = "SYNTHETIC mock response"
_NS = uuid.UUID("6f1c2a52-6a0e-4c43-9a57-000000000000")

JOBS = {
    7001: {"jobId": 7001, "openDate": "2026-08-21", "jobName": "Senior Backend Engineer", "status": "open", "isPrivate": True,
           "skills": ["Python", "Go", "PostgreSQL"], "minExp": 5, "maxExp": 8,
           "countryName": "Singapore", "vacancy": 2},
    7002: {"jobId": 7002, "openDate": "2026-07-02", "jobName": "Data Analyst", "status": "closed", "isPrivate": False,
           "skills": ["SQL", "Tableau"], "minExp": 2, "maxExp": 4,
           "countryName": "Malaysia", "vacancy": 1},
    7003: {"jobId": 7003, "openDate": "2026-09-10", "jobName": "Product Designer", "status": "open", "isPrivate": False,
           "skills": ["Figma", "User research"], "minExp": 3, "maxExp": 6,
           "countryName": "Singapore", "vacancy": 1},
    686412: {"jobId": 686412, "openDate": "2026-09-15", "jobName": "Data Scientist", "status": "open", "isPrivate": False,
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
ROLES = {1: "administrator", 5: "team member"}

# The activity log keeps ids, skills, titles and flags: never names, emails or search text.
ACTIVITY_FIELDS = {"job_id", "app_id", "app_ids", "user_ids", "role_id", "skills", "job_title",
                   "is_private", "reason_for_closure", "min_exp", "max_exp", "vacancy",
                   "country_name", "limit", "skip"}
ACTIVITY_LIMIT = 200
CREATES = {"task_create_job": "jobId", "task_clone_job": "jobId",
           "task_create_application_to_job": "appId"}


class State:
    """Jobs, applications and job teams, as the task handlers read and change them.

    A task group gets a fresh State unless remember_changes() made one that lives across
    calls (`kept`).  Only a kept State is shared between threads, so it has a lock.
    """

    def __init__(self, kept: bool = False):
        self.kept = kept
        self.lock = threading.Lock()
        self.db: Dict[str, Any] = {"company_id": None, "jobs": [], "users": [], "applications": []}
        self.reset()

    def reset(self) -> None:
        """Back to the fixtures.  The db lists are refilled in place: fake_db_queries keeps them."""
        with self.lock:
            self.jobs = {i: {**j, "skills": list(j["skills"])} for i, j in JOBS.items()}
            self.apps = {a: {"jobId": j, "matchScore": s, "stage": st}
                         for a, (j, s, st) in APPLICATIONS.items()}
            self.team: Dict[int, Dict[int, int]] = {}      # jobId -> {userId: role_id}
            self.activity: list[dict] = []                  # every sub-task, oldest first
            self._sync_db()

    def db_fixtures(self, company_id: int) -> Dict:
        """This State as db_queries.fake_db_queries fixtures, kept current as tasks change it."""
        with self.lock:
            self.db["company_id"] = company_id
            self._sync_db()
        return self.db

    def _sync_db(self) -> None:
        cid = self.db["company_id"]
        self.db["jobs"][:] = [{"jobId": j["jobId"], "jobName": j["jobName"], "company_id": cid,
                               "openDate": _date(j.get("openDate"))}
                              for j in self.jobs.values()]
        self.db["users"][:] = [{"userId": u["userId"], "firstname": u["firstName"],
                                "lastname": u["lastName"], "email": u["email"], "company_id": cid}
                               for u in USERS]
        self.db["applications"][:] = [{"app_id": a, "job_id": r["jobId"]}
                                      for a, r in self.apps.items()]

    def _log(self, task_name: str, fields: Dict, status: str, response: Dict | None,
             reason: str | None) -> None:
        kept = {k: v for k, v in fields.items() if k in ACTIVITY_FIELDS and v not in (None, "", [])}
        if isinstance(fields.get("emails"), list):
            kept["recipients"] = len(fields["emails"])
        entry = {"at": time.strftime("%H:%M:%S"), "task": task_name.removeprefix("task_"),
                 "status": status, "reason": reason, "fields": kept}
        if task_name in CREATES and response:
            entry["created"] = response.get(CREATES[task_name])
        self.activity = (self.activity + [entry])[-ACTIVITY_LIMIT:]

    def snapshot(self) -> Dict:
        """A copy to display (demo/app.py): jobs with their team and applications, and the
        activity log.  Colleagues by name; candidates by application id only."""
        people = {u["userId"]: f"{u['firstName']} {u['lastName']}" for u in USERS}
        with self.lock:
            jobs = [{"jobId": job_id, "jobName": job["jobName"], "status": job["status"],
                     "isPrivate": job["isPrivate"], "skills": list(job["skills"]),
                     "linkedIn": bool(job.get("publishedToLinkedIn")),
                     "owner": people.get(job.get("ownerUserId")),
                     "team": [{"userId": u, "name": people.get(u, f"user {u}"),
                               "role": ROLES.get(r, str(r))}
                              for u, r in self.team.get(job_id, {}).items()],
                     "applications": [{"appId": a, "matchScore": r["matchScore"], "stage": r["stage"]}
                                      for a, r in sorted(self.apps.items()) if r["jobId"] == job_id]}
                    for job_id, job in sorted(self.jobs.items())]
            activity = [{**e, "fields": dict(e["fields"])} for e in self.activity]
        return {"jobs": jobs, "activity": activity}


_kept: State | None = None


def remember_changes() -> State:
    """From now on one State lives across calls (the demo).  Returns it."""
    global _kept
    if _kept is None:
        _kept = State(kept=True)
    return _kept


def forget_changes() -> None:
    """Back to stateless: every task group starts from the fixtures again."""
    global _kept
    _kept = None


def db_fixtures(company_id: int) -> Dict:
    """These jobs, users and applications as db_queries.fake_db_queries fixtures, so in mock
    mode the db lookups (--tools jeni_db) return only ids this mock knows.  A kept State
    has its own, which follow its changes (State.db_fixtures)."""
    return State().db_fixtures(company_id)


def created_job_id(title: str) -> int:
    """The id create_job gives a job with this title (7100-7999)."""
    return 7100 + zlib.crc32(title.strip().lower().encode()) % 900


def cloned_job_id(job_id: int) -> int:
    return 8000 + job_id % 1000


def created_app_id(email: str) -> int:
    return 6000 + zlib.crc32(email.strip().lower().encode()) % 1000


def _date(text: str | None) -> datetime.date | None:
    return datetime.date.fromisoformat(text) if text else None


def _new_job(job_id: int, name: str, **fields) -> Dict:
    return {"jobId": job_id, "jobName": name, "status": "open", "isPrivate": True,
            "skills": [], "minExp": None, "maxExp": None, "countryName": None, "vacancy": 1,
            "openDate": time.strftime("%Y-%m-%d"), **fields}


def _job(s: State, job_id: Any) -> Dict | None:
    if job_id in s.jobs:
        return s.jobs[job_id]
    if not s.kept and isinstance(job_id, int) and 7100 <= job_id < 9000:
        return _new_job(job_id, "(new job)")    # created or cloned in a call this mock forgot
    return None


def _application(s: State, app_id: int) -> Dict:
    a = s.apps[app_id]
    return {"appId": app_id, "jobId": a["jobId"],
            "candidateName": a.get("candidateName", f"Mock Candidate {app_id}"),
            "candidateEmail": a.get("candidateEmail", f"candidate{app_id}@example.com"),
            "matchScore": a["matchScore"], "stage": a["stage"]}


# --- one handler per task: (fields, state) -> (agentSubTaskResponse, failedReason) ----
Result = tuple[Dict | None, str | None]
Handler = Callable[[Dict, State], Result]
# edit_job's fields -> the job's keys
JOB_KEYS = {"job_title": "jobName", "job_description": "jobDescription",
            "job_requirements": "jobRequirements", "country_name": "countryName",
            "min_exp": "minExp", "max_exp": "maxExp", "min_salary": "minSalary",
            "max_salary": "maxSalary"}


def _no_job(job_id) -> Result:
    return None, f"Job {job_id} not found"


def _create_job(f, s) -> Result:
    job_id = created_job_id(f["job_title"])
    while job_id in s.jobs:             # the same title again (a kept State)
        job_id += 1
    s.jobs[job_id] = _new_job(job_id, f["job_title"], skills=list(f.get("skills") or []),
                              minExp=f.get("min_exp"), maxExp=f.get("max_exp"),
                              countryName=f.get("country_name"))
    return {"jobId": job_id, "jobName": f["job_title"], "skills": f.get("skills") or [],
            "minExp": f.get("min_exp"), "maxExp": f.get("max_exp"), "status": "open",
            "isPrivate": True, "message": "Job created successfully"}, None


def _edit_job(f, s) -> Result:
    job = _job(s, f["job_id"])
    if not job:
        return _no_job(f["job_id"])
    changed = sorted(k for k, v in f.items() if k != "job_id" and v is not None)
    job.update({JOB_KEYS.get(k, k): f[k] for k in changed})
    return {"jobId": f["job_id"], "updatedFields": changed, "message": "Job updated"}, None


def _skills(add: bool) -> Handler:
    def handler(f, s) -> Result:
        job = _job(s, f["job_id"])
        if not job:
            return _no_job(f["job_id"])
        current = list(job["skills"])
        given = [x for x in f["skills"] if isinstance(x, str)]
        if add:
            skills = current + [x for x in given if x.lower() not in {c.lower() for c in current}]
        else:
            skills = [c for c in current if c.lower() not in {x.lower() for x in given}]
        job["skills"] = skills
        return {"jobId": f["job_id"], "skills": skills}, None
    return handler


def _clone_job(f, s) -> Result:
    job = _job(s, f["job_id"])
    if not job:
        return _no_job(f["job_id"])
    new_id = cloned_job_id(f["job_id"])
    s.jobs[new_id] = _new_job(new_id, job["jobName"], skills=list(job["skills"]),
                              minExp=job["minExp"], maxExp=job["maxExp"],
                              countryName=job["countryName"], vacancy=job["vacancy"])
    return {"jobId": new_id, "clonedFromJobId": f["job_id"], "message": "Job cloned"}, None


def _transfer_job_ownership(f, s) -> Result:
    job = _job(s, f["job_id"])
    if not job:
        return _no_job(f["job_id"])
    email = str(f["new_owner_user_email"]).lower()
    user = next((u for u in USERS if u["email"] == email), None)
    if not user:
        return None, "No active user with that email in your company"
    job["ownerUserId"] = user["userId"]
    return {"jobId": f["job_id"], "newOwnerUserId": user["userId"],
            "message": "Job ownership transferred"}, None


def _add_job_collaborators(f, s) -> Result:
    if not _job(s, f["job_id"]):
        return _no_job(f["job_id"])
    known = {u["userId"] for u in USERS}
    team = s.team.setdefault(f["job_id"], {})
    already = set(team)
    passed, failed, existing = [], [], []
    for uid in f["user_ids"]:
        if uid not in known:
            failed.append({"error": True, "jobId": f["job_id"], "record": None, "userId": uid,
                           "message": "User not found in your company"})
        elif uid in already:
            existing.append({"error": False, "jobId": f["job_id"], "record": None, "userId": uid,
                             "message": "This user is already a collaborator of this job"})
        else:
            team[uid] = f.get("role_id")
            passed.append({"error": False, "jobId": f["job_id"], "record": None, "userId": uid,
                           "message": "This user successfully added as a collaborator of this job!"})
    return {"failedArr": failed, "passedArr": passed, "existingArr": existing}, None


def _publish_job_to_linkedin(f, s) -> Result:
    job = _job(s, f["job_id"])
    if not job:
        return _no_job(f["job_id"])
    if job["status"] != "open" or job["isPrivate"]:
        return None, "Job must be open and public before publishing to LinkedIn"
    job["publishedToLinkedIn"] = True
    return {"jobId": f["job_id"], "message": "Job published to LinkedIn"}, None


def _visibility(f, s) -> Result:
    job = _job(s, f["job_id"])
    if not job:
        return _no_job(f["job_id"])
    job["isPrivate"] = bool(f["is_private"])
    return {"jobId": f["job_id"], "isPrivate": bool(f["is_private"])}, None


def _status(status: str) -> Handler:
    def handler(f, s) -> Result:
        job = _job(s, f["job_id"])
        if not job:
            return _no_job(f["job_id"])
        job["status"] = status
        out = {"jobId": f["job_id"], "status": status}
        if f.get("reason_for_closure"):
            out["reasonForClosure"] = job["reasonForClosure"] = f["reason_for_closure"]
        return out, None
    return handler


def _per_application(stage: str) -> Handler:
    def handler(f, s) -> Result:
        # User-driven action: the user chooses which applications to act on, so accept the
        # ids they give rather than gating on the fixture.  (A real backend still checks the
        # ids belong to the recruiter's company; the db validate_app_ids tool covers that.)
        passed = [{"error": False, "appId": a, "stage": stage, "message": f"Application {stage}"}
                  for a in f["app_ids"]]
        for a in f["app_ids"]:
            if a in s.apps:                     # one this mock holds: it moves to the new stage
                s.apps[a]["stage"] = stage
        return {"failedArr": [], "passedArr": passed, "existingArr": []}, None
    return handler


def _share_application(f, s) -> Result:
    emails = [e for e in f["emails"] if isinstance(e, str)]
    passed = [{"error": False, "appId": a, "sharedWithCount": len(emails)}
              for a in f["app_ids"] if a in s.apps]
    failed = [{"error": True, "appId": a, "message": "Application not found"}
              for a in f["app_ids"] if a not in s.apps]
    return {"failedArr": failed, "passedArr": passed, "existingArr": []}, None


def _page(f, rows: list) -> list:
    skip, limit = f.get("skip") or 0, f.get("limit") or 10
    return rows[skip:skip + limit]


def _get_applications(f, s) -> Result:
    key = str(f.get("search_key") or "").lower()
    rows = [_application(s, a) for a, r in s.apps.items()
            if f.get("job_id") in (None, r["jobId"])]
    rows = [r for r in rows if key in r["candidateName"].lower() or key in r["candidateEmail"]]
    return {"applications": _page(f, rows), "total": len(rows)}, None


def _get_single_application_details(f, s) -> Result:
    if f["app_id"] not in s.apps:
        return None, f"Application {f['app_id']} not found"
    return {**_application(s, f["app_id"]), "phone": f"+65 8000 {f['app_id']}",
            "skills": ["Python", "SQL"], "experienceYears": 6}, None


def _create_application_to_job(f, s) -> Result:
    if not _job(s, f["job_id"]):
        return _no_job(f["job_id"])
    app_id = created_app_id(str(f["candidate_email"]))
    while app_id in s.apps:             # the same email again (a kept State)
        app_id += 1
    s.apps[app_id] = {"jobId": f["job_id"], "matchScore": None, "stage": "applied",
                      "candidateName": str(f.get("candidate_name") or ""),
                      "candidateEmail": str(f["candidate_email"])}
    return {"appId": app_id, "jobId": f["job_id"], "message": "Application created"}, None


def _get_single_job_details(f, s) -> Result:
    job = _job(s, f["job_id"])
    if not job:
        return _no_job(f["job_id"])
    out = {**job, "skills": list(job["skills"])}
    if s.team.get(f["job_id"]):
        out["collaborators"] = [{"userId": u, "roleId": r} for u, r in s.team[f["job_id"]].items()]
    return out, None


def _search_users(f, s) -> Result:
    key = str(f.get("search_key") or "").lower()
    rows = [u for u in USERS
            if key in f"{u['firstName']} {u['lastName']}".lower() or key in u["email"]]
    return {"users": _page(f, rows), "total": len(rows)}, None


def _candidates(pool: Dict[int, list]) -> Handler:
    def handler(f, s) -> Result:
        if not _job(s, f["job_id"]):
            return _no_job(f["job_id"])
        return {"jobId": f["job_id"], "candidates": [{"profileId": p, "matchScore": score}
                                                      for p, score in pool.get(f["job_id"], [])]}, None
    return handler


HANDLERS: Dict[str, Handler] = {
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
def _run_sub_task(task_name: str, sub: Dict, s: State) -> tuple[str, Dict | None, str | None]:
    fields = {f.get("field_name"): f.get("field_value") for f in sub.get("fields") or []}
    missing = [f["field_name"] for f in sub.get("fields") or []
               if f.get("mandatory") and f.get("field_value") in (None, "", [])]
    handler = HANDLERS.get(task_name)
    if handler is None:
        status, response, reason = "failed", None, f"Unknown task {task_name}"
    elif missing:
        status, response, reason = "failed", None, f"Missing mandatory field(s): {', '.join(missing)}"
    else:
        response, reason = handler(fields, s)
        status = "failed" if reason else "completed"
    s._log(task_name, fields, status, response, reason)
    return status, response, reason


def run_task_group(body: Dict) -> Dict:
    tasks = body.get("tasks") or []
    if not tasks:
        return {"status": "error", "http_status": 400, "result": {"message": "tasks is required"}}
    state = _kept or State()
    with state.lock:
        group = _run_group(body, tasks, state)
        state._sync_db()
    return group


def _run_group(body: Dict, tasks: list, state: State) -> Dict:
    seed = json.dumps(body, sort_keys=True, default=str)
    group_id = 1000 + zlib.crc32(seed.encode()) % 9000
    stamp = "2026-09-30T00:00:00.000Z"
    out_tasks, statuses = [], []
    for n, task in enumerate(tasks, 1):
        subs = []
        for m, sub in enumerate(task.get("sub_tasks") or [], 1):
            status, response, reason = _run_sub_task(task.get("task_name", ""), sub, state)
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
