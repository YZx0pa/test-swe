"""Reading a VIRA engine task group's result back.

The engine answers a task group at once ("We are processing your tasks…") and runs it in the
background, so jeni_tools.run waits here for the outcome and then answers as if VIRA had replied
with it.  VIRA_RESULT_SOURCE says where the outcome is read:

  api     the engine's own info call (what its task panel reads, handleGetTaskGroupInfo):
          GET VIRA_RESULT_LOCATION with "{uuid}" filled in, with the xrtoken, only on the
          engine's own host;
  db      the engine's own tables (hris.agenttaskgroup, agenttask, agentsubtask) on
          TRON_POSTGRES_DSN, by the group's uuid: read-only, and only the group just sent;
  (unset) nothing is read, and the task is reported as queued.

VIRA_RESULT_WAIT is how long to wait, in seconds (default 30; the engine used to finish a group
in about 3).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

UNFINISHED = {"queued", "pending", "processing", "in_progress", "running"}
POLL_SECONDS = 1.0

GROUP_SQL = """
    SELECT g.agent_task_group_status AS group_status,
           t.agent_task_no AS task_no, t.agent_task_key AS task_key, t.agent_task_status AS task_status,
           s.agent_sub_task_no AS sub_no, s.agent_sub_task_key AS sub_key,
           s.agent_sub_task_status AS sub_status, s.agent_sub_task_response AS sub_response,
           s.failed_reason AS failed_reason
    FROM hris.agenttaskgroup g
    JOIN hris.agenttask t ON t.agent_task_group_id = g.agent_task_group_id
    JOIN hris.agentsubtask s ON s.agent_task_id = t.agent_task_id
    WHERE g.agent_task_group_uuid = $1::uuid
    ORDER BY t.agent_task_no, s.agent_sub_task_no
"""


def source() -> str:
    return os.environ.get("VIRA_RESULT_SOURCE", "").strip().lower()


def wait_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("VIRA_RESULT_WAIT", "30")))
    except ValueError:
        return 30.0


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def group_from_rows(group_uuid: str, rows: list) -> dict | None:
    """The engine's reply shape (group > tasks > subTasks) from the joined rows, or None."""
    if not rows:
        return None
    tasks: dict[Any, dict] = {}
    for r in rows:
        task = tasks.setdefault(r["task_no"], {"agentTaskKey": r["task_key"], "agentTaskName": r["task_key"],
                                               "agentTaskStatus": r["task_status"], "subTasks": []})
        task["subTasks"].append({"agentSubTaskKey": r["sub_key"], "agentSubTaskName": r["sub_key"],
                                 "agentSubTaskStatus": r["sub_status"],
                                 "agentSubTaskResponse": _json(r["sub_response"]),
                                 "failedReason": r["failed_reason"]})
    return {"status": "ok", "http_status": 200, "result": {
        "agentTaskGroupUuid": group_uuid, "agentTaskGroupStatus": rows[0]["group_status"],
        "tasks": [tasks[k] for k in sorted(tasks)]}}


def finished(group: dict | None) -> bool:
    if not group:
        return False
    subs = [s for t in group["result"]["tasks"] for s in t["subTasks"]]
    return bool(subs) and all(str(s["agentSubTaskStatus"]).lower() not in UNFINISHED for s in subs)


async def _poll_db(group_uuid: str, timeout: float, connect: Callable | None = None) -> dict | None:
    if connect is None:
        import asyncpg
        dsn = os.environ.get("TRON_POSTGRES_DSN", "")
        if not dsn:
            return None
        connect = lambda: asyncpg.connect(dsn=dsn, command_timeout=15)    # noqa: E731
    conn = await asyncio.wait_for(connect(), 15)
    try:
        deadline, group = time.monotonic() + timeout, None
        while True:
            group = group_from_rows(group_uuid, await conn.fetch(GROUP_SQL, group_uuid))
            if finished(group) or time.monotonic() >= deadline:
                return group
            await asyncio.sleep(POLL_SECONDS)
    finally:
        await conn.close()


def result_url(group_uuid: str) -> str:
    return os.environ.get("VIRA_RESULT_LOCATION", "").strip().replace("{uuid}", group_uuid)

def result_trigger_url(group_uuid: str) -> str:
    return os.environ.get("VIRA_result_trigger", "").strip().replace("{uuid}", group_uuid)

def api_problem(group_uuid: str = "00000000-0000-0000-0000-000000000000") -> str | None:
    """Why the info call must not be made, or None.  The xrtoken goes only to the engine's host."""
    import recruiter_cli as vira
    url = result_url(group_uuid)
    if "{uuid}" not in os.environ.get("VIRA_RESULT_LOCATION", ""):
        return "VIRA_RESULT_LOCATION must be the info URL with {uuid} where the group's uuid goes"
    problem = vira._url_problem(url, "VIRA_RESULT_LOCATION")
    if problem:
        return problem
    if urlsplit(url).hostname != urlsplit(vira._task_group_url()).hostname:
        return "VIRA_RESULT_LOCATION must be on the engine's host (VIRA_ACTUAL_LOCATION)"
    return None if vira.VIRA_XRTOKEN else "VIRA credentials not configured: VIRA_XRTOKEN"


def _poll_api(group_uuid: str, timeout: float, session_factory: Callable | None = None) -> dict | None:
    import recruiter_cli as vira
    problem = api_problem(group_uuid)
    if problem:
        raise ValueError(problem)
    if session_factory is None:
        import requests
        session_factory = requests.Session
    deadline, group = time.monotonic() + timeout, None
    cutqueque_post = os.environ.get("cutqueque", False)
    with session_factory() as session:
        session.trust_env = False                 # no proxies or .netrc, as for every VIRA call
        session.verify = vira.VIRA_CA_BUNDLE or True
        while True:
            resp = session.get(result_url(group_uuid), headers={"xrtoken": vira.VIRA_XRTOKEN},
                               timeout=20, allow_redirects=False)
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError:
                    body = None
                if isinstance(body, dict) and isinstance(body.get("tasks"), list):
                    group = {"status": "ok", "http_status": 200, "result": body}
            if time.monotonic() >= deadline-10:
                if cutqueque_post:
                    session.post(result_trigger_url(group_uuid), headers={"xrtoken": vira.VIRA_XRTOKEN},
                                timeout=20, allow_redirects=False)                
                    cutqueque_post = False
    
            if finished(group) or time.monotonic() >= deadline:
                return group            
            time.sleep(POLL_SECONDS)


def _run(coro):
    """Run a coroutine to completion from sync code, on a thread of its own when this thread
    already runs an event loop (asyncpg needs a loop it owns)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    out: dict = {}

    def work():
        try:
            out["value"] = asyncio.run(coro)
        except BaseException as exc:      # handed back to the caller's thread
            out["error"] = exc

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    if "error" in out:
        raise out["error"]
    return out.get("value")


def wait_for(group_uuid: str, timeout: float | None = None, via: str | None = None) -> dict | None:
    """The group in the engine's reply shape, as far as it got within the wait, or None when no
    source is set up.  Unmasked: the caller masks it like any VIRA reply."""
    via = via or source()
    timeout = wait_seconds() if timeout is None else timeout
    if via == "api":
        return _poll_api(group_uuid, timeout)
    if via == "db":
        return _run(_poll_db(group_uuid, timeout))
    return None
