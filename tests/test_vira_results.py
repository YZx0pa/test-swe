"""vira_results: waiting for a VIRA engine task group's outcome, and jeni_tools reading it back."""
import asyncio
import json

import pytest

import jeni_tools
import pii_vault
import recruiter_cli
import vira_results
import vira_tools
from conftest import read_audit

vira_tools.configure("mock")
UUID = "6f1c2a52-0000-4000-8000-0000000000aa"
RECEIVED = {"status": "ok", "http_status": 200, "result": {
    "agentTaskGroupUuid": UUID, "message": "Your tasks have been received. We are processing your tasks."}}


def rows(*subs, group_status="completed"):
    """Joined rows as the engine's tables give them: (key, status, response, failed_reason)."""
    return [{"group_status": group_status, "task_no": 1, "task_key": "task_search_users",
             "task_status": "completed", "sub_no": n, "sub_key": key, "sub_status": status,
             "sub_response": response, "failed_reason": reason}
            for n, (key, status, response, reason) in enumerate(subs, 1)]


class FakeConn:
    """An asyncpg connection that answers GROUP_SQL with a list of row sets, one per poll."""

    def __init__(self, *polls):
        self.polls, self.asked, self.closed = list(polls), [], False

    async def fetch(self, sql, *params):
        self.asked.append((sql, params))
        return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]

    async def close(self):
        self.closed = True


def poll(conn, timeout):
    async def connect():
        return conn
    return asyncio.run(vira_results._poll_db(UUID, timeout, connect))


# --- the reader -----------------------------------------------------------------
def test_rows_become_the_engines_reply_shape():
    group = vira_results.group_from_rows(UUID, rows(
        ("sub_task_search_users", "completed", '{"users": [{"userId": 802}]}', None)))
    [task] = group["result"]["tasks"]
    [sub] = task["subTasks"]
    assert group["result"]["agentTaskGroupUuid"] == UUID and task["agentTaskKey"] == "task_search_users"
    assert sub["agentSubTaskResponse"] == {"users": [{"userId": 802}]}        # jsonb text parsed
    assert vira_results.group_from_rows(UUID, []) is None


def test_it_polls_until_nothing_is_queued_and_reads_only_this_group():
    queued = rows(("sub_task_search_users", "queued", None, None), group_status="queued")
    done = rows(("sub_task_search_users", "completed", {"users": []}, None))
    conn = FakeConn(queued, done)
    vira_results.POLL_SECONDS, saved = 0.01, vira_results.POLL_SECONDS
    try:
        group = poll(conn, timeout=5)
    finally:
        vira_results.POLL_SECONDS = saved
    assert vira_results.finished(group) and len(conn.asked) == 2 and conn.closed
    sql, params = conn.asked[0]
    assert params == (UUID,) and "WHERE g.agent_task_group_uuid = $1::uuid" in sql
    assert sql.lstrip().upper().startswith("SELECT")


def test_a_group_still_queued_at_the_deadline_comes_back_unfinished():
    conn = FakeConn(rows(("sub_task_search_users", "queued", None, None), group_status="queued"))
    group = poll(conn, timeout=0)
    assert not vira_results.finished(group) and len(conn.asked) == 1


def test_nothing_is_read_unless_a_source_is_set(monkeypatch):
    monkeypatch.delenv("VIRA_RESULT_SOURCE", raising=False)
    monkeypatch.setattr(vira_results, "_poll_db", lambda *a, **k: pytest.fail("read without a source"))
    assert vira_results.wait_for(UUID) is None


# --- jeni_tools: the read-back ---------------------------------------------------
@pytest.fixture
def engine(monkeypatch):
    """The engine as it behaves: it acknowledges the group, and the result comes later."""
    monkeypatch.setattr(recruiter_cli, "_call", lambda path, query, body, mode: RECEIVED)


def test_the_outcome_is_read_back_masked_and_audited(audit_log, engine, monkeypatch):
    monkeypatch.setattr(vira_results, "wait_for", lambda uuid: vira_results.group_from_rows(uuid, rows(
        ("sub_task_search_users", "completed", {"users": [{"userId": 803, "email": "priya.nair@example.com"}]}, None))))
    result = jeni_tools.run("search_users", {"search_key": "Priya"})
    assert result["status"] == "ok" and result["task_status"] == "completed"
    [user] = result["result"]["users"]
    assert user["userId"] == 803 and pii_vault.VAULT.sources(user["email"]) >= {"colleague"}
    assert [a["command"] for a in read_audit(audit_log)] == ["search-users", "read-result"]
    assert "priya.nair@example.com" not in json.dumps(read_audit(audit_log))


def test_a_failed_sub_task_says_why(audit_log, engine, monkeypatch):
    monkeypatch.setattr(vira_results, "wait_for", lambda uuid: vira_results.group_from_rows(uuid, rows(
        ("sub_task_get_single_job_details", "failed", {"error": "x"}, "Job not found"))))
    result = jeni_tools.run("get_single_job_details", {"job_id": 7001})
    assert (result["status"], result["failed_reason"]) == ("error", "Job not found")


@pytest.mark.parametrize("outcome", ["no source", "still queued", "database down"])
def test_without_an_outcome_the_task_is_reported_queued(audit_log, engine, monkeypatch, outcome):
    def wait_for(uuid):
        if outcome == "database down":
            raise OSError("unreachable")
        if outcome == "still queued":
            return vira_results.group_from_rows(uuid, rows(("sub_task_search_users", "queued", None, None),
                                                           group_status="queued"))
        return None
    monkeypatch.setattr(vira_results, "wait_for", wait_for)
    assert jeni_tools.run("search_users", {"search_key": "Bob"}) == jeni_tools.QUEUED
    if outcome == "database down":
        assert read_audit(audit_log)[-1] == {**read_audit(audit_log)[-1], "status": "exception", "error": "OSError"}


# --- vira_check: the real-engine check -------------------------------------------
def group(*tasks):
    """A finished-or-not group from [(task key, [(sub key, status)])]."""
    out = []
    for n, (task, subs) in enumerate(tasks, 1):
        out += [{"group_status": "completed", "task_no": n, "task_key": task, "task_status": "completed",
                 "sub_no": m, "sub_key": key, "sub_status": status, "sub_response": None,
                 "failed_reason": "Missing mandatory field(s): job_title" if status == "failed" else None}
                for m, (key, status) in enumerate(subs, 1)]
    return vira_results.group_from_rows(UUID, out)


def test_the_failure_check_reads_chaining_and_independence():
    import vira_check
    chained = group(("task_setup_job", [("sub_task_get_job_description", "failed"),
                                        ("sub_task_get_job_skills", "failed")]),
                    ("task_search_users", [("sub_task_search_users", "completed")]))
    assert vira_check.verdicts(chained) == [
        "sub-tasks are chained: after a failed sub-task, the rest of its task failed",
        "tasks are independent: the next task still ran"]
    loose = group(("task_setup_job", [("sub_task_get_job_description", "failed"),
                                      ("sub_task_get_job_skills", "completed")]),
                  ("task_search_users", [("sub_task_search_users", "failed")]))
    assert vira_check.verdicts(loose)[0].startswith("sub-tasks are NOT chained")
    assert vira_check.verdicts(loose)[1].startswith("the next task didn't complete")
    queued = group(("task_setup_job", [("sub_task_get_job_description", "queued")]),
                   ("task_search_users", [("sub_task_search_users", "queued")]))
    assert vira_check.verdicts(queued) == ["not run yet: no verdict"]


def test_the_checks_only_read_and_name_their_groups():
    import vira_check
    sent = vira_check.checks("20261002-120000")
    assert [label for label, _ in sent] == ["runs", "engine", "failure"]
    subs = {s["sub_task_name"] for _, body in sent for t in body["tasks"] for s in t["sub_tasks"]}
    assert subs == {"sub_task_search_users", "sub_task_get_job_description", "sub_task_get_job_skills"}
    assert all(body["agent_session_uuid"] and body["task_group_name"].startswith("Jeni v2 check ")
               for _, body in sent)


def test_the_check_needs_its_settings_before_it_sends_anything(monkeypatch, capsys):
    import vira_check
    monkeypatch.delenv("VIRA_ACTUAL_LOCATION", raising=False)
    monkeypatch.setattr(recruiter_cli, "execute", lambda *a, **k: pytest.fail("sent without settings"))
    assert vira_check.main([]) == 2 and "VIRA_ACTUAL_LOCATION" in capsys.readouterr().out
