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
    unknown = vira_results.group_from_rows(UUID, [
        {"group_status": "completed", "task_no": t, "task_key": task, "task_status": "completed",
         "sub_no": n, "sub_key": key, "sub_status": status, "sub_response": None, "failed_reason": reason}
        for t, n, task, key, status, reason in (
            (1, 1, "task_setup_job", "sub_task_get_job_description", "failed",
             "agentSubTaskKey(sub_task_get_job_description) is not registered"),
            (1, 2, "task_setup_job", "sub_task_get_job_skills", "failed",
             "agentSubTaskKey(sub_task_get_job_skills) is not registered"),
            (2, 1, "task_search_users", "sub_task_search_users", "completed", None))])
    assert vira_check.verdicts(unknown)[0].startswith("no verdict on chaining: the engine doesn't know")


def test_the_checks_only_read_and_name_their_groups():
    import vira_check
    sent = vira_check.checks("20261002-120000")
    assert [label for label, _ in sent] == ["runs", "engine", "failure"]
    subs = {s["sub_task_name"] for _, body in sent for t in body["tasks"] for s in t["sub_tasks"]}
    assert subs == {"sub_task_search_users", "sub_task_get_job_description", "sub_task_get_single_job_details"}
    [failing, _] = dict(sent)["failure"]["tasks"]
    no_job = failing["sub_tasks"][0]                          # job details without a job id: must fail
    assert no_job["sub_task_name"] == "sub_task_get_single_job_details"
    assert [f.get("field_value") for f in no_job["fields"] if f["field_name"] == "job_id"] == [None]
    assert all(body["agent_session_uuid"] and body["task_group_name"].startswith("Jeni v2 check ")
               for _, body in sent)
    assert len({body["agent_session_uuid"] for _, body in sent}) == 3       # one session per group


def test_the_check_needs_its_settings_before_it_sends_anything(monkeypatch, capsys):
    import vira_check
    monkeypatch.delenv("VIRA_ACTUAL_LOCATION", raising=False)
    monkeypatch.setattr(recruiter_cli, "execute", lambda *a, **k: pytest.fail("sent without settings"))
    assert vira_check.main([]) == 2 and "VIRA_ACTUAL_LOCATION" in capsys.readouterr().out


# --- the engine's info call (VIRA_RESULT_SOURCE=api) ------------------------------
class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def info_body(status):
    return {"agentTaskGroupUuid": UUID, "agentTaskGroupStatus": status, "creatorName": "A Person",
            "tasks": [{"agentTaskKey": "task_search_users", "agentTaskStatus": status,
                       "subTasks": [{"agentSubTaskKey": "sub_task_search_users", "agentSubTaskStatus": status,
                                     "agentSubTaskResponse": {"users": []}, "failedReason": None}]}]}


@pytest.fixture
def info_api(monkeypatch):
    monkeypatch.setattr(recruiter_cli, "VIRA_ACTUAL_LOCATION", "https://engine.example.test/agent/task-group")
    monkeypatch.setattr(recruiter_cli, "VIRA_XRTOKEN", "xr-test")
    monkeypatch.setenv("VIRA_RESULT_LOCATION", "https://engine.example.test/agent/info/{uuid}")
    monkeypatch.setattr(vira_results, "POLL_SECONDS", 0.01)

    class Session:
        made, replies = [], []

        def __init__(self):
            self.trust_env, self.gets = True, []
            Session.made.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, **kwargs):
            self.gets.append((url, kwargs))
            return Session.replies.pop(0) if len(Session.replies) > 1 else Session.replies[0]
    return Session


def test_the_info_call_is_polled_until_the_group_has_run(info_api):
    info_api.replies = [FakeResponse(404, None), FakeResponse(200, info_body("queued")),
                        FakeResponse(200, info_body("completed"))]
    group = vira_results._poll_api(UUID, 5, info_api)
    assert vira_results.finished(group) and group["result"]["agentTaskGroupStatus"] == "completed"
    [session] = info_api.made
    url, kwargs = session.gets[0]
    assert url == f"https://engine.example.test/agent/info/{UUID}" and len(session.gets) == 3
    assert kwargs["headers"] == {"xrtoken": "xr-test"} and kwargs["allow_redirects"] is False
    assert session.trust_env is False


@pytest.mark.parametrize("location,problem", [
    ("https://elsewhere.example.test/info/{uuid}", "on the engine's host"),
    ("http://engine.example.test/agent/info/{uuid}", "must be https"),
    ("https://engine.example.test/agent/info/", "with {uuid}"),
])
def test_the_token_goes_only_to_the_engines_host_over_https(info_api, monkeypatch, location, problem):
    monkeypatch.setenv("VIRA_RESULT_LOCATION", location)
    assert problem in vira_results.api_problem()
    with pytest.raises(ValueError, match="VIRA_RESULT_LOCATION"):
        vira_results._poll_api(UUID, 0, info_api)
    assert info_api.made == []


def test_an_info_reply_is_masked_and_projected_like_any_other(audit_log, engine, info_api, monkeypatch):
    monkeypatch.setenv("VIRA_RESULT_SOURCE", "api")
    body = info_body("completed")
    body["tasks"][0]["subTasks"][0]["agentSubTaskResponse"] = {"users": [{"userId": 801, "email": "alice.johnson@example.com"}]}
    info_api.replies = [FakeResponse(200, body)]
    import requests
    monkeypatch.setattr(requests, "Session", info_api)
    result = jeni_tools.run("search_users", {"search_key": "Alice"})
    assert result["status"] == "ok" and "@" not in json.dumps(result)
    assert pii_vault.VAULT.sources(result["result"]["users"][0]["email"]) >= {"colleague"}


# --- vira_check --writes: the write test -------------------------------------------
@pytest.fixture
def check_settings(monkeypatch):
    """vira_check's settings present (reading back from the tables), with nothing really sent:
    every group is acknowledged, and `sent` lists the commands that went out."""
    import vira_check
    for name, value in (("VIRA_ACTUAL_LOCATION", "https://engine.example.test/agent/task-group"),
                        ("VIRA_XRTOKEN", "xr-test"), ("TRON_POSTGRES_DSN", "postgresql://test")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VIRA_RESULT_SOURCE", "db")
    monkeypatch.delenv("VIRA_RESULT_LOCATION", raising=False)
    sent = []
    monkeypatch.setattr(vira_check.vira, "execute", lambda cmd, *a, **k: sent.append(cmd) or RECEIVED)
    return sent


def test_no_write_is_sent_while_the_engine_leaves_reads_queued(check_settings, monkeypatch, capsys):
    import vira_check
    monkeypatch.setattr(vira_results, "wait_for", lambda uuid, *a, **k: group(
        ("task_search_users", [("sub_task_search_users", "queued")])))
    monkeypatch.setattr(vira_check, "run_writes", lambda *a, **k: pytest.fail("wrote while reads queue"))
    assert vira_check.main(["--writes", "--wait", "0"]) == 1
    assert check_settings == ["check-runs", "check-engine", "check-failure"]
    assert "Write test not sent" in capsys.readouterr().out


def test_the_write_test_follows_only_once_the_reads_have_run(check_settings, monkeypatch):
    import vira_check
    ran = group(("task_setup_job", [("sub_task_get_job_description", "failed"),
                                    ("sub_task_get_job_skills", "failed")]),
                ("task_search_users", [("sub_task_search_users", "completed")]))
    monkeypatch.setattr(vira_results, "wait_for", lambda uuid, *a, **k: ran)
    writes = []
    monkeypatch.setattr(vira_check, "run_writes", lambda stamp, **k: writes.append(stamp) or True)
    assert vira_check.main(["--wait", "0"]) == 0 and writes == []          # not without --writes
    assert vira_check.main(["--writes", "--wait", "0"]) == 0 and len(writes) == 1


def test_only_the_planned_cards_are_approved():
    import vira_check
    title, created, cards = "Jeni v2 test 1", vira_check.CreatedJobs(), []
    decide = vira_check.planned_decisions(title, created, cards)

    def card(name, **args):
        return decide({"action_requests": [{"name": name, "args": args}]})[0]["type"]
    assert card("create_job", job_title="Something else") == "reject"
    assert card("add_job_skills", job_id=7001, skills=["Kafka"]) == "reject"     # before any job exists
    assert card("create_job", job_title="jeni v2 test 1 ") == "approve"
    created.ids.append(7123)
    assert card("create_job", job_title=title) == "reject"                        # one job only
    assert card("add_job_skills", job_id=7001, skills=["Kafka"]) == "reject"     # not the new job
    assert card("add_job_skills", job_id=7123, skills=["Kafka", "Spark"]) == "reject"
    assert card("transfer_job_ownership", job_id=7123, new_owner_user_email="me") == "reject"
    assert card("add_job_skills", job_id=7123, skills=["kafka"]) == "approve"
    assert [ok for _, ok in cards] == [False, False, True, False, False, False, False, True]


def test_the_callback_keeps_the_id_create_job_returned():
    import uuid as uuid_lib
    from langchain_core.messages import ToolMessage
    import vira_check
    created = vira_check.CreatedJobs()
    for name, reply in (("search_users", {"request_status": "ok", "subtasks": [{"result": {"jobId": 1}}]}),
                        ("create_job", {"request_status": "failed", "message": "no"}),
                        ("create_job", {"request_status": "ok", "group_status": "completed",
                                        "subtasks": [{"status": "completed", "result": {"jobId": 7123}}]}),
                        ("create_job", {"request_status": "ok", "group_status": "completed",   # the real engine
                                        "subtasks": [{"status": "completed", "result": {"jobId": "686569"}}]})):
        run = uuid_lib.uuid4()
        created.on_tool_start({"name": name}, "{}", run_id=run)
        created.on_tool_end(ToolMessage(json.dumps(reply), tool_call_id="c"), run_id=run)
    assert created.ids == [7123, 686569]
    def details(reply):
        return {"request_status": "ok", "group_status": "completed",
                "subtasks": [{"status": "completed", "result": reply}]}
    assert vira_check.has_skill(details({"skills": ["Python", "Kafka"]}), "kafka")
    assert vira_check.has_skill(details({"skillsText": "Python, Kafka"}), "Kafka")
    assert not vira_check.has_skill(details({"skills": [{"name": "Python"}]}), "Kafka")
    assert vira_check.succeeded(details({})) and not vira_check.succeeded(jeni_tools.QUEUED)
    failed = {**details({}), "subtasks": [{"status": "failed", "failed_reason": "Job not found"}]}
    assert not vira_check.succeeded(failed) and "Job not found" in vira_check.outcome(failed)
    assert vira_check.is_closed(details({"isJobClosed": True, "jobStatus": "Both"}))     # the real engine
    assert vira_check.is_closed(details({"status": "closed"}))                           # the mock
    assert not vira_check.is_closed(details({"isJobClosed": False}))


def stand_in(name, description, *fields):
    """A catalog entry in the fixture's style; fields are (name, mandatory)."""
    return {"task_name": f"task_{name}", "description": description, "level": 1, "task_output": [],
            "sub_tasks": [{"sub_task_name": f"sub_task_{name}", "description": description, "fields": [
                {"field_name": f, "mandatory": m, "field_value": "{{to_be_filled}}"} for f, m in fields]}]}


@pytest.fixture
def kept_mock(monkeypatch, tmp_path):
    """The mock remembering changes, and a catalog with the three tasks the write test needs
    that the shared fixture leaves out."""
    import os
    import mock_jeni
    catalog = json.loads(open(os.environ["JENI_TASKS_FILE"], encoding="utf-8").read())
    catalog["tasks"] += [
        stand_in("remove_job_skills", "Remove skills from a job.", ("job_id", True), ("skills", True)),
        stand_in("edit_job", "Edit a job.", ("job_id", True), ("job_title", False), ("job_description", False)),
        stand_in("make_job_closed", "Close a job.", ("job_id", True), ("reason_for_closure", False))]
    (tmp_path / "jeni_tasks.json").write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setenv("JENI_TASKS_FILE", str(tmp_path / "jeni_tasks.json"))
    state = mock_jeni.remember_changes()
    state.reset()
    yield state
    mock_jeni.forget_changes()


def write_agent(title, job_for_skill=None):
    import mock_jeni
    from fakes import call, calls, say, scripted
    job = mock_jeni.created_job_id(title)
    return job, scripted(
        calls(call("create_job", {"job_title": title, "skills": ["Python"], "min_exp": 1, "max_exp": 2}, "c1")),
        calls(call("add_job_skills", {"job_id": job_for_skill or job, "skills": ["Kafka"]}, "c2")),
        say("Created the job and added Kafka."))


def test_the_write_test_runs_through_on_the_mock_and_closes_its_job(audit_log, kept_mock):
    import vira_check
    job, model = write_agent("Jeni v2 test 20261002-120000")
    lines = []
    assert vira_check.run_writes("20261002-120000", lines.append, model=model, mode="mock")
    assert lines[0] == "agent: cards [('create_job', True), ('add_job_skills', True)]"
    assert any("steps chain" in line for line in lines)
    assert not any("NOT as expected" in line for line in lines)
    record = kept_mock.jobs[job]
    assert record["status"] == "closed" and "Kafka" not in record["skills"] and record["jobDescription"]
    assert [a["command"] for a in read_audit(audit_log)] == [
        "create-job", "add-job-skills", "make-job-private", "get-single-job-details", "remove-job-skills",
        "edit-job", "get-single-job-details", "make-job-closed", "get-single-job-details"]
    assert lines[-1] == f"job {job} reads closed: True"


def test_a_skill_aimed_at_another_job_is_refused_and_the_test_fails(audit_log, kept_mock):
    import mock_jeni
    import vira_check
    job, model = write_agent("Jeni v2 test 20261002-130000", job_for_skill=7001)
    lines = []
    assert not vira_check.run_writes("20261002-130000", lines.append, model=model, mode="mock")
    assert "Kafka" not in kept_mock.jobs[7001]["skills"] and kept_mock.jobs[job]["status"] == "closed"
    assert "add-job-skills" not in [a["command"] for a in read_audit(audit_log)]
    assert mock_jeni.JOBS[7001]["skills"] == ["Python", "Go", "PostgreSQL"]
