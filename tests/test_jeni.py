"""jeni_tools: Jeni's task catalog as typed tools, on the mock task-group VIRA.

The catalog here is tests/fixtures/jeni_tasks.json (conftest sets JENI_TASKS_FILE): 15 tasks in
v1's format with v1's field names, but not Jeni's real catalog, which stays out of git.
"""
import argparse
import asyncio
import json
import uuid

import pytest

import agent_kit
import db_queries
import grounding
import jeni_eval
import jeni_tools
import mock_jeni
import pii_vault
import recruiter_cli
import run_deepagent
import run_langgraph
import vira_tools
from conftest import done, read_audit, reply
from fakes import call, calls, say, scripted
from mock_vira import MockVira

vira_tools.configure("mock")

# One valid call per task in the test catalog: every one must complete against the mock.
VALID = {
    "create_job": {"job_title": "Data Engineer", "skills": ["Python"], "min_exp": 3, "max_exp": 5},
    "add_job_skills": {"job_id": 7001, "skills": ["Kafka"]},
    "clone_job": {"job_id": 7001},
    "transfer_job_ownership": {"job_id": 7001, "new_owner_user_email": "bob.tan@example.com"},
    "add_job_collaborators": {"job_id": 7001, "user_ids": [802], "role_id": 5},
    "publish_job_to_linkedin": {"job_id": 7003},
    "make_job_private": {"job_id": 7003},
    "make_job_public": {"job_id": 7001},
    "shortlist_multiple_application": {"app_ids": [5102, 5103]},
    "share_application": {"app_ids": [5102], "emails": ["hm@example.com"], "message": "Please review"},
    "get_applications": {"job_id": 7001},
    "get_single_application_details": {"app_id": 5102},
    "create_application_to_job": {"job_id": 7001, "candidate_name": "Maya Lim",
                                  "candidate_email": "maya.lim@example.com"},
    "get_single_job_details": {"job_id": 7001},
    "search_users": {"search_key": "Bob"},
}
# The keys of a real VIRA task-group reply (raw_data/response.js), level by level.
GROUP_KEYS = {"agentTaskGroupUuid", "agentTaskGroupId", "agentTaskGroupKey", "agentTaskGroupName",
              "agentTaskGroupStatus", "agentSessionUuid", "agentTaskGroupDescription",
              "agentTaskGroupNo", "createdAt", "updatedAt", "creatorName", "tasks"}
TASK_KEYS = {"agentTaskId", "agentTaskUuid", "agentTaskGroupId", "agentTaskKey", "agentTaskName",
             "agentTaskDescription", "agentTaskNo", "agentTaskStatus", "createdAt", "updatedAt",
             "subTasks"}
SUB_TASK_KEYS = {"agentSubTaskId", "agentSubTaskUuid", "agentTaskId", "agentSubTaskKey",
                 "agentSubTaskName", "agentSubTaskDescription", "agentSubTaskNo",
                 "agentSubTaskStatus", "createdAt", "updatedAt", "agentSubTaskResponse",
                 "failedReason"}


def approve(request):
    """A reviewer who approves every call (shortlist, share and transfer pause in any mode)."""
    return [{"type": "approve"} for _ in request["action_requests"]]


def run(model, task, **build):
    return agent_kit.run_task(run_langgraph.build_agent(model=model, toolset=agent_kit.toolset("jeni"),
                                                        **build), task, decide=approve)


def tool_messages(result):
    return {m.tool_call_id: m for m in result["messages"] if m.type == "tool"}


def fields_of(entry: dict) -> dict:
    [task] = entry["body"]["tasks"]
    [sub] = task["sub_tasks"]
    return {f["field_name"]: f.get("field_value") for f in sub["fields"]}


# --- the catalog -------------------------------------------------------------
def test_the_catalog_is_read_from_the_configured_file():
    assert jeni_tools.catalog_path().parts[-2:] == ("fixtures", "jeni_tasks.json")
    tasks = jeni_tools.load_catalog()
    assert len(tasks) == 15 and all(len(t["sub_tasks"]) == 1 for t in tasks)
    assert jeni_tools.names() == set(VALID)
    fields = {f["field_name"] for t in tasks for f in t["sub_tasks"][0]["fields"]}
    assert jeni_tools.USER_ONLY <= fields


def test_v2s_task_lists_name_real_v1_tasks():
    # READ_ONLY, FIXED_VALUES and NO_DEFAULT hold tool names; a typo would silently not apply.
    known = {name.removeprefix("task_") for name in mock_jeni.HANDLERS}
    assert len(known) == 22 and len(jeni_tools.READ_ONLY) == 6
    assert jeni_tools.READ_ONLY | set(jeni_tools.FIXED_VALUES) | {t for t, _ in jeni_tools.NO_DEFAULT} <= known
    assert {t["task_name"] for t in jeni_tools.load_catalog()} <= set(mock_jeni.HANDLERS)


def test_a_missing_catalog_is_a_clear_error_and_only_for_jeni(tmp_path, monkeypatch):
    monkeypatch.setenv("JENI_TASKS_FILE", str(tmp_path / "missing.json"))
    agent_kit._jeni.cache_clear()
    with pytest.raises(jeni_tools.CatalogMissing, match="JENI_TASKS_FILE"):
        jeni_tools.names()
    with pytest.raises(SystemExit, match="shared internally"):
        agent_kit.cli_toolset(argparse.Namespace(tools="jeni"))
    assert agent_kit.cli_toolset(argparse.Namespace(tools="vira")) is agent_kit.VIRA
    monkeypatch.setenv("JENI_TASKS_FILE", "tests/fixtures/jeni_tasks.json")    # relative: to the repo
    assert jeni_tools.catalog_path() == jeni_tools.HERE / "tests" / "fixtures" / "jeni_tasks.json"
    agent_kit._jeni.cache_clear()


def test_catalog_from_js_reads_v1s_file_format():
    js = """const XAGENT_SUBTASK_API_CONFIG_SCHEMA = {
  "tasks": [
    {
      "task_name": "task_a",
      // "task_name": "task_old",
      "fields": [1, 2,],
    },
  ]
}

XAGENT_SUBTASK_API_CONFIG_SCHEMA.tasks.forEach(() => {});
"""
    assert jeni_tools.catalog_from_js(js) == {"tasks": [{"task_name": "task_a", "fields": [1, 2]}]}


def test_every_tool_is_typed_and_described():
    tools = {t.name: t for t in jeni_tools.langchain_tools()}
    assert set(tools) == jeni_tools.names()
    for name, tool in tools.items():
        schema = tool.tool_call_schema.model_json_schema()
        task = jeni_tools.catalog()[name]
        mandatory = {f["field_name"] for f in task["sub_tasks"][0]["fields"] if f["mandatory"]}
        assert set(schema.get("required", [])) == mandatory - set(jeni_tools.FIXED_VALUES.get(name, {})), name
        assert all(p.get("description") for p in schema["properties"].values()), name
        assert ("(read-only)" in tool.description) == (name in jeni_tools.READ_ONLY), name


def test_role_has_no_default_and_visibility_is_set_by_the_task(audit_log):
    role = jeni_tools.models()["add_job_collaborators"].model_json_schema()["properties"]["role_id"]
    assert role["enum"] == [1, 5] and "default" not in role        # v1 pre-fills 1 (administrator)
    for name, private in (("make_job_private", True), ("make_job_public", False)):
        assert "is_private" not in jeni_tools.models()[name].model_fields
        jeni_tools.run(name, {"job_id": 7001})
        assert fields_of(read_audit(audit_log)[-1])["is_private"] is private


@pytest.mark.parametrize("name,args", [
    ("share_application", {"app_ids": [5102], "emails": ["not-an-email"], "message": "hi"}),
    ("share_application", {"app_ids": [5102], "emails": ["<redacted-email>"], "message": "hi"}),
    ("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 3}),
    ("add_job_collaborators", {"job_id": 7001, "user_ids": [802]}),
    ("get_applications", {"job_id": 7001, "limit": 500}),
    ("get_single_job_details", {"job_id": 0}),
    ("get_single_job_details", {"job_id": 7001, "extra": 1}),
    ("add_job_skills", {"job_id": 7001, "skills": [f"s{i}" for i in range(31)]}),
    ("create_job", {"job_title": "x" * 201}),
])
def test_invalid_input_never_reaches_vira(audit_log, name, args):
    result = jeni_tools.run(name, args)
    assert result["request_status"] == "failed" and result["message"].startswith("invalid arguments")
    assert read_audit(audit_log) == []


# --- the call and the reply --------------------------------------------------
def test_a_call_sends_one_task_group_in_v1s_format(audit_log, monkeypatch):
    paths = []
    monkeypatch.setattr(recruiter_cli, "_call",
                        lambda path, query, body, mode: paths.append(path) or MockVira.call(path, query, body))
    jeni_tools.run("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5})
    [entry] = read_audit(audit_log)
    assert paths == [jeni_tools.PATH] and entry["command"] == "add-job-collaborators"
    assert entry["body"]["task_group_name"] == "Jeni v2 - add_job_collaborators"
    [task] = entry["body"]["tasks"]
    assert (task["task_name"], task["level"]) == ("task_add_job_collaborators", 1)
    assert task["sub_tasks"][0]["sub_task_name"] == "sub_task_add_job_collaborators"
    assert task["sub_tasks"][0]["fields"] == [
        {"field_name": "job_id", "mandatory": True, "field_value": 7001},
        {"field_name": "user_ids", "mandatory": True, "field_value": [802]},
        {"field_name": "role_id", "mandatory": True, "field_value": 5}]
    assert uuid.UUID(entry["body"]["agent_session_uuid"])                       # VIRA requires one


def test_every_task_group_is_its_own_vira_session(audit_log):
    """VIRA scopes a task group's uniqueness by agent_session_uuid, so even the same thread
    sends each group in a session of its own (jeni_tools.session_uuid)."""
    def agent():
        return run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), model=scripted(
            calls(call("get_single_job_details", {"job_id": 7001}, "c1")), say("Done.")))
    for thread in ("t1", "t1", "t2"):
        agent_kit.run_task(agent(), "Show job 7001.", thread_id=thread)
    sessions = [a["body"]["agent_session_uuid"] for a in read_audit(audit_log)]
    assert len(sessions) == len(set(sessions)) == 3 and all(uuid.UUID(s) for s in sessions)
    assert jeni_tools.session_uuid("t1") != jeni_tools.session_uuid("t1")


def test_a_queued_task_is_reported_as_queued_not_done():
    received = {"status": "ok", "http_status": 200, "result": {
        "agentTaskGroupUuid": "6f1c2a52-0000-4000-8000-000000000001",
        "message": "Your tasks have been received. We are processing your tasks."}}
    assert jeni_tools.project(received) == {
        "request_status": "ok", "group_status": "queued", "subtasks": [],
        "message": "VIRA accepted the task and runs it in the background; its outcome isn't known yet."}
    assert "queued" in jeni_tools.RULES


def test_the_mock_answers_every_task_in_the_real_reply_shape():
    for name, args in VALID.items():
        clean = jeni_tools.models()[name].model_validate(args).model_dump(exclude_none=True)
        reply = MockVira.call(jeni_tools.PATH, {}, jeni_tools.payload(jeni_tools.catalog()[name], clean))
        group = reply["result"]
        assert set(group) == GROUP_KEYS | {"_note"}, name
        [task] = group["tasks"]
        [sub] = task["subTasks"]
        assert set(task) == TASK_KEYS and set(sub) == SUB_TASK_KEYS, name
        assert (group["agentTaskGroupStatus"], sub["agentSubTaskStatus"]) == ("completed", "completed"), name
        assert sub["agentSubTaskKey"] == f"sub_{jeni_tools.catalog()[name]['task_name']}"


def test_the_model_sees_only_the_sub_task_result():
    for name, args in VALID.items():
        result = jeni_tools.run(name, args)
        assert set(result) == {"request_status", "group_status", "subtasks"}, name
        assert done(result) and result["group_status"] == "completed", name
        text = json.dumps(result)
        assert "creatorName" not in text and "agentTaskGroupUuid" not in text, name


def test_a_failed_task_says_why():
    assert jeni_tools.run("get_single_job_details", {"job_id": 4242}) == {
        "request_status": "ok", "group_status": "failed",
        "subtasks": [{"status": "failed", "failed_reason": "Job 4242 not found"}]}
    assert jeni_tools.run("publish_job_to_linkedin", {"job_id": 7001})["subtasks"][0]["failed_reason"] == (
        "Job must be open and public before publishing to LinkedIn")
    payload = jeni_tools.payload(jeni_tools.catalog()["clone_job"], {})
    sub = MockVira.call(jeni_tools.PATH, {}, payload)["result"]["tasks"][0]["subTasks"][0]
    assert sub["failedReason"] == "Missing mandatory field(s): job_id"


def test_partial_success_is_visible():
    result = jeni_tools.run("add_job_collaborators",
                            {"job_id": 7001, "user_ids": [802, 999], "role_id": 5})
    assert done(result)
    assert [p["userId"] for p in reply(result)["passedArr"]] == [802]
    assert [f["userId"] for f in reply(result)["failedArr"]] == [999]


def test_people_in_results_are_masked_but_their_ids_are_not():
    apps = json.dumps(jeni_tools.run("get_applications", {"job_id": 7001}))
    assert "@example.com" not in apps and "Mock Candidate" not in apps and "5102" in apps
    users = reply(jeni_tools.run("search_users", {"search_key": "Bob"}))["users"]
    [user] = users
    assert (user["userId"], user["firstName"], user["lastName"]) == (802, "<redacted>", "<redacted>")
    assert pii_vault.TOKEN.fullmatch(user["email"]) and pii_vault.VAULT.sources(user["email"]) >= {"colleague"}


def test_candidate_details_are_masked_in_the_audit_log(audit_log):
    jeni_tools.run("create_application_to_job", VALID["create_application_to_job"])
    [entry] = read_audit(audit_log)
    fields = fields_of(entry)
    assert fields["candidate_name"] == fields["candidate_email"] == "<redacted>"
    assert fields["job_id"] == 7001 and "Maya" not in json.dumps(entry)


def test_creator_and_owner_names_are_masked():
    masked = recruiter_cli._mask_pii({"creatorName": "A B", "ownerName": "C", "jobName": "Data Analyst",
                                      "agentTaskGroupName": "Add collaborator"})
    assert masked == {"creatorName": "<redacted>", "ownerName": "<redacted>",
                      "jobName": "Data Analyst", "agentTaskGroupName": "Add collaborator"}


# --- the mock's memory (the demo) ------------------------------------------------
@pytest.fixture
def kept():
    """remember_changes() for one test, from the fixtures, and stateless again after it."""
    state = mock_jeni.remember_changes()
    state.reset()
    yield state
    mock_jeni.forget_changes()


def test_the_mock_forgets_changes_by_default():
    assert done(jeni_tools.run("make_job_public", {"job_id": 7001}))
    assert not done(jeni_tools.run("publish_job_to_linkedin", {"job_id": 7001}))
    jeni_tools.run("add_job_skills", {"job_id": 7001, "skills": ["Kafka"]})
    details = reply(jeni_tools.run("get_single_job_details", {"job_id": 7001}))
    assert details["skills"] == ["Python", "Go", "PostgreSQL"] and details["isPrivate"] is True
    assert mock_jeni.JOBS[7001]["skills"] == ["Python", "Go", "PostgreSQL"]


def test_a_kept_state_remembers_what_changed(kept):
    assert not done(jeni_tools.run("publish_job_to_linkedin", {"job_id": 7001}))
    jeni_tools.run("make_job_public", {"job_id": 7001})
    assert done(jeni_tools.run("publish_job_to_linkedin", {"job_id": 7001}))
    jeni_tools.run("add_job_skills", {"job_id": 7001, "skills": ["Kafka"]})
    jeni_tools.run("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5})
    details = reply(jeni_tools.run("get_single_job_details", {"job_id": 7001}))
    assert details["skills"][-1] == "Kafka" and details["isPrivate"] is False
    assert details["publishedToLinkedIn"] is True
    assert details["collaborators"] == [{"userId": 802, "roleId": 5}]
    again = jeni_tools.run("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5})
    assert reply(again)["passedArr"] == [] and len(reply(again)["existingArr"]) == 1
    jeni_tools.run("shortlist_multiple_application", {"app_ids": [5102, 5103]})
    stages = {a["appId"]: a["stage"]
              for a in reply(jeni_tools.run("get_applications", {"job_id": 7001}))["applications"]}
    assert stages == {5101: "applied", 5102: "shortlisted", 5103: "shortlisted", 5104: "applied"}
    assert mock_jeni.JOBS[7001]["isPrivate"] is True and mock_jeni.APPLICATIONS[5102][2] == "applied"


def test_a_created_job_and_candidate_are_found_by_the_db_lookups(kept):
    queries = db_queries.fake_db_queries(kept.db_fixtures(5143))
    context = {"auth_profile": {"company_id": 5143}}

    def lookup(name, inputs):
        return asyncio.run(queries[name].handler(inputs, context))

    assert lookup("find_job_by_title", {"title": "Data Engineer"})["status"] == "not_found"
    job_id = reply(jeni_tools.run("create_job", VALID["create_job"]))["jobId"]
    assert job_id == mock_jeni.created_job_id("Data Engineer")
    assert lookup("find_job_by_title", {"title": "Data Engineer"}) == {
        "status": "resolved", "resolved_fields": {"job_id": job_id}}
    app_id = reply(jeni_tools.run("create_application_to_job", {**VALID["create_application_to_job"],
                                                                "job_id": job_id}))["appId"]
    assert lookup("list_job_applications", {"job_id": job_id})["applications"] == [app_id]
    assert reply(jeni_tools.run("create_job", VALID["create_job"]))["jobId"] == job_id + 1
    kept.reset()
    assert lookup("find_job_by_title", {"title": "Data Engineer"})["status"] == "not_found"
    assert not done(jeni_tools.run("get_single_job_details", {"job_id": job_id}))


def test_the_activity_log_keeps_ids_not_names_or_emails(kept):
    jeni_tools.run("create_application_to_job", VALID["create_application_to_job"])
    jeni_tools.run("share_application", VALID["share_application"])
    jeni_tools.run("search_users", {"search_key": "Bob"})
    jeni_tools.run("publish_job_to_linkedin", {"job_id": 7001})
    create, share, search, publish = kept.snapshot()["activity"]
    assert create["task"] == "create_application_to_job" and create["fields"] == {"job_id": 7001}
    assert create["created"] == mock_jeni.created_app_id("maya.lim@example.com")
    assert share["fields"] == {"app_ids": [5102], "recipients": 1}
    assert search["fields"] == {}
    assert (publish["status"], publish["reason"]) == (
        "failed", "Job must be open and public before publishing to LinkedIn")
    text = json.dumps(kept.snapshot())
    assert "@" not in text and "Maya" not in text and "Mock Candidate" not in text


# --- the agent -----------------------------------------------------------------
def test_a_read_runs_again_after_a_write_but_not_twice_in_a_row(audit_log):
    result = run(scripted(
        calls(call("get_single_job_details", {"job_id": 7001}, "c1")),
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Kafka"]}, "c2")),
        calls(call("get_single_job_details", {"job_id": 7001}, "c3")),     # a write since: runs
        calls(call("get_single_job_details", {"job_id": 7001}, "c4")),     # nothing since: refused
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Kafka"]}, "c5")),   # a write: refused
        say("done")), "Show job 7001, add Kafka to it, then show it again.")
    replies = tool_messages(result)
    assert "Refused" not in replies["c3"].text
    assert "identical to an earlier call" in replies["c4"].text
    assert "identical to an earlier call" in replies["c5"].text
    assert [a["command"] for a in read_audit(audit_log)] == [
        "get-single-job-details", "add-job-skills", "get-single-job-details"]


def test_a_reviewers_edit_stands_until_the_user_writes_again(audit_log):
    agent = run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), gate_writes=True, model=scripted(
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Kafka", "Spark"]}, "c1")),
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Spark"]}, "c2")),   # around the edit
        say("Added Kafka, as edited."),
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Spark"]}, "c3")),   # now the user asked
        say("Added Spark too.")))
    cards = []

    def edit_the_first(request):
        cards.append([a["args"] for a in request["action_requests"]])
        if len(cards) > 1:
            return [{"type": "approve"}]
        return [{"type": "edit", "edited_action": {"name": "add_job_skills",
                                                   "args": {"job_id": 7001, "skills": ["Kafka"]}}}]

    first = agent_kit.run_task(agent, "Add Kafka and Spark to job 7001.", decide=edit_the_first,
                               thread_id="t1")
    assert "reviewer already edited or rejected" in tool_messages(first)["c2"].text
    agent_kit.run_task(agent, "Add Spark too.", decide=edit_the_first, thread_id="t1")
    assert cards == [[{"job_id": 7001, "skills": ["Kafka", "Spark"]}],
                     [{"job_id": 7001, "skills": ["Spark"]}]]                      # no card for c2
    assert [fields_of(a)["skills"] for a in read_audit(audit_log)] == [["Kafka"], ["Spark"]]


def test_an_always_confirm_tool_gets_a_new_card_even_after_an_edit(audit_log):
    """ALWAYS_CONFIRM: every proposed run of a high-impact tool goes back to the reviewer, so
    an edit or a rejection never turns into a refusal the reviewer doesn't see."""
    agent = run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), gate_writes=True, model=scripted(
        calls(call("shortlist_multiple_application", {"app_ids": [5102, 5103]}, "c1")),
        calls(call("shortlist_multiple_application", {"app_ids": [5103]}, "c2")),   # around the edit
        say("Shortlisted 5102, as edited.")))
    cards = []

    def edit_then_reject(request):
        cards.append([a["args"] for a in request["action_requests"]])
        if len(cards) > 1:
            return [{"type": "reject", "message": "Only 5102."}]
        return [{"type": "edit", "edited_action": {"name": "shortlist_multiple_application",
                                                   "args": {"app_ids": [5102]}}}]

    agent_kit.run_task(agent, "Shortlist applicants 5102 and 5103.", decide=edit_then_reject)
    assert cards == [[{"app_ids": [5102, 5103]}], [{"app_ids": [5103]}]]          # a card for c2
    assert [fields_of(a)["app_ids"] for a in read_audit(audit_log)] == [[5102]]


def test_a_rejection_stands_and_gets_no_second_card(audit_log):
    agent = run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), gate_writes=True, model=scripted(
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Kafka"]}, "c1")),
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Kafka", "Spark"]}, "c2")),
        say("I left the skills as they were.")))
    cards = []
    result = agent_kit.run_task(agent, "Add Kafka to job 7001.", decide=lambda r: cards.append(r) or [
        {"type": "reject", "message": "Not now."}])
    assert len(cards) == 1 and read_audit(audit_log) == []
    assert "reviewer already edited or rejected" in tool_messages(result)["c2"].text
    assert agent_kit.overruled(result["messages"]) == {"add_job_skills"}


def test_a_failed_write_runs_again_once_another_write_ran(audit_log, kept):
    result = run(scripted(
        calls(call("publish_job_to_linkedin", {"job_id": 7001}, "c1")),    # private: fails
        calls(call("publish_job_to_linkedin", {"job_id": 7001}, "c2")),    # nothing since: refused
        calls(call("make_job_public", {"job_id": 7001}, "c3")),
        calls(call("publish_job_to_linkedin", {"job_id": 7001}, "c4")),    # a write since: runs
        calls(call("make_job_public", {"job_id": 7001}, "c5")),            # it succeeded: refused
        say("Published.")), "Make job 7001 public if it has to be, and publish it to LinkedIn.")
    replies = tool_messages(result)
    assert "must be open and public" in replies["c1"].text
    assert "identical to an earlier call" in replies["c2"].text
    assert done(json.loads(replies["c4"].text))
    assert "identical to an earlier call" in replies["c5"].text
    assert [a["command"] for a in read_audit(audit_log)] == [
        "publish-job-to-linkedin", "make-job-public", "publish-job-to-linkedin"]


def test_a_failed_sub_task_counts_as_a_failure_for_the_retry_rule(audit_log, kept):
    """jeni_tools.project reports a failed task as a failed sub-task under request_status ok;
    the guard reads that as a failure, so fixing the cause lets the same call run again."""
    result = run(scripted(
        calls(call("publish_job_to_linkedin", {"job_id": 7001}, "c1")),    # private: fails
        calls(call("make_job_public", {"job_id": 7001}, "c2")),
        calls(call("publish_job_to_linkedin", {"job_id": 7001}, "c3")),    # a write since: runs
        say("Published.")), "Publish job 7001 to LinkedIn, making it public if needed.")
    assert agent_kit._failed(tool_messages(result)["c1"])
    assert done(json.loads(tool_messages(result)["c3"].text))
    assert [a["command"] for a in read_audit(audit_log)] == [
        "publish-job-to-linkedin", "make-job-public", "publish-job-to-linkedin"]


def test_a_user_found_by_search_is_added_by_their_id(audit_log):
    result = run(scripted(
        calls(call("search_users", {"search_key": "Bob"}, "c1")),
        calls(call("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5}, "c2")),
        say("Bob is now a team member on job 7001.")), "Assign job 7001 to Bob as a team member.")
    assert all(m.status != "error" for m in tool_messages(result).values())
    audit = read_audit(audit_log)
    assert [a["command"] for a in audit] == ["search-users", "add-job-collaborators"]
    assert fields_of(audit[1])["user_ids"] == [802]


def test_a_new_jobs_id_is_used_in_the_next_call(audit_log):
    new_id = mock_jeni.created_job_id("Data Engineer")
    result = run(scripted(
        calls(call("create_job", {"job_title": "Data Engineer"}, "c1")),
        calls(call("add_job_skills", {"job_id": new_id, "skills": ["Spark"]}, "c2")),
        say(f"Created job {new_id} and added Spark.")), "Create a Data Engineer job, then add Spark to it.")
    assert all(m.status != "error" for m in tool_messages(result).values())
    assert [fields_of(a).get("job_id") for a in read_audit(audit_log)] == [None, new_id]


def test_an_invented_user_id_is_refused(audit_log):
    result = run(scripted(
        calls(call("add_job_collaborators", {"job_id": 7001, "user_ids": [804], "role_id": 5}, "c1")),
        say("Stopped.")), "Add Dana to job 7001 as a team member.")
    refused = tool_messages(result)["c1"]
    assert refused.status == "error" and "804 isn't in the task or any earlier result" in refused.text
    assert read_audit(audit_log) == []


def test_a_user_id_is_not_an_application_id(audit_log):
    result = run(scripted(
        calls(call("search_users", {"search_key": "Bob"}, "c1")),
        calls(call("shortlist_multiple_application", {"app_ids": [802]}, "c2")),
        say("Stopped.")), "Shortlist Bob.")
    refused = tool_messages(result)["c2"]
    assert refused.status == "error" and "802 is a user_id" in refused.text
    assert [a["command"] for a in read_audit(audit_log)] == ["search-users"]


def test_an_email_the_user_never_gave_is_refused(audit_log):
    result = run(scripted(
        calls(call("share_application", {"app_ids": [5102], "emails": ["hiring.manager@example.com"],
                                         "message": "Please review"}, "c1")),
        say("Which email should I share it with?")),
        "Share the CV of applicant 5102 with the hiring manager.")
    refused = tool_messages(result)["c1"]
    assert refused.status == "error" and "emails must be exactly what the user wrote" in refused.text
    assert "hiring.manager" not in refused.text                     # the value isn't echoed
    assert read_audit(audit_log) == []


def test_an_email_the_user_gave_is_used(audit_log):
    run(scripted(
        calls(call("share_application", {"app_ids": [5102], "emails": ["HM@example.com"],
                                         "message": "Please review"}, "c1")),
        say("Shared.")), "Share applicant 5102's CV with hm@example.com.")
    [entry] = read_audit(audit_log)
    assert entry["command"] == "share-application" and fields_of(entry)["emails"] == "<redacted>"


def test_the_user_answers_the_agents_question_in_the_next_turn(audit_log):
    # The agent asks for the email instead of guessing; the reply continues the same thread.
    agent = run_langgraph.build_agent(model=scripted(
        say("Which email should I share it with?"),
        calls(call("share_application", {"app_ids": [5102], "emails": ["hm@example.com"],
                                         "message": "Please review"}, "c1")),
        say("Shared.")), toolset=agent_kit.toolset("jeni"))
    agent_kit.run_task(agent, "Share the CV of applicant 5102 with the hiring manager.",
                       thread_id="t1")
    result = agent_kit.run_task(agent, "hm@example.com", thread_id="t1", decide=approve)
    assert tool_messages(result)["c1"].status != "error"
    [entry] = read_audit(audit_log)
    assert entry["command"] == "share-application"


def test_real_mode_gates_every_task_that_changes_data(monkeypatch):
    jeni = agent_kit.toolset("jeni")
    writes = jeni_tools.names() - jeni_tools.READ_ONLY
    assert len(writes) == 11                        # of the test catalog's 15 (Jeni's real one: 16 of 22)
    assert set(agent_kit.interrupt_on(False, "mock", toolset=jeni)) == {
        "shortlist_multiple_application", "share_application", "transfer_job_ownership"}
    assert set(agent_kit.interrupt_on(False, "real", toolset=jeni)) == writes
    assert set(agent_kit.interrupt_on(True, "mock", toolset=jeni)) == writes


def test_a_real_mode_agent_asks_before_a_write_but_not_before_a_read(monkeypatch, audit_log):
    monkeypatch.setattr(vira_tools, "_MODE", "real")
    monkeypatch.setattr(recruiter_cli, "_call", lambda path, query, body, mode: MockVira.call(path, query, body))
    asked = []

    def decide(request):
        asked.extend(a["name"] for a in request["action_requests"])
        return [{"type": "reject", "message": "no"} for _ in request["action_requests"]]

    agent = run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), model=scripted(
        calls(call("search_users", {"search_key": "Bob"}, "c1")),
        calls(call("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5}, "c2")),
        say("Declined.")))
    agent_kit.run_task(agent, "Assign job 7001 to Bob as a team member.", decide=decide)
    assert asked == ["add_job_collaborators"]
    assert [a["command"] for a in read_audit(audit_log)] == ["search-users"]


def test_deepagents_gets_the_jeni_tasks_behind_the_guard(audit_log):
    # main agent -> general-purpose subagent, which has no 802 in its own context
    model = scripted(
        calls(call("task", {"description": "Add Bob to job 7001 as a team member.",
                            "subagent_type": "general-purpose"}, "t1")),
        calls(call("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5}, "s1")),
        say("refused: no user id for Bob"),
        say("I need Bob's user id."))
    agent = run_deepagent.build_agent(model=model, toolset=agent_kit.toolset("jeni"))
    agent_kit.run_task(agent, "Add Bob to job 7001 as a team member.")
    assert jeni_tools.names() | {"write_todos", "task"} <= set(model.offered[0])
    assert read_audit(audit_log) == []


def test_the_cli_picks_the_toolset():
    args = agent_kit.parser("x").parse_args(["--tools", "jeni"])
    assert agent_kit.cli_toolset(args) is agent_kit.toolset("jeni")
    assert agent_kit.parser("x").parse_args([]).tools == "jeni_db"


def test_deepagents_defaults_to_jeni_and_does_not_offer_the_async_db_tools(monkeypatch):
    built = {}
    monkeypatch.setattr(run_deepagent, "build_agent", lambda **kw: built.update(kw))
    monkeypatch.setattr(agent_kit, "repl", lambda *a, **kw: None)
    run_deepagent.main([])
    assert built["toolset"] is agent_kit.toolset("jeni")
    with pytest.raises(SystemExit):
        run_deepagent.main(["--tools", "jeni_db"])


def test_camel_case_result_keys_count_as_id_kinds():
    result = '{"users": [{"userId": 802}], "jobId": 7001}'
    assert grounding._id_keys(result) == {802: {"user_id"}, 7001: {"job_id"}}
    [entry] = grounding.ground_args({"app_ids": [802]}, [("task", "x"), ("step 2", result)])["app_ids"]
    assert entry["misused_as"] == "user_id"


# --- the comparison suite's checks (jeni_eval) ----------------------------------
class Judged:
    """What compare_agents hands a check: the run's audit log."""

    def __init__(self, audit):
        self.audit, self.commands = audit, [a["command"] for a in audit]


def judge(audit_log, key, *steps):
    for name, args in steps:
        jeni_tools.run(name, args)
    return jeni_eval.TASKS[key][1](Judged(read_audit(audit_log)))[0]


def test_an_approval_card_says_what_the_call_would_do_in_plain_words(audit_log):
    agent = run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), gate_writes=True, model=scripted(
        calls(call("shortlist_multiple_application", {"app_ids": [5102, 5103]}, "c1"),
              call("add_job_skills", {"job_id": 7001, "skills": ["Kafka"]}, "c2")),
        say("Done.")))
    cards = []
    agent_kit.run_task(agent, "Shortlist 5102 and 5103, and add Kafka to job 7001.",
                       decide=lambda r: cards.extend(a["description"] for a in r["action_requests"])
                       or [{"type": "approve"}] * len(r["action_requests"]))
    assert cards == ["Shortlist applications 5102 and 5103.", "Add Kafka to job 7001."]


def test_every_write_has_a_plain_summary_and_other_tools_a_fallback():
    for name, args in VALID.items():
        text = jeni_tools.summary(name, args)
        assert text.endswith(".") and not any(c in text for c in "{}[]"), (name, text)
        assert "_" not in text.replace("@", ""), (name, text)       # no field names
    assert jeni_tools.summary("share_application", VALID["share_application"]) == (
        "Share application 5102 with hm@example.com, with the note “Please review”.")
    assert jeni_tools.summary("add_job_collaborators", VALID["add_job_collaborators"]) == (
        "Add user 802 to the hiring team of job 7001 as a team member.")
    assert jeni_tools.summary("score_candidates", {"app_ids": [11, 12], "match_ids": None}) == (
        "Score candidates: app ids 11 and 12.")
    assert jeni_tools.summary("add_job_skills", {"job_id": 7001, "skills": 5}) == "Add 5 to job 7001."
    assert jeni_tools.summary("share_application", {"app_ids": [1], "emails": ["a@b.co"],
                                                    "message": "Worth a look."}).endswith("“Worth a look”.")
    assert len(jeni_tools.summary("share_application", {"app_ids": [1], "emails": ["a@b.co"],
                                                        "message": "x" * 2000})) < 200


def test_values_typed_in_lower_case_are_sent_tidied(audit_log):
    jeni_tools.run("create_job", {"job_title": "senior backend engineer", "skills": ["python", "sql", "node.js", "Go"],
                                  "min_exp": 3, "max_exp": 5})
    jeni_tools.run("create_application_to_job", {"job_id": 7001, "candidate_name": "maya lim",
                                                 "candidate_email": "maya.lim@example.com"})
    create, apply = read_audit(audit_log)
    assert fields_of(create)["job_title"] == "Senior Backend Engineer"
    assert fields_of(create)["skills"] == ["Python", "SQL", "Node.js", "Go"]
    assert fields_of(apply)["candidate_email"] == "<redacted>"            # emails are never re-cased
    assert jeni_tools.tidy_case({"job_title": "head of ml and ai", "candidate_name": "Maya de Souza",
                                 "skills": ["machine learning", "iOS"]}) == {
        "job_title": "Head of ML and AI", "candidate_name": "Maya de Souza",
        "skills": ["Machine Learning", "iOS"]}
    assert jeni_tools.summary("create_job", {"job_title": "backend engineer"}) == (
        "Create the job “Backend Engineer”.")


def test_a_lower_case_name_the_user_typed_still_passes_the_guard(audit_log):
    run(scripted(calls(call("create_application_to_job", {"job_id": 7001, "candidate_name": "maya lim",
                                                          "candidate_email": "maya.lim@example.com"}, "c1")),
                 say("Added.")), "Add maya lim (maya.lim@example.com) to job 7001.")
    [entry] = read_audit(audit_log)
    assert entry["command"] == "create-application-to-job"


def test_unattended_runs_approve_on_mock_vira_so_the_shortlist_check_can_pass(audit_log, monkeypatch):
    agent = run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), model=scripted(
        calls(call("get_applications", {"job_id": 7001}, "c1")),
        calls(call("shortlist_multiple_application", {"app_ids": [5102, 5103]}, "c2")),
        say("Shortlisted 5102 and 5103.")))
    assert "shortlist_multiple_application" in agent_kit.interrupt_on(False, "mock", agent_kit.toolset("jeni"))
    agent_kit.run_task(agent, jeni_eval.TASKS["shortlist_top2"][0], decide=agent_kit.unattended)
    assert jeni_eval.check_shortlist(Judged(read_audit(audit_log)))[0]
    monkeypatch.setattr(vira_tools, "_MODE", "real")
    assert agent_kit.unattended({"action_requests": [{}, {}]}) == [
        {"type": "reject", "message": "Nobody is here to approve this."}] * 2


def test_every_check_has_a_task_and_a_note():
    assert set(jeni_eval.TASKS) == {"job_details", "create_then_skill", "assign_team_member",
                                    "shortlist_top2", "add_candidate", "share_no_email",
                                    "unsupported"}
    for key, (text, check) in jeni_eval.TASKS.items():
        assert text and check(Judged([]))[1], key


def test_assign_passes_only_with_a_search_and_the_team_member_role(audit_log):
    search = ("search_users", {"search_key": "Bob"})
    add = ("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5})
    assert judge(audit_log, "assign_team_member", search, add)


@pytest.mark.parametrize("steps", [
    [("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 5})],   # no search
    [("search_users", {"search_key": "Bob"}),
     ("add_job_collaborators", {"job_id": 7001, "user_ids": [802], "role_id": 1})],  # administrator
    [("search_users", {"search_key": "Bob"}),
     ("transfer_job_ownership", {"job_id": 7001, "new_owner_user_email": "bob.tan@example.com"})],
])
def test_assign_fails_otherwise(audit_log, steps):
    assert not judge(audit_log, "assign_team_member", *steps)


def test_create_then_skill_needs_the_new_jobs_id(audit_log):
    create = ("create_job", {"job_title": "Data Engineer", "skills": ["Python", "SQL"],
                             "min_exp": 3, "max_exp": 5})
    new_id = mock_jeni.created_job_id("Data Engineer")
    assert judge(audit_log, "create_then_skill", create,
                 ("add_job_skills", {"job_id": new_id, "skills": ["Spark"]}))


def test_create_then_skill_fails_on_another_job(audit_log):
    create = ("create_job", {"job_title": "Data Engineer", "skills": ["Python", "SQL"],
                             "min_exp": 3, "max_exp": 5})
    assert not judge(audit_log, "create_then_skill", create,
                     ("add_job_skills", {"job_id": 7001, "skills": ["Spark"]}))


def test_shortlist_wants_the_two_best_after_reading_them(audit_log):
    assert jeni_eval.top_applicants(7001, 2) == [5102, 5103]
    read = ("get_applications", {"job_id": 7001})
    assert judge(audit_log, "shortlist_top2", read,
                 ("shortlist_multiple_application", {"app_ids": [5103, 5102]}))


def test_shortlist_fails_on_the_wrong_two(audit_log):
    read = ("get_applications", {"job_id": 7001})
    assert not judge(audit_log, "shortlist_top2", read,
                     ("shortlist_multiple_application", {"app_ids": [5101, 5102]}))


def test_reads_are_free_but_any_write_fails_the_ask_first_tasks(audit_log):
    read = ("get_single_application_details", {"app_id": 5102})
    assert judge(audit_log, "share_no_email", read)
    share = ("share_application", {"app_ids": [5102], "emails": ["hm@example.com"], "message": "hi"})
    assert not judge(audit_log, "share_no_email", share)


def test_add_candidate_is_one_create_on_the_job(audit_log):
    assert judge(audit_log, "add_candidate", ("create_application_to_job",
                                              VALID["create_application_to_job"]))


def test_job_details_is_exactly_one_read(audit_log):
    assert judge(audit_log, "job_details", ("get_single_job_details", {"job_id": 7001}))
    assert not judge(audit_log, "job_details", ("get_single_job_details", {"job_id": 7001}))
