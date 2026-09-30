"""The db lookups (db_queries, db_tools) and the jeni_db toolset, offline: fixtures and a fake pool."""
import asyncio
import contextlib
import json
from datetime import datetime, timezone

import pytest

import agent_kit
import db_queries
import db_tools
import jeni_tools
import mock_jeni
import run_langgraph
import vira_tools
from conftest import read_audit
from fakes import call, calls, say, scripted

vira_tools.configure("mock")

CID = 1
CTX = {"auth_profile": {"company_id": CID}}
FIXTURES = {
    "company_id": CID,
    "jobs": [{"jobId": 501, "jobName": "Data Scientist", "company_id": CID,
              "openDate": datetime(2026, 8, 21, tzinfo=timezone.utc)},
             {"jobId": 502, "jobName": "Senior Data Scientist", "company_id": CID,
              "openDate": datetime(2026, 8, 27, tzinfo=timezone.utc)},
             {"jobId": 900, "jobName": "Data Scientist", "company_id": 2}],
    "users": [{"userId": 9, "firstname": "Bob", "lastname": "Lee", "email": "bob@example.com",
               "company_id": CID},
              {"userId": 10, "firstname": "Bobby", "lastname": "Tan", "email": "bobby@example.com",
               "company_id": CID}],
    "applications": [{"app_id": 11, "job_id": 501}, {"app_id": 12, "job_id": 501},
                     {"app_id": 91, "job_id": 900}],
}


def query(name, inputs, fixtures=FIXTURES, context=CTX):
    return asyncio.run(db_queries.fake_db_queries(fixtures)[name].handler(inputs, context))


def tools(fixtures=FIXTURES):
    return {t.name: t for t in db_tools.langchain_tools(db_queries.fake_db_queries(fixtures), CTX)}


def tool(name, args, fixtures=FIXTURES):
    """What the model reads back from one db tool call."""
    return json.loads(asyncio.run(tools(fixtures)[name].ainvoke(args)))


class FakeConn:
    """Records the SQL and parameters the real queries send; answers with fixed rows."""

    def __init__(self, rows):
        self.rows, self.seen = rows, []

    async def fetch(self, sql, *params):
        self.seen.append((" ".join(sql.split()), params))
        return self.rows


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield self.conn

    async def close(self):
        pass


def real(name, inputs, rows):
    conn = FakeConn(rows)
    result = asyncio.run(db_queries.build_db_queries(FakePool(conn))[name].handler(inputs, CTX))
    return result, conn.seen


# --- the tool contract ---------------------------------------------------------------
def test_the_model_supplies_search_terms_and_ids_but_never_the_company():
    offered = tools()
    assert set(offered) == db_tools.READ_ONLY
    for name, t in offered.items():
        assert not {"company_id", "context", "auth_profile"} & set(t.args), name


def test_a_query_without_a_tenant_is_an_error_not_a_guess():
    assert query("validate_job_id", {"job_id": 501}, context={})["status"] == "error"


def test_another_companys_job_lists_no_applications_and_fails_validation():
    assert query("list_job_applications", {"job_id": 900})["applications"] == []
    assert query("validate_job_id", {"job_id": 900})["status"] == "not_found"
    assert query("validate_app_ids", {"app_ids": [91]})["invalid_values"] == [91]


def test_one_match_resolves_and_several_are_ambiguous_with_their_open_dates():
    assert tool("find_job_by_title", {"title": "senior"}) == {"status": "resolved", "job_id": 502}
    found = tool("find_job_by_title", {"title": "data scientist"})
    assert found["status"] == "ambiguous" and found["message"] == "2 jobs match"
    assert found["candidates"] == [
        {"job_id": 502, "label": "Senior Data Scientist (opened 2026-08-27)"},
        {"job_id": 501, "label": "Data Scientist (opened 2026-08-21)"}]


def test_a_long_list_says_it_is_cut_short():
    jobs = [{"jobId": n, "jobName": "Data Scientist", "company_id": CID} for n in range(1, 13)]
    found = tool("find_job_by_title", {"title": "data"}, fixtures={**FIXTURES, "jobs": jobs})
    assert [c["job_id"] for c in found["candidates"]] == list(range(12, 2, -1))    # newest 10
    assert found["message"].startswith("more than 10 jobs match")


def test_user_search_results_reach_the_model_masked():
    found = tool("find_user", {"search_key": "bob"})
    assert found["status"] == "ambiguous"
    assert [c["user_id"] for c in found["candidates"]] == [9, 10]
    assert "@example.com" not in json.dumps(found)


def test_validation_names_the_ids_that_are_not_found():
    assert tool("validate_app_ids", {"app_ids": [11, 999]}) == {
        "status": "not_found", "valid_values": [11], "invalid_values": [999],
        "message": "unknown or unavailable application IDs"}


# --- the real queries, against a fake pool ---------------------------------------------
def test_the_real_application_list_is_scoped_to_the_tenant():
    result, [(sql, params)] = real("list_job_applications", {"job_id": 501},
                                   [{"app_id": 11}, {"app_id": 12}])
    assert "recuiter_company_id = $1" in sql and params == (CID, 501)
    assert result["applications"] == [11, 12]


def test_the_real_title_search_is_literal_and_asks_for_one_more_row():
    rows = [{"job_id": 502, "job_name": "Senior Data Scientist",
             "open_date": datetime(2026, 8, 27, tzinfo=timezone.utc)},
            {"job_id": 501, "job_name": "Data Scientist", "open_date": None}]
    result, [(sql, params)] = real("find_job_by_title", {"title": "100%_sure"}, rows)
    assert "recuiter_company_id = $1" in sql
    assert params == (CID, "%100\\%\\_sure%", db_queries.SEARCH_LIMIT + 1)
    assert [c["label"] for c in result["candidates"]] == [
        "Senior Data Scientist (opened 2026-08-27)", "Data Scientist"]


def test_the_real_user_search_is_literal_and_tenant_scoped():
    _, [(sql, params)] = real("find_user", {"search_key": "a_b"}, [])
    assert "ui.company_id = $1" in sql and params == (CID, "%a\\_b%", db_queries.SEARCH_LIMIT + 1)


# --- jeni_db: one agent that looks up, then acts ---------------------------------------
def jeni_db(fixtures=FIXTURES):
    return agent_kit.toolset("jeni_db", query_tools=db_queries.fake_db_queries(fixtures),
                             context=CTX)


def test_jeni_db_is_jeni_plus_the_read_only_db_tools():
    ts = jeni_db()
    assert ts.names == jeni_tools.names() | db_tools.READ_ONLY
    assert db_tools.READ_ONLY <= ts.read_only
    assert "Jeni rules" in ts.prompt and "Database lookup rules" in ts.prompt
    gated = agent_kit.interrupt_on(False, "real", toolset=ts)
    assert "shortlist_multiple_application" in gated and not db_tools.READ_ONLY & set(gated)
    with pytest.raises(SystemExit):
        agent_kit.toolset("jeni_db")          # no pool or context: the runner must supply both


def approve(request):
    """A reviewer who approves every call (shortlist pauses in any mode)."""
    return [{"type": "approve"} for _ in request["action_requests"]]


def arun(model, task, ts=None):
    agent = run_langgraph.build_agent(model=model, toolset=ts or jeni_db())
    return asyncio.run(agent_kit.arun_task(agent, task, decide=approve))


def test_ids_the_db_returned_are_used_to_act(audit_log):
    arun(scripted(calls(call("find_job_by_title", {"title": "senior data scientist"}, "c1")),
                  calls(call("list_job_applications", {"job_id": 501}, "c2")),
                  calls(call("shortlist_multiple_application", {"app_ids": [11, 12]}, "c3")),
                  say("done")),
         "Shortlist the applicants of job 501, and find the senior data scientist job.")
    [sent] = read_audit(audit_log)
    assert sent["command"] == "shortlist-multiple-application"
    fields = sent["body"]["tasks"][0]["sub_tasks"][0]["fields"]
    assert {"field_name": "app_ids", "mandatory": True, "field_value": [11, 12]} in fields


def test_an_id_from_nowhere_is_refused_by_the_db_tools_too():
    result = arun(scripted(calls(call("validate_job_id", {"job_id": 777}, "c1")), say("stop")),
                  "Validate the data scientist job.")
    [refused] = [m for m in result["messages"] if m.type == "tool"]
    assert "777 isn't in the task" in refused.text


def test_a_failing_query_is_an_error_result_not_a_crash():
    async def broken(inputs, context):
        raise RuntimeError("connection lost to 10.0.0.1")

    qt = {"validate_job_id": db_queries.QueryTool("Validate a job id.", {"job_id": "int"}, {},
                                                  broken)}
    ts = agent_kit.toolset("jeni_db", query_tools=qt, context=CTX)
    result = arun(scripted(calls(call("validate_job_id", {"job_id": 501}, "c1")), say("stop")),
                  "Validate job 501.", ts)
    [failed] = [m for m in result["messages"] if m.type == "tool"]
    assert json.loads(failed.text) == {"status": "error", "message": "tool failed (RuntimeError)"}


# --- the runner: the db follows --mode, like VIRA --------------------------------------
def test_mock_mode_looks_up_only_ids_the_task_mock_knows(audit_log):
    result = arun(scripted(
        calls(call("find_job_by_title", {"title": "backend"}, "c1")),
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Kubernetes"]}, "c2")),
        calls(call("list_job_applications", {"job_id": 7001}, "c3")),
        calls(call("shortlist_multiple_application", {"app_ids": [5101, 5102]}, "c4")),
        say("done")),
        "Add Kubernetes to the backend engineer job and shortlist two of its applicants.",
        jeni_db(mock_jeni.db_fixtures(CID)))
    replies = {m.name: json.loads(m.text) for m in result["messages"] if m.type == "tool"}
    assert replies["find_job_by_title"] == {"status": "resolved", "job_id": 7001}
    assert replies["add_job_skills"]["result"]["skills"][-1] == "Kubernetes"
    assert replies["list_job_applications"]["applications"] == [5101, 5102, 5103, 5104]
    assert replies["shortlist_multiple_application"]["result"]["failedArr"] == []
    assert [a["command"] for a in read_audit(audit_log)] == [
        "add-job-skills", "shortlist-multiple-application"]


def runner_args(*argv):
    return agent_kit.parser("x").parse_args(["--task", "hi", *argv])


@pytest.fixture
def runner(monkeypatch):
    """run_langgraph.amain up to the REPL: records the toolset it built and any pool it opened."""
    monkeypatch.setattr(vira_tools, "_MODE", "mock")      # amain configures the mode; restore it
    seen = {"opened": []}

    async def open_pool(dsn):
        seen["opened"].append(dsn)
        return FakePool(FakeConn([{"job_id": 686441}]))

    async def repl(label, agent, args, names):
        seen["names"] = names

    monkeypatch.setattr(run_langgraph, "_open_pool", open_pool)
    monkeypatch.setattr(run_langgraph, "build_agent", lambda **kw: seen.update(kw))
    monkeypatch.setattr(agent_kit, "arepl", repl)
    return seen


def validate_job(toolset, job_id):
    [validate] = [t for t in toolset.tools() if t.name == "validate_job_id"]
    return json.loads(asyncio.run(validate.ainvoke({"job_id": job_id})))["status"]


def test_mock_mode_ignores_a_dsn_and_answers_from_mock_vira(runner, capsys):
    asyncio.run(run_langgraph.amain(runner_args("--dsn", "postgres://staging")))
    assert runner["opened"] == []
    assert validate_job(runner["toolset"], 7001) == "resolved"
    assert "used with --mode real only" in capsys.readouterr().out


def test_real_mode_opens_the_dsn_and_needs_one(runner):
    asyncio.run(run_langgraph.amain(runner_args("--mode", "real", "--dsn", "postgres://staging")))
    assert runner["opened"] == ["postgres://staging"]
    assert validate_job(runner["toolset"], 686441) == "resolved"
    with pytest.raises(SystemExit, match="needs a Postgres DSN"):
        asyncio.run(run_langgraph.amain(runner_args("--mode", "real")))
