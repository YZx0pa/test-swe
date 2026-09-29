"""vira_tools + agent_kit on LangGraph: create_agent loop and the explicit workflow."""
import json

import pytest
from pydantic import ValidationError

import agent_kit
import recruiter_cli
import run_langgraph
import run_workflow
import vira_tools
from conftest import read_audit
from fakes import call, calls, say, scripted

vira_tools.configure("mock")


@pytest.fixture
def vira(monkeypatch):
    """Record every request that reaches the VIRA call; answer with a PII-bearing payload."""
    seen = []

    def fake_call(path, query, body, mode):
        seen.append({"path": path, "query": query, "body": body, "mode": mode})
        return {"status": "ok", "http_status": 200,
                "result": {"email": "jane@example.com", "profile_id": 900001}}

    monkeypatch.setattr(recruiter_cli, "_call", fake_call)
    return seen


def tool_messages(result):
    return {m.tool_call_id: m for m in result["messages"] if m.type == "tool"}


def run(model, task, **build):
    return agent_kit.run_task(run_langgraph.build_agent(model=model, **build), task)


# --- the tool contract -------------------------------------------------------
def test_tool_schemas_are_typed_and_described():
    tools = {t.name: t for t in vira_tools.langchain_tools()}
    schemas = {name: t.tool_call_schema.model_json_schema() for name, t in tools.items()}
    assert {name: set(s.get("required", [])) for name, s in schemas.items()} == {
        "find_talents": {"job_ids"},
        "generate_jd": {"job_title"},
        "score_candidates": set(),
        "candidate_insights": {"app_ids"},
    }
    for name, tool in tools.items():
        assert tool.description and "--" not in tool.description, name   # no CLI flag syntax
        assert all(p.get("description") for p in schemas[name]["properties"].values()), name


def test_tools_use_the_configured_mode_and_never_confirm(vira):
    vira_tools.find_talents([123])
    assert vira[0]["mode"] == "mock"
    with pytest.raises(ValueError):
        vira_tools.configure("prod")


def test_scoring_needs_some_ids_and_does_not_call_vira(vira, audit_log):
    assert vira_tools.score_candidates()["status"] == "error"
    assert vira == [] and read_audit(audit_log) == []


def test_vira_failure_becomes_an_error_result_without_hosts(monkeypatch):
    def down(*args):
        raise ConnectionError("http://vira.internal:8080/v1 refused")
    monkeypatch.setattr(recruiter_cli, "_call", down)
    assert vira_tools.find_talents([123]) == {
        "status": "error", "message": "VIRA call failed (ConnectionError)"}


def test_tracing_stays_off_even_with_langsmith_tracing_v2_set(monkeypatch):
    from langsmith import utils
    monkeypatch.setenv("LANGSMITH_TRACING_V2", "true")
    utils.get_env_var.cache_clear()
    assert utils.tracing_is_enabled() is True          # the variable alone turns it on
    agent_kit.set_tracing(False)
    assert utils.tracing_is_enabled() is False


def test_build_chat_model_accepts_litellm_style_ids(monkeypatch):
    for model_id in ("gpt-5-mini", "openai/gpt-5-mini"):
        model = agent_kit.build_chat_model(model_id)
        assert type(model).__name__ == "ChatOpenAI"
        assert model.model_name == "gpt-5-mini" and model.use_responses_api is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    assert type(agent_kit.build_chat_model("anthropic/claude-sonnet-5")).__name__ == "ChatAnthropic"


# --- create_agent loop -------------------------------------------------------
def test_agent_sees_only_masked_tool_output(vira, audit_log):
    result = run(scripted(calls(call("find_talents", {"job_ids": [123]}, "c1")),
                          say("Found profile 900001.")), "Find talents for job 123.")
    content = tool_messages(result)["c1"].text
    assert "jane@example.com" not in content and "<redacted>" in content
    assert [a["command"] for a in read_audit(audit_log)] == ["find-talents"]
    assert agent_kit.final_text(result) == "Found profile 900001."


def test_exact_repeat_is_refused_without_calling_vira(vira):
    result = run(scripted(
        calls(call("find_talents", {"job_ids": [123]}, "c1")),
        calls(call("find_talents", {"job_ids": [123], "profile_ids": []}, "c2")),  # same call
        say("done")), "Find talents for job 123.")
    assert len(vira) == 1
    refused = tool_messages(result)["c2"]
    assert refused.status == "error" and "Refused" in refused.text


def test_parallel_duplicates_run_once(vira):
    run(scripted(calls(call("find_talents", {"job_ids": [5]}, "a"),
                       call("find_talents", {"job_ids": [5]}, "b")),
                 say("done")), "Find talents for job 5.")
    assert len(vira) == 1


def test_different_args_are_not_duplicates(vira):
    run(scripted(calls(call("find_talents", {"job_ids": [1]}, "c1")),
                 calls(call("find_talents", {"job_ids": [2]}, "c2")),
                 say("done")), "Find talents for jobs 1 and 2.")
    assert len(vira) == 2


def test_an_id_of_the_wrong_kind_is_refused_before_vira(monkeypatch, audit_log):
    # Seen live with gpt-4o-mini: profile ids from find_talents sent to scoring as match ids.
    result = run(scripted(
        calls(call("find_talents", {"job_ids": [123]}, "c1")),
        calls(call("score_candidates", {"match_ids": [900001]}, "c2")),
        say("Stopped: I only have profile ids.")), task="Find talents for job 123 and score them.")
    refused = tool_messages(result)["c2"]
    assert refused.status == "error"
    assert "900001 is a profile_id, not a match_id" in refused.text
    assert [a["command"] for a in read_audit(audit_log)] == ["find-talents"]


def test_ids_the_user_named_are_not_refused(vira):
    run(scripted(calls(call("score_candidates", {"match_ids": [900001]}, "c1")), say("done")),
        task="Score match 900001.")
    assert len(vira) == 1


def test_an_invented_id_is_refused_before_vira(vira, audit_log):
    result = run(scripted(calls(call("score_candidates", {"app_ids": [77]}, "c1")),
                          say("Stopped: no application ids.")), task="Score the applicants.")
    refused = tool_messages(result)["c1"]
    assert refused.status == "error" and "77 isn't in the task or any earlier result" in refused.text
    assert vira == [] and read_audit(audit_log) == []


def test_ids_from_an_earlier_result_are_not_invented(vira):
    # the fake VIRA answers every call with profile_id 900001
    run(scripted(calls(call("find_talents", {"job_ids": [123]}, "c1")),
                 calls(call("find_talents", {"job_ids": [123], "profile_ids": [900001]}, "c2")),
                 say("done")), task="Find talents for job 123.")
    assert len(vira) == 2


@pytest.mark.parametrize("name,args", [
    ("find_talents", {"job_ids": list(range(1, 52))}),
    ("find_talents", {"job_ids": [0]}),
    ("score_candidates", {"app_ids": [-3]}),
    ("generate_jd", {"job_title": "x" * 201}),
    ("generate_jd", {"job_title": "Dev", "lang": "en; drop"}),
    ("generate_jd", {"job_title": "Dev", "skills": ["s"] * 31}),
    ("generate_jd", {"job_title": "Dev", "industry": ["i" * 201]}),
])
def test_schemas_reject_oversized_or_malformed_input(vira, name, args):
    tool = {t.name: t for t in vira_tools.langchain_tools()}[name]
    with pytest.raises(ValidationError):
        tool.invoke(args)
    assert vira == []


def test_a_crashing_tool_does_not_crash_the_run(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("internal detail")
    monkeypatch.setattr(vira_tools, "_call", boom)
    result = run(scripted(calls(call("find_talents", {"job_ids": [1]}, "c1")), say("done")),
                 "Find talents for job 1.")
    failed = tool_messages(result)["c1"]
    assert failed.status == "error" and "RuntimeError" in failed.text
    assert "internal detail" not in failed.text
    assert agent_kit.final_text(result) == "done"


def test_invalid_args_come_back_as_an_error_for_the_model(vira):
    result = run(scripted(calls(call("find_talents", {"job_ids": []}, "c1")), say("done")),
                 "Find talents.")
    assert tool_messages(result)["c1"].status == "error"
    assert vira == []


def test_model_call_cap_ends_the_run(vira):
    loop = [calls(call("find_talents", {"job_ids": [i]}, f"c{i}")) for i in range(1, 10)]
    run(scripted(*loop), "Find talents for jobs 1, 2, 3, 4, 5, 6, 7, 8 and 9.", step_limit=3)
    assert len(vira) == 3


# --- human in the loop -------------------------------------------------------
def approve_all_run(vira_model_replies, decision):
    requests = []

    def decide(request):
        requests.append(request)
        return [decision for _ in request["action_requests"]]

    agent = run_langgraph.build_agent(model=scripted(*vira_model_replies), approve_all=True)
    return agent_kit.run_task(agent, "Find talents for job 123.", decide=decide), requests


def test_approve_all_pauses_and_approval_runs_the_call(vira):
    result, requests = approve_all_run(
        [calls(call("find_talents", {"job_ids": [123]}, "c1")), say("done")],
        {"type": "approve"})
    [action] = requests[0]["action_requests"]
    assert action["name"] == "find_talents" and action["args"] == {"job_ids": [123]}
    assert len(vira) == 1 and vira[0]["body"]["job_ids"] == [123]


def test_rejection_means_vira_is_never_called(vira):
    result, _ = approve_all_run(
        [calls(call("find_talents", {"job_ids": [123]}, "c1")), say("stopped")],
        {"type": "reject", "message": "not now"})
    assert vira == []
    assert tool_messages(result)["c1"].status == "error"


def test_edit_runs_the_reviewers_args(vira):
    approve_all_run(
        [calls(call("find_talents", {"job_ids": [123]}, "c1")), say("done")],
        {"type": "edit", "edited_action": {"name": "find_talents", "args": {"job_ids": [999]}}})
    assert [v["body"]["job_ids"] for v in vira] == [[999]]


# --- explicit workflow -------------------------------------------------------
def test_workflow_waits_for_approval_then_runs_insights_on_the_shortlist(audit_log):
    asked = []
    state = run_workflow.run(run_workflow.build_graph(), [11, 12, 13], 2,
                             lambda req: asked.append(req) or True)
    assert asked[0]["shortlist"] == [12, 11]           # mock scores 11/12/13: 0.78/0.95/0.71
    assert state["approved"] is True
    assert [i["app_id"] for i in state["insights"]] == [12, 11]
    assert [a["command"] for a in read_audit(audit_log)] == ["score-candidates",
                                                            "candidate-insights"]
    assert "Shortlisted [12, 11]" in state["summary"]


def test_workflow_rejection_skips_insights(audit_log):
    state = run_workflow.run(run_workflow.build_graph(), [11, 12], 1, lambda req: False)
    assert state["approved"] is False and "insights" not in state
    assert [a["command"] for a in read_audit(audit_log)] == ["score-candidates"]


def test_workflow_summary_comes_from_the_model(audit_log):
    graph = run_workflow.build_graph(model=scripted(say("11 and 12 shortlisted.")))
    state = run_workflow.run(graph, [11, 12], 2, lambda req: True)
    assert state["summary"] == "11 and 12 shortlisted."


def test_workflow_stops_on_a_vira_error(monkeypatch, audit_log):
    monkeypatch.setattr(recruiter_cli, "_call", lambda *a: {"status": "error", "http_status": 500,
                                                            "result": {"message": "down"}})
    state = run_workflow.run(run_workflow.build_graph(), [11], 1, lambda req: True)
    assert state["error"].startswith("scoring failed") and "shortlist" not in state
    assert json.loads(state["error"].split(": ", 1)[1])["status"] == "error"
