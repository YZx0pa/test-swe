"""vira_tools + agent_kit on LangGraph: create_agent loop and the explicit workflow."""
import argparse
import json
import os
import stat

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
        "get_match_id_from_profile_id": {"job_id", "profile_ids"},
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
    assert "jane@example.com" not in content and "<email:" in content
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
    assert "get_match_id_from_profile_id" in refused.text      # points at the converter
    assert [a["command"] for a in read_audit(audit_log)] == ["find-talents"]


def test_profile_ids_are_looked_up_as_match_ids_then_scored(audit_log):
    # The id_trap task done right, against mock VIRA: find -> look up match ids -> score.
    result = run(scripted(
        calls(call("find_talents", {"job_ids": [123]}, "c1")),
        calls(call("get_match_id_from_profile_id",
                   {"job_id": 123, "profile_ids": [900001, 900002]}, "c2")),
        calls(call("score_candidates", {"match_ids": [123900001, 123900002]}, "c3")),
        say("Scored 2 talents.")), task="Find talents for job 123 and score them.")
    assert all(m.status != "error" for m in tool_messages(result).values())
    audit = read_audit(audit_log)
    assert [a["command"] for a in audit] == ["find-talents", "get-match-id-from-profile-id",
                                             "score-candidates"]
    assert audit[1]["body"] == {"job_id": 123, "profile_ids": [900001, 900002]}
    assert audit[2]["body"]["match_ids"] == [123900001, 123900002]


@pytest.mark.parametrize("job_id,reason", [
    (124, "124 isn't in the task or any earlier result"),
    (900001, "900001 is a profile_id, not a job_id"),
])
def test_the_lookup_refuses_a_job_id_the_model_cant_have(audit_log, job_id, reason):
    result = run(scripted(
        calls(call("find_talents", {"job_ids": [123]}, "c1")),
        calls(call("get_match_id_from_profile_id",
                   {"job_id": job_id, "profile_ids": [900001]}, "c2")),
        say("Stopped.")), task="Find talents for job 123 and score them.")
    refused = tool_messages(result)["c2"]
    assert refused.status == "error" and reason in refused.text
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
    ("get_match_id_from_profile_id", {"job_id": 123, "profile_ids": []}),
    ("get_match_id_from_profile_id", {"job_id": 0, "profile_ids": [900001]}),
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


# --- conversations: later turns continue the same thread ---------------------
def test_a_reply_continues_the_conversation_and_its_ids_are_the_users(vira):
    agent = run_langgraph.build_agent(model=scripted(
        say("Which applicants should I score?"),
        calls(call("score_candidates", {"app_ids": [11, 12]}, "c1")), say("Scored 11 and 12.")))
    agent_kit.run_task(agent, "Score the applicants.", thread_id="t1")
    result = agent_kit.run_task(agent, "Applicants 11 and 12.", thread_id="t1")
    assert [m.text for m in result["messages"] if m.type == "human"] == [
        "Score the applicants.", "Applicants 11 and 12."]
    assert tool_messages(result)["c1"].status != "error"      # 11 and 12 came from the user
    assert [v["body"]["app_ids"] for v in vira] == [[11, 12]]


def test_the_model_call_cap_is_per_turn_not_per_conversation(vira):
    agent = run_langgraph.build_agent(model=scripted(
        calls(call("find_talents", {"job_ids": [1]}, "c1")),
        calls(call("find_talents", {"job_ids": [2]}, "c2")), say("Found talents for 1 and 2."),
        calls(call("find_talents", {"job_ids": [3]}, "c3")), say("Found talents for 3.")),
        step_limit=3)
    agent_kit.run_task(agent, "Find talents for jobs 1 and 2.", thread_id="t1")
    result = agent_kit.run_task(agent, "Now job 3.", thread_id="t1")    # a 4th and 5th call
    assert len(vira) == 3 and agent_kit.final_text(result) == "Found talents for 3."


def test_a_repeat_in_a_later_turn_is_still_refused(vira):
    agent = run_langgraph.build_agent(model=scripted(
        calls(call("find_talents", {"job_ids": [123]}, "c1")), say("Found 900001."),
        calls(call("find_talents", {"job_ids": [123]}, "c2")), say("Same as before: 900001.")))
    agent_kit.run_task(agent, "Find talents for job 123.", thread_id="t1")
    result = agent_kit.run_task(agent, "Find them again.", thread_id="t1")
    assert len(vira) == 1 and "identical to an earlier call" in tool_messages(result)["c2"].text


def test_the_repl_keeps_one_conversation_until_new(vira, monkeypatch, capsys):
    agent = run_langgraph.build_agent(model=scripted(
        say("Which applicants should I score?"),
        calls(call("score_candidates", {"app_ids": [11, 12]}, "c1")), say("Scored 11 and 12."),
        say("Hello.")))
    replies = iter(["Score the applicants.", "Applicants 11 and 12.", "new", "hi", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    agent_kit.repl("LangGraph", agent,
                   argparse.Namespace(task=None, trace_json=None, mode="mock", tools="vira"))
    out = capsys.readouterr().out
    turns = [t.strip() for t in out.split("=== trajectory ===")[1:]]
    assert turns[0].startswith("[0] user      : Score the applicants.")
    assert turns[1].startswith("[0] user      : Applicants 11 and 12.")    # this turn only, numbered from 0
    assert "Score the applicants." not in turns[1]
    assert "(new conversation)" in out and turns[2].startswith("[0] user      : hi")
    assert [v["body"]["app_ids"] for v in vira] == [[11, 12]]


# --- human in the loop -------------------------------------------------------
def approve_all_run(vira_model_replies, decision, task="Score applicants 11 and 12."):
    requests = []

    def decide(request):
        requests.append(request)
        return [decision for _ in request["action_requests"]]

    agent = run_langgraph.build_agent(model=scripted(*vira_model_replies), approve_all=True)
    return agent_kit.run_task(agent, task, decide=decide), requests


def test_approve_all_pauses_and_approval_runs_the_call(vira):
    result, requests = approve_all_run(
        [calls(call("score_candidates", {"app_ids": [11]}, "c1")), say("done")],
        {"type": "approve"})
    [action] = requests[0]["action_requests"]
    assert action["name"] == "score_candidates" and action["args"] == {"app_ids": [11]}
    assert len(vira) == 1 and vira[0]["body"]["app_ids"] == [11]


def test_approve_all_does_not_pause_a_read(vira):
    result, requests = approve_all_run(
        [calls(call("find_talents", {"job_ids": [123]}, "c1")), say("done")],
        {"type": "reject", "message": "no"}, task="Find talents for job 123.")
    assert requests == [] and [v["body"]["job_ids"] for v in vira] == [[123]]


def test_rejection_means_vira_is_never_called(vira):
    result, _ = approve_all_run(
        [calls(call("score_candidates", {"app_ids": [11]}, "c1")), say("stopped")],
        {"type": "reject", "message": "not now"})
    assert vira == []
    assert tool_messages(result)["c1"].status == "error"


def test_edit_runs_the_reviewers_args(vira):
    approve_all_run(
        [calls(call("score_candidates", {"app_ids": [11, 12]}, "c1")), say("done")],
        {"type": "edit", "edited_action": {"name": "score_candidates", "args": {"app_ids": [12]}}})
    assert [v["body"]["app_ids"] for v in vira] == [[12]]


def test_real_mode_always_gates_the_calls_that_change_vira(monkeypatch):
    assert agent_kit.interrupt_on(False, "mock") == {}
    assert set(agent_kit.interrupt_on(False, "real")) == {"score_candidates", "candidate_insights"}
    assert set(agent_kit.interrupt_on(True, "mock")) == vira_tools.NAMES - vira_tools.READ_ONLY
    monkeypatch.setattr(vira_tools, "_MODE", "real")            # the default follows configure()
    assert set(agent_kit.interrupt_on(False)) == {"score_candidates", "candidate_insights"}


def test_a_real_mode_agent_pauses_before_scoring_but_not_before_a_read(vira, monkeypatch):
    monkeypatch.setattr(vira_tools, "_MODE", "real")
    asked = []

    def decide(request):
        asked.extend(a["name"] for a in request["action_requests"])
        return [{"type": "reject", "message": "no"} for _ in request["action_requests"]]

    agent = run_langgraph.build_agent(model=scripted(
        calls(call("find_talents", {"job_ids": [123]}, "c1")),
        calls(call("score_candidates", {"app_ids": [11]}, "c2")),
        say("Scoring was declined.")))
    agent_kit.run_task(agent, "Find talents for job 123, then score applicant 11.", decide=decide)
    assert asked == ["score_candidates"]
    assert [(v["path"], v["mode"]) for v in vira] == [("fast_retargeting", "real")]


def confirm(monkeypatch, answers, args):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    return agent_kit.ask_human({"action_requests": [{"name": "score_candidates", "args": args}]})


def test_enter_or_y_approves_and_n_rejects(monkeypatch):
    assert confirm(monkeypatch, [""], {"app_ids": [11]}) == [{"type": "approve"}]
    assert confirm(monkeypatch, ["y"], {"app_ids": [11]}) == [{"type": "approve"}]
    assert confirm(monkeypatch, ["n"], {"app_ids": [11]})[0]["type"] == "reject"


def test_e_replaces_the_arguments_with_a_json_object(monkeypatch, capsys):
    decisions = confirm(monkeypatch, ["e", "drop 11", "[12]", '{"app_ids": [12]}'], {"app_ids": [11, 12]})
    assert decisions == [{"type": "edit", "edited_action": {"name": "score_candidates",
                                                            "args": {"app_ids": [12]}}}]
    out = capsys.readouterr().out
    assert "not valid JSON" in out and "expected a JSON object" in out


def test_a_change_in_words_is_not_interpreted_but_asked_again(monkeypatch, capsys):
    assert confirm(monkeypatch, ["drop 11", "y"], {"app_ids": [11, 12]}) == [{"type": "approve"}]
    assert "enter y to approve, e to replace all arguments as JSON" in capsys.readouterr().out


# --- what reaches your terminal and disk -------------------------------------
def test_printable_strips_terminal_escapes_and_bidi_controls():
    raw = "ok\x1b]52;c;ZXZpbA==\x07\x1b[2J\u202egnp.exe\r\nnext\tcol\x9b"
    assert agent_kit.printable(raw) == "ok]52;c;ZXZpbA==[2Jgnp.exe\nnext\tcol"


def test_trajectories_print_without_escapes(vira, capsys):
    run(scripted(calls(call("find_talents", {"job_ids": [123]}, "c1")),
                 say("Found \x1b[2J\u202e3 profiles.")), "Find talents for job 123.")
    result = run(scripted(say("done \x1b]0;pwned\x07")), "Find talents.")
    agent_kit.show(result["messages"])
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out and "done ]0;pwned" in out


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_trace_json_is_owner_only_and_scrubbed(vira, tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_text("{}")
    path.chmod(0o644)                                   # an older, shared trace file
    trace = {"path": str(path), "model": "m", "mode": "mock", "repeat": 1, "tasks": {},
             "runs": [], "created": "now"}
    agent = run_langgraph.build_agent(model=scripted(
        calls(call("find_talents", {"job_ids": [123]}, "c1")),
        say("Found 900001; the recruiter is jane@example.com, +65 9123 4567.")))
    agent_kit.run_and_show(agent, "Find talents for job 123 (reply to jane@example.com).",
                           trace=trace, label="LangGraph")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text()
    assert "jane@example.com" not in text and "9123 4567" not in text
    assert json.loads(text)["runs"][0]["final"].startswith("Found 900001")
    agent_kit.write_private(tmp_path / "traces" / "runs.json", "{}")     # creates traces/
    assert stat.S_IMODE((tmp_path / "traces").stat().st_mode) == 0o700


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
