"""run_deepagent: deepagents harness over the same typed VIRA tools."""
import pytest

import agent_kit
import recruiter_cli
import run_deepagent
import vira_tools
from conftest import read_audit
from fakes import call, calls, say, scripted

vira_tools.configure("mock")

def backend_of(tool):
    """The backend a deepagents filesystem tool closes over (via its FilesystemMiddleware)."""
    for cell in tool.func.__closure__ or ():
        backend = getattr(cell.cell_contents, "backend", None)
        if backend is not None:
            return backend
    raise AssertionError(f"no backend found for {tool.name}")


def test_state_backend_only_and_no_shell():
    model = scripted(say("hi"))
    agent = run_deepagent.build_agent(model=model)
    assert agent_kit.final_text(agent_kit.run_task(agent, "hello")) == "hi"

    offered = set(model.offered[0])
    assert {"find_talents", "generate_jd", "score_candidates", "candidate_insights",
            "write_todos", "task", "read_file", "write_file"} <= offered
    assert "execute" not in offered                  # never offered to the model

    tools = agent.nodes["tools"].bound.tools_by_name
    for name in ("ls", "read_file", "write_file", "edit_file", "glob", "grep"):
        assert type(backend_of(tools[name])).__name__ == "StateBackend", name


def test_a_hallucinated_execute_call_runs_nothing():
    agent = run_deepagent.build_agent(model=scripted(
        calls(call("execute", {"command": "env"}, "x1")), say("ok")))
    result = agent_kit.run_task(agent, "run env")
    refused = next(m for m in result["messages"] if m.type == "tool" and m.tool_call_id == "x1")
    assert refused.status == "error" and "Execution not available" in refused.text


def test_main_agent_calls_vira_through_the_guarded_path(audit_log):
    result = agent_kit.run_task(
        run_deepagent.build_agent(model=scripted(
            calls(call("find_talents", {"job_ids": [123]}, "c1")),
            calls(call("find_talents", {"job_ids": [123]}, "c2")),     # repeat: refused
            say("3 profiles."))),
        "find talents for job 123")
    assert [a["command"] for a in read_audit(audit_log)] == ["find-talents"]
    refused = next(m for m in result["messages"] if m.type == "tool" and m.tool_call_id == "c2")
    assert "Refused" in refused.text


def test_subagent_runs_vira_and_returns_only_its_answer(audit_log):
    # One script, consumed in order: main -> subagent (tool call, answer) -> main.
    model = scripted(
        calls(call("task", {"description": "Find talents for job 7.",
                            "subagent_type": "sourcing-analyst"}, "t1")),
        calls(call("find_talents", {"job_ids": [7]}, "s1")),
        say("job 7: 900001, 900002, 900003"),
        say("Done: 3 profiles for job 7."))
    result = agent_kit.run_task(run_deepagent.build_agent(model=model), "talents for job 7")
    assert [a["command"] for a in read_audit(audit_log)] == ["find-talents"]
    task_result = next(m for m in result["messages"] if m.type == "tool" and m.tool_call_id == "t1")
    assert "900001" in task_result.text
    # The subagent's own tool call never enters the main agent's context.
    assert not any(tc["id"] == "s1" for m in result["messages"] if m.type == "ai"
                   for tc in m.tool_calls)


def test_subagents_inherit_approval(audit_log):
    model = scripted(
        calls(call("task", {"description": "Score applicant 11.",
                            "subagent_type": "sourcing-analyst"}, "t1")),
        calls(call("score_candidates", {"app_ids": [11]}, "s1")),
        say("could not score: declined"),
        say("Scoring was declined."))
    seen = []

    def decide(request):
        seen.extend(a["name"] for a in request["action_requests"])
        return [{"type": "reject", "message": "no"} for _ in request["action_requests"]]

    agent = run_deepagent.build_agent(model=model, approve_all=True)
    agent_kit.run_task(agent, "score applicant 11", decide=decide)
    assert "score_candidates" in seen
    assert read_audit(audit_log) == []


def test_virtual_files_stay_in_state(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    model = scripted(
        calls(call("write_file", {"file_path": "/report.md", "content": "# Report\nok"}, "w1")),
        say("Wrote /report.md"))
    result = agent_kit.run_task(run_deepagent.build_agent(model=model), "write a report")
    assert result["files"]["/report.md"]["content"] == "# Report\nok"
    assert list(tmp_path.iterdir()) == []                       # nothing on the host disk
    run_deepagent.show_files(result)
    assert "# Report" in capsys.readouterr().out
