"""demo/: the agent `langgraph dev` serves (langgraph.json), and the data panel's routes."""
import asyncio
import json
import os
import threading
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from starlette.testclient import TestClient

import agent_kit
import db_tools
import jeni_tools
import mock_jeni
import vira_tools
from demo import app as demo_app
from demo import jeni_graph
from fakes import call, calls, say, scripted

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def stateless_after():
    """The demo keeps the mock's State; every other test expects a stateless mock."""
    yield
    mock_jeni.forget_changes()
    vira_tools.configure("mock")


def demo_agent(*replies):
    """The demo graph with a checkpointer of its own: the server supplies one in the demo."""
    return jeni_graph.build(model=scripted(*replies)).copy(update={"checkpointer": InMemorySaver()})


def test_langgraph_json_serves_the_demo_graph_and_panel():
    config = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))
    assert config["graphs"] == {"jeni": "./demo/jeni_graph.py:make_graph"}
    assert config["http"] == {"app": "./demo/app.py:app"}
    assert config["env"] == ".env" and config["python_version"] == "3.11"


def test_importing_the_demo_changes_nothing():
    assert mock_jeni._kept is None and vira_tools.current_mode() == "mock"


def test_the_demo_agent_is_mock_only_and_brings_no_checkpointer(monkeypatch):
    monkeypatch.setattr(vira_tools, "_MODE", "real")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    agent = jeni_graph.build(model=scripted(say("hi")))
    assert vira_tools.current_mode() == "mock" and agent.checkpointer is None
    assert mock_jeni._kept is not None
    assert all(os.environ[v] == "false" for v in agent_kit.TRACING_VARS)


def test_the_demo_prompt_changes_only_the_closing_rule():
    prompt = jeni_graph.demo_toolset(mock_jeni.State()).prompt
    assert "SUMMARY:" not in prompt and jeni_graph.CHAT_RULE in prompt
    assert prompt.replace(jeni_graph.CHAT_RULE, jeni_graph.SUMMARY_RULE) == (
        agent_kit.SYSTEM_PROMPT + jeni_tools.RULES + db_tools.RULES)


def test_writes_pause_reads_do_not_and_changes_show_on_the_panel(audit_log):
    agent = demo_agent(
        calls(call("find_job_by_title", {"title": "backend engineer"}, "c1")),
        calls(call("get_single_job_details", {"job_id": 7001}, "c2")),
        calls(call("make_job_public", {"job_id": 7001}, "c3")),
        calls(call("publish_job_to_linkedin", {"job_id": 7001}, "c4")),
        say("It's public and on LinkedIn."))
    asked = []

    def approve(request):
        asked.extend(a["name"] for a in request["action_requests"])
        return [{"type": "approve"} for _ in request["action_requests"]]

    asyncio.run(agent_kit.arun_task(agent, "Make the backend engineer job public and publish it "
                                           "to LinkedIn.", decide=approve))
    assert asked == ["make_job_public", "publish_job_to_linkedin"]
    snap = TestClient(demo_app.app).get("/demo/state").json()
    [job] = [j for j in snap["jobs"] if j["jobId"] == 7001]
    assert job["isPrivate"] is False and job["linkedIn"] is True
    assert [(e["task"], e["kind"], e["status"]) for e in snap["activity"]] == [
        ("get_single_job_details", "read", "completed"),
        ("make_job_public", "write", "completed"),
        ("publish_job_to_linkedin", "write", "completed")]


def test_a_rejected_write_never_reaches_the_mock(audit_log):
    agent = demo_agent(calls(call("make_job_closed", {"job_id": 7003}, "c1")),
                       say("Okay, I left it open."))
    asyncio.run(agent_kit.arun_task(agent, "Close job 7003.", decide=lambda r: [
        {"type": "reject", "message": "Not yet."} for _ in r["action_requests"]]))
    snap = TestClient(demo_app.app).get("/demo/state").json()
    assert snap["activity"] == [] and {j["jobId"]: j["status"] for j in snap["jobs"]}[7003] == "open"


def test_the_panel_shows_resets_and_never_writes_html_from_data():
    client = TestClient(demo_app.app)
    page = client.get("/demo")
    assert page.status_code == 200 and "Jeni data" in page.text and "innerHTML" not in page.text
    state = mock_jeni.remember_changes()
    vira_tools.configure("mock")
    jeni_tools.run("add_job_skills", {"job_id": 7001, "skills": ["Kafka"]})
    assert len(client.get("/demo/state").json()["activity"]) == 1
    assert client.get("/demo/reset").status_code == 405
    assert client.post("/demo/reset").json() == {"status": "reset"}
    snap = client.get("/demo/state").json()
    assert snap["activity"] == [] and "Kafka" not in json.dumps(snap) and state is mock_jeni._kept


def test_the_factory_builds_once_off_the_event_loop(monkeypatch):
    built = []
    monkeypatch.setattr(jeni_graph, "build", lambda: built.append(threading.current_thread()) or "graph")
    monkeypatch.setattr(jeni_graph, "_graph", None)

    async def twice():
        return await jeni_graph.make_graph(), await jeni_graph.make_graph()

    assert asyncio.run(twice()) == ("graph", "graph")
    assert len(built) == 1 and built[0] is not threading.main_thread()
