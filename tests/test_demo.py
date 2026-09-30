"""demo/: the agent `langgraph dev` serves (langgraph.json), and the data panel's routes."""
import asyncio
import json
import os
import threading
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from starlette.testclient import TestClient

import agent_kit
import db_tools
import jeni_tools
import mock_jeni
import vira_tools
from demo import app as demo_app
from demo import jeni_graph, rehearse, script
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


# --- the script and its rehearsal (demo/script.py, demo/rehearse.py) --------------------
class InProcess:
    """rehearse.Server's interface on the demo graph in this process, with a scripted model."""

    def __init__(self, *replies):
        self.agent = demo_agent(*replies)
        self.panel = TestClient(demo_app.app)
        self.deleted = []

    @staticmethod
    def _config(thread):
        return {"configurable": {"thread_id": thread}}

    def new_thread(self):
        return uuid.uuid4().hex

    def send(self, thread, text):
        asyncio.run(self.agent.ainvoke({"messages": [{"role": "user", "content": text}]},
                                       self._config(thread)))

    def resume(self, thread, decisions):
        asyncio.run(self.agent.ainvoke(Command(resume={"decisions": decisions}), self._config(thread)))

    def state(self, thread):
        state = self.agent.get_state(self._config(thread))
        return {"messages": [{"type": m.type, "content": m.content,
                              "tool_calls": getattr(m, "tool_calls", None) or []}
                             for m in state.values.get("messages", [])],
                "interrupts": [{"value": i.value} for i in state.interrupts]}

    def snapshot(self):
        return self.panel.get("/demo/state").json()

    def reset(self):
        self.panel.post("/demo/reset")

    def delete_thread(self, thread):
        self.deleted.append(thread)


def act(key):
    return next(a for a in script.ACTS if a.key == key)


def rehearsed(server, key):
    mock_jeni.remember_changes().reset()
    lines = []
    return rehearse.run_act(server, act(key), out=lines.append), lines


def test_the_presenters_copy_has_every_prompt_and_the_ui_gets_the_first_ones():
    text = (ROOT / "demo" / "SCRIPT.md").read_text(encoding="utf-8")
    missing = [t.say for a in script.ACTS for t in a.turns if f"`{t.say}`" not in text]
    assert missing == []
    assert TestClient(demo_app.app).get("/demo/prompts").json() == [
        {"key": a.key, "title": a.title, "prompt": a.turns[0].say} for a in script.ACTS]


def test_every_check_fails_on_the_starting_data_unless_its_cards_are_named():
    snap = TestClient(demo_app.app).get("/demo/state").json()
    jobs = {j["jobId"]: j for j in snap["jobs"]}
    for a in script.ACTS:
        assert a.check(jobs, []) is (a.key in {"reject", "unsupported"}), a.key
    assert act("reject").cards == ("make_job_closed",)


def test_an_edited_shortlist_passes_when_the_edit_is_what_ran(audit_log):
    server = InProcess(
        calls(call("find_job_by_title", {"title": "backend engineer"}, "c1")),
        calls(call("get_applications", {"job_id": 7001}, "c2")),
        calls(call("shortlist_multiple_application", {"app_ids": [5102, 5103]}, "c3")),
        say("Shortlisted 5102, as you edited it."))
    result, lines = rehearsed(server, "shortlist")
    assert result["ok"]
    assert any('→ edit {"app_ids": [5102]}' in line for line in lines)


def test_going_around_the_reviewer_gets_no_card_and_is_refused(audit_log):
    server = InProcess(
        calls(call("find_job_by_title", {"title": "backend engineer"}, "c1")),
        calls(call("get_applications", {"job_id": 7001}, "c2")),
        calls(call("shortlist_multiple_application", {"app_ids": [5102, 5103]}, "c3")),
        calls(call("shortlist_multiple_application", {"app_ids": [5103]}, "c4")),
        say("Shortlisted 5102."))
    result, lines = rehearsed(server, "shortlist")
    assert result["ok"] and sum(line.startswith("  card  :") for line in lines) == 1
    assert any("reviewer already edited or rejected" in line for line in lines)
    [job] = [j for j in server.snapshot()["jobs"] if j["jobId"] == 7001]
    assert {a["appId"]: a["stage"] for a in job["applications"]}[5103] == "applied"


def test_a_reject_act_fails_if_the_card_never_appeared(audit_log):
    result, lines = rehearsed(InProcess(say("Which job do you mean?")), "reject")
    assert not result["ok"] and "  no approval card for make_job_closed" in lines


@pytest.mark.parametrize("asks_first", [True, False])
def test_an_if_needed_turn_is_sent_only_while_the_check_fails(audit_log, asks_first):
    act_calls = [calls(call("find_job_by_title", {"title": "backend engineer"}, "c1"),
                       call("find_user", {"search_key": "Bob"}, "c2")),
                 calls(call("add_job_collaborators",
                            {"job_id": 7001, "user_ids": [802], "role_id": 5}, "c3")),
                 say("Bob is on the team.")]
    replies = [say("As a team member or an administrator?"), *act_calls] if asks_first else act_calls
    result, lines = rehearsed(InProcess(*replies), "ask")
    assert result["ok"]
    assert ("  you   > As a team member." in lines) is asks_first


def test_a_rehearsal_deletes_its_threads_and_leaves_the_starting_data(audit_log):
    server = InProcess(calls(call("find_job_by_title", {"title": "backend engineer"}, "c1")),
                       calls(call("add_job_skills", {"job_id": 7001,
                                                     "skills": ["Kubernetes", "Terraform"]}, "c2")),
                       say("Added."))
    [result] = rehearse.rehearse(server, [act("lookup")], out=lambda line: None)
    assert result["ok"] and server.deleted == [result["thread"]]
    assert server.snapshot()["activity"] == []
    assert "| lookup | 1/1 |" in rehearse.summary([result], [act("lookup")])
