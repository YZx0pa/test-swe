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
import pii_vault
import vira_tools
from demo import app as demo_app
from demo import jeni_graph, rehearse, script
from conftest import STAND_INS, extend_catalog
from fakes import call, calls, say, scripted

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def stateless_after():
    """The demo keeps the mock's State; every other test expects a stateless mock."""
    yield
    mock_jeni.forget_changes()
    vira_tools.configure("mock")
    pii_vault.VAULT.user_email = None


def demo_agent(*replies, gate_writes=jeni_graph.GATE_WRITES):
    """The demo graph with a checkpointer of its own: the server supplies one in the demo."""
    return jeni_graph.build(model=scripted(*replies), gate_writes=gate_writes).copy(
        update={"checkpointer": InMemorySaver()})


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


def test_the_demo_prompt_is_jeni_db_s_and_ends_with_the_terminal_response():
    """The closing rules are agent_kit's own: a structured TerminalResponse (message, status)
    instead of SUMMARY/STATUS lines, so the demo no longer rewrites the prompt."""
    prompt = jeni_graph.demo_toolset(mock_jeni.State()).prompt
    assert prompt == agent_kit.SYSTEM_PROMPT + jeni_tools.RULES + db_tools.RULES
    assert "SUMMARY:" not in prompt and "STATUS:" not in prompt and "terminal response" in prompt


def cards_approved(asked):
    """A decide() that approves every card, recording which tools asked."""
    def approve(request):
        asked.extend(a["name"] for a in request["action_requests"])
        return [{"type": "approve"} for _ in request["action_requests"]]
    return approve


def test_routine_writes_run_high_stakes_ones_ask_and_changes_show_on_the_panel(audit_log):
    agent = demo_agent(
        calls(call("find_job_by_title", {"title": "backend engineer"}, "c1")),
        calls(call("get_single_job_details", {"job_id": 7001}, "c2")),
        calls(call("make_job_public", {"job_id": 7001}, "c3")),
        calls(call("publish_job_to_linkedin", {"job_id": 7001}, "c4")),
        calls(call("shortlist_multiple_application", {"app_ids": [5102]}, "c5")),
        say("It's public, on LinkedIn, and 5102 is shortlisted."))
    asked = []
    asyncio.run(agent_kit.arun_task(agent, "Make the backend engineer job public, publish it to "
                                           "LinkedIn and shortlist application 5102.",
                                    decide=cards_approved(asked)))
    assert asked == ["shortlist_multiple_application"]          # ALWAYS_CONFIRM only
    snap = TestClient(demo_app.app).get("/demo/state").json()
    [job] = [j for j in snap["jobs"] if j["jobId"] == 7001]
    assert job["isPrivate"] is False and job["linkedIn"] is True
    assert {a["appId"]: a["stage"] for a in job["applications"]}[5102] == "shortlisted"
    assert [(e["task"], e["kind"], e["status"]) for e in snap["activity"]] == [
        ("get_single_job_details", "read", "completed"),
        ("make_job_public", "write", "completed"),
        ("publish_job_to_linkedin", "write", "completed"),
        ("shortlist_multiple_application", "write", "completed")]


def test_with_gate_writes_every_write_asks(audit_log):
    agent = demo_agent(calls(call("make_job_public", {"job_id": 7001}, "c1")), say("Done."),
                       gate_writes=True)
    asked = []
    asyncio.run(agent_kit.arun_task(agent, "Make job 7001 public.", decide=cards_approved(asked)))
    assert asked == ["make_job_public"] and jeni_graph.GATE_WRITES is False


def test_a_rejected_transfer_never_reaches_the_mock(audit_log):
    agent = demo_agent(calls(call("transfer_job_ownership",
                                  {"job_id": 7003, "new_owner_user_email": "alice.johnson@example.com"},
                                  "c1")),
                       say("Okay, the ownership stays as it is."))
    cards = []
    asyncio.run(agent_kit.arun_task(
        agent, "Transfer ownership of job 7003 to alice.johnson@example.com.",
        decide=lambda r: cards.append(r) or [{"type": "reject", "message": "Not yet."}
                                             for _ in r["action_requests"]]))
    snap = TestClient(demo_app.app).get("/demo/state").json()
    assert len(cards) == 1 and snap["activity"] == []
    assert {j["jobId"]: j["owner"] for j in snap["jobs"]}[7003] is None


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
    served = TestClient(demo_app.app).get("/demo/prompts").json()
    assert sorted(served, key=lambda p: p["number"]) == [
        {"key": a.key, "number": n, "category": a.category, "title": a.title, "prompt": a.turns[0].say}
        for n, a in enumerate(script.ACTS, 1)]
    groups = [p["category"] for p in served]                   # grouped, in the UI's order
    assert groups == sorted(groups, key=script.CATEGORIES.index)


def test_every_act_has_a_starter_group_and_every_group_has_acts():
    assert {a.category for a in script.ACTS} == set(script.CATEGORIES)
    assert len({a.key for a in script.ACTS}) == len(script.ACTS) == 17


def test_every_check_fails_on_the_starting_data_unless_its_cards_are_named():
    snap = TestClient(demo_app.app).get("/demo/state").json()
    jobs = {j["jobId"]: j for j in snap["jobs"]}
    for a in script.ACTS:
        assert a.check(jobs, []) is (a.key in {"reject", "unsupported"}), a.key
    assert act("reject").cards == ("transfer_job_ownership",)


def test_an_edited_shortlist_passes_when_the_edit_is_what_ran(audit_log):
    server = InProcess(
        calls(call("find_job_by_title", {"title": "backend engineer"}, "c1")),
        calls(call("get_applications", {"job_id": 7001}, "c2")),
        calls(call("shortlist_multiple_application", {"app_ids": [5102, 5103]}, "c3")),
        say("Shortlisted 5102, as you edited it."))
    result, lines = rehearsed(server, "shortlist")
    assert result["ok"]
    assert any('→ edit {"app_ids": [5102]}' in line for line in lines)


def test_going_around_the_reviewer_gets_a_card_of_its_own(audit_log):
    """shortlist is ALWAYS_CONFIRM: a second attempt after an edit goes back to the reviewer,
    whose edit (5102 only) applies to it as well, so 5103 is never shortlisted."""
    server = InProcess(
        calls(call("find_job_by_title", {"title": "backend engineer"}, "c1")),
        calls(call("get_applications", {"job_id": 7001}, "c2")),
        calls(call("shortlist_multiple_application", {"app_ids": [5102, 5103]}, "c3")),
        calls(call("shortlist_multiple_application", {"app_ids": [5103]}, "c4")),
        say("Shortlisted 5102."))
    result, lines = rehearsed(server, "shortlist")
    assert sum(line.startswith("  card  :") for line in lines) == 2
    assert not any("reviewer already edited or rejected" in line for line in lines)
    [job] = [j for j in server.snapshot()["jobs"] if j["jobId"] == 7001]
    assert {a["appId"]: a["stage"] for a in job["applications"]}[5103] == "applied"


def test_a_reject_act_fails_if_the_card_never_appeared(audit_log):
    result, lines = rehearsed(InProcess(say("Which job do you mean?")), "reject")
    assert not result["ok"] and "  no approval card for transfer_job_ownership" in lines


@pytest.mark.parametrize("asks_first", [True, False])
def test_an_if_needed_turn_is_sent_only_while_the_check_fails(audit_log, asks_first):
    act_calls = [calls(call("find_job_by_title", {"title": "backend engineer"}, "c1"),
                       call("search_users", {"search_key": "Bob"}, "c2")),
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


# --- more to try (acts 9-17): each check passes on the calls it describes ----------------
def find(title, call_id="c1"):
    return calls(call("find_job_by_title", {"title": title}, call_id))


def more_to_try(key, priya):
    return {
        "clone": [find("backend engineer"), calls(call("clone_job", {"job_id": 7001}, "c2")),
                  calls(call("add_job_skills", {"job_id": mock_jeni.cloned_job_id(7001), "skills": ["Rust"]},
                             "c3")),
                  say("Copied it as job 8001 and added Rust to the copy.")],
        "tidy": [find("backend engineer"),
                 calls(call("remove_job_skills", {"job_id": 7001, "skills": ["PostgreSQL"]}, "c2")),
                 calls(call("make_job_public", {"job_id": 7001}, "c3")), say("Done.")],
        "edit": [find("product designer"),
                 calls(call("edit_job", {"job_id": 7003, "vacancy": 2, "min_exp": 2, "max_exp": 4}, "c2")),
                 say("Updated.")],
        "close": [find("product designer"),
                  calls(call("make_job_closed", {"job_id": 7003, "reason_for_closure": "The role has been filled"},
                             "c2")), say("Closed.")],
        "reject_weakest": [find("backend engineer"), calls(call("get_applications", {"job_id": 7001}, "c2")),
                           calls(call("reject_multiple_application", {"app_ids": [5104]}, "c3")),
                           say("Rejected 5104.")],
        "compare": [find("data analyst"), calls(call("get_applications", {"job_id": 7002}, "c2")),
                    say("5201, at 0.88, is shortlisted.")],
        "suggest": [find("backend engineer"),
                    calls(call("get_suggested_candidates_for_a_job", {"job_id": 7001}, "c2")),
                    say("Three suggested candidates.")],
        "me": [calls(call("share_application", {"app_ids": [5103], "emails": ["me"], "message": "For later"},
                          "c1")), say("Shared with you.")],
        "hand_over": [find("data analyst"), calls(call("search_users", {"search_key": "Priya"}, "c2")),
                      calls(call("transfer_job_ownership", {"job_id": 7002, "new_owner_user_email": priya}, "c3")),
                      say("Priya owns it now.")],
    }[key]


@pytest.fixture
def full_catalog(monkeypatch, tmp_path):
    extend_catalog(monkeypatch, tmp_path, *STAND_INS)


@pytest.mark.parametrize("key", ["clone", "tidy", "edit", "close", "reject_weakest", "compare", "suggest",
                                 "me", "hand_over"])
def test_each_act_to_try_passes_its_check_on_the_calls_it_describes(audit_log, full_catalog, key):
    priya = pii_vault.VAULT.token("priya.nair@example.com", pii_vault.COLLEAGUE)
    result, lines = rehearsed(InProcess(*more_to_try(key, priya)), key)
    assert result["ok"], "\n".join(lines)


def test_a_skill_added_to_the_original_fails_the_copy_act(audit_log, full_catalog):
    result, _ = rehearsed(InProcess(
        find("backend engineer"), calls(call("clone_job", {"job_id": 7001}, "c2")),
        calls(call("add_job_skills", {"job_id": 7001, "skills": ["Rust"]}, "c3")), say("Done.")), "clone")
    assert not result["ok"]


def test_rejecting_the_wrong_applicant_fails_the_reject_act(audit_log, full_catalog):
    result, _ = rehearsed(InProcess(
        find("backend engineer"), calls(call("get_applications", {"job_id": 7001}, "c2")),
        calls(call("reject_multiple_application", {"app_ids": [5101]}, "c3")), say("Rejected 5101.")),
        "reject_weakest")
    assert not result["ok"]


def test_every_act_starts_from_the_starting_data(audit_log):
    """One act's changes (a copied job makes "backend engineer" ambiguous) never reach the next."""
    server = InProcess(say("That isn't supported."), say("That isn't supported."))
    resets = []
    server.reset = lambda: resets.append(len(resets))
    rehearse.rehearse(server, [act("unsupported"), act("unsupported")], out=lambda line: None)
    assert len(resets) == 3                                    # before each act, and at the end
