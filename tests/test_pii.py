"""pii_vault: email tokens instead of redaction, the decrypt step before a call, and "me"."""
import json

import pytest

import agent_kit
import db_queries
import db_tools
import jeni_tools
import mock_jeni
import pii_vault
import recruiter_cli
import run_langgraph
import vira_tools
from conftest import done, read_audit, reply
from fakes import call, calls, say, scripted
from mock_vira import MockVira

vira_tools.configure("mock")


@pytest.fixture
def sent(monkeypatch):
    """The bodies as they leave recruiter_cli._call for the mock: after the decrypt step."""
    bodies = []
    real = MockVira.call

    def record(path, query, body):
        bodies.append(body)
        return real(path, query, body)

    monkeypatch.setattr(MockVira, "call", staticmethod(record))
    return bodies


@pytest.fixture
def signed_in(monkeypatch):
    monkeypatch.setattr(pii_vault.VAULT, "user_email", "sam.lee@example.com")


def recipients(body):
    [task] = body["tasks"]
    return {f["field_name"]: f.get("field_value") for f in task["sub_tasks"][0]["fields"]}


def agent(*replies):
    return run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), model=scripted(*replies))


def approve(request):
    return [{"type": "approve"} for _ in request["action_requests"]]


# --- the vault ----------------------------------------------------------------
def test_a_token_hides_the_address_and_only_this_vault_can_open_it():
    vault = pii_vault.Vault(key=b"k" * 32)
    token = vault.token("Priya.Nair@example.com", pii_vault.COLLEAGUE)
    assert pii_vault.TOKEN.fullmatch(token) and "priya" not in token.lower()
    assert vault.token("priya.nair@example.com", pii_vault.RECORD) == token       # same address
    assert vault.reveal(token) == "Priya.Nair@example.com"
    assert vault.sources(token) == {pii_vault.COLLEAGUE, pii_vault.RECORD}
    nonce, sealed, _ = vault._entries[token]
    assert b"priya" not in sealed.lower()                                          # encrypted
    other = pii_vault.Vault(key=b"o" * 32)
    assert other.token("priya.nair@example.com", pii_vault.COLLEAGUE) != token
    with pytest.raises(pii_vault.UnknownToken):
        other.reveal(token)


def test_who_may_send_a_token_or_me():
    vault = pii_vault.Vault(key=b"k" * 32)
    colleague = vault.token("bob.tan@example.com", pii_vault.COLLEAGUE)
    candidate = vault.token("candidate5102@example.com", pii_vault.RECORD)
    assert vault.allowed("emails", colleague) and vault.allowed("new_owner_user_email", colleague)
    assert not vault.allowed("emails", candidate)                     # never a candidate's
    assert not vault.allowed("candidate_email", colleague)            # not a recipient field
    assert not vault.allowed("emails", "<email:000000000000>")        # one it never issued
    assert not vault.allowed("emails", "me")                          # nobody signed in
    vault.user_email = "sam.lee@example.com"
    assert vault.allowed("emails", "me") and vault.allowed("emails", vault.me())
    assert vault.display("me") == "you" and vault.display(colleague) == "b•••n@example.com"


# --- results, the decrypt step and the audit log --------------------------------
def test_results_carry_tokens_and_the_address_goes_out_only_at_send_time(audit_log, sent):
    found = jeni_tools.run("search_users", {"search_key": "Priya"})
    [priya] = reply(found)["users"]
    assert "@" not in json.dumps(found) and pii_vault.VAULT.sources(priya["email"]) >= {"colleague"}
    result = jeni_tools.run("share_application", {"app_ids": [5102], "emails": [priya["email"]],
                                                  "message": "Worth a call"})
    assert done(result)
    assert recipients(sent[-1])["emails"] == ["priya.nair@example.com"]    # decrypted for VIRA
    log = read_audit(audit_log)
    assert "priya.nair@example.com" not in json.dumps(log) and "<email:" not in json.dumps(log)
    assert recipients(log[-1]["body"])["emails"] == "<redacted>"            # the log stays redacted


def test_a_token_this_process_never_issued_sends_nothing(audit_log, sent):
    result = jeni_tools.run("share_application", {"app_ids": [5102], "emails": ["<email:000000000000>"],
                                                  "message": "hi"})
    assert result["request_status"] == "failed" and "didn't receive" in result["message"]
    assert sent == []


def test_the_cli_redacts_emails_since_its_tokens_die_with_the_process(capsys):
    recruiter_cli.main(["--mode", "mock", "find-talents", "--job-ids", "123"])
    assert "<email:" not in capsys.readouterr().out


# --- the agent: colleagues, candidates and "me" ------------------------------------
def test_a_colleague_found_by_search_can_be_shared_with(audit_log, sent):
    token = reply(jeni_tools.run("search_users", {"search_key": "Priya"}))["users"][0]["email"]
    cards = []
    agent_kit.run_task(
        run_langgraph.build_agent(toolset=agent_kit.toolset("jeni"), gate_writes=True, model=scripted(
            calls(call("search_users", {"search_key": "Priya"}, "c1")),
            calls(call("share_application", {"app_ids": [5102], "emails": [token], "message": "Have a look"}, "c2")),
            say("Shared with Priya."))),
        "Share application 5102 with Priya.",
        decide=lambda r: cards.extend(a["description"] for a in r["action_requests"]) or approve(r))
    assert cards == ["Share application 5102 with p•••r@example.com, with the note “Have a look”."]
    assert recipients(sent[-1])["emails"] == ["priya.nair@example.com"]


def test_a_candidates_email_token_is_refused_as_a_recipient(audit_log, sent):
    details = reply(jeni_tools.run("get_single_application_details", {"app_id": 5102}))
    candidate = details["candidateEmail"]
    assert pii_vault.TOKEN.fullmatch(candidate) and pii_vault.VAULT.sources(candidate) == {"record"}
    before = len(sent)
    result = agent_kit.run_task(agent(
        calls(call("share_application", {"app_ids": [5102], "emails": [candidate], "message": "x"}, "c1")),
        say("I can't send it there.")), "Share application 5102 with the candidate.", decide=approve)
    refused = [m for m in result["messages"] if m.type == "tool"][0]
    assert "must be exactly what the user wrote" in refused.text and len(sent) == before


def test_me_is_the_signed_in_user(audit_log, sent, signed_in):
    agent_kit.run_task(agent(
        calls(call("share_application", {"app_ids": [5102], "emails": ["me"], "message": "For later"}, "c1")),
        say("Shared with you.")), "Share application 5102 with me.", decide=approve)
    assert recipients(sent[-1])["emails"] == ["sam.lee@example.com"]
    assert jeni_tools.summary("share_application", {"app_ids": [5102], "emails": ["me"]}) == (
        "Share application 5102 with you.")
    assert reply(jeni_tools.run("transfer_job_ownership", {"job_id": 7001, "new_owner_user_email": "me"}))[
        "newOwnerUserId"] == 804


def test_me_without_a_signed_in_user_asks_for_the_address(audit_log, sent):
    result = jeni_tools.run("share_application", {"app_ids": [5102], "emails": ["me"], "message": "x"})
    assert result["request_status"] == "failed" and "type it" in result["message"] and sent == []


def test_candidate_fields_still_take_only_a_typed_address():
    token = pii_vault.VAULT.token("bob.tan@example.com", pii_vault.COLLEAGUE)
    for value in (token, "me"):
        result = jeni_tools.run("create_application_to_job", {"job_id": 7001, "candidate_name": "Bob",
                                                              "candidate_email": value})
        assert result["request_status"] == "failed" and result["message"].startswith("invalid arguments")


# --- the database checks (jeni_db) -----------------------------------------------
@pytest.mark.parametrize("to,allowed", [
    (["me", "priya"], True),                       # "me" and a colleague's token from search_users
    (["<email:000000000000>"], False),             # a token this process never issued
    (["stranger@elsewhere.example"], False),       # not a user of the company
])
def test_entity_checks_take_me_and_colleague_tokens_only(audit_log, sent, signed_in, to, allowed):
    import asyncio
    priya = reply(jeni_tools.run("search_users", {"search_key": "Priya"}))["users"][0]["email"]
    emails = [priya if r == "priya" else r for r in to]
    before = len(sent)
    queries = db_queries.fake_db_queries(mock_jeni.db_fixtures(1))
    context = {"auth_profile": {"company_id": 1}}
    toolset = agent_kit.toolset("jeni_db", query_tools=queries, context=context)
    agent = run_langgraph.build_agent(toolset=toolset, query_tools=queries, context=context, model=scripted(
        calls(call("share_application", {"app_ids": [5102], "emails": emails, "message": "x"}, "c1")),
        say("Done.")))
    result = asyncio.run(agent_kit.arun_task(agent, "Share application 5102 with me and Priya.",
                                             decide=approve))
    outcome = json.loads([m for m in result["messages"] if m.type == "tool"][0].text)
    if allowed:
        assert done(outcome)
        assert recipients(sent[-1])["emails"] == ["sam.lee@example.com", "priya.nair@example.com"]
    else:
        assert outcome["status"] == "validation_error" and len(sent) == before
