"""recruiter_cli: the CLI and the typed actions share one guarded path."""
import json
import os
import stat

import pytest

import recruiter_cli
from conftest import read_audit

PII_RESULT = {"candidate_name": "Jane Doe", "email": "jane@example.com", "score": 0.9}

# (CLI argv after --mode, the equivalent typed-action call)
CASES = [
    (["find-talents", "--job-ids", "123,124", "--profile-ids", "900001"],
     lambda: recruiter_cli.find_talents([123, 124], [900001], mode="mock")),
    (["generate-jd", "--job-title", "Data Analyst", "--skills", "SQL, Python", "--lang", "ar",
      "--job-id", "55", "--industry", "Fintech"],
     lambda: recruiter_cli.generate_jd("Data Analyst", ["SQL", "Python"], "ar", 55,
                                       industry=["Fintech"], mode="mock")),
    (["score-candidates", "--app-ids", "11,12", "--match-ids", "7"],
     lambda: recruiter_cli.score_candidates([11, 12], [7], mode="mock")),
    (["candidate-insights", "--app-ids", "11"],
     lambda: recruiter_cli.candidate_insights([11], mode="mock")),
    (["get-match-id-from-profile-id", "--job-id", "123", "--profile-ids", "900001,900002"],
     lambda: recruiter_cli.get_match_id_from_profile_id(123, [900001, 900002], mode="mock")),
]


@pytest.fixture
def calls(monkeypatch):
    """Record what would be sent to VIRA and answer with a PII-bearing payload."""
    seen = []

    def fake_call(path, query, body, mode):
        seen.append((path, query, body, mode))
        return {"status": "ok", "http_status": 200, "result": dict(PII_RESULT)}

    monkeypatch.setattr(recruiter_cli, "_call", fake_call)
    return seen


@pytest.mark.parametrize("argv,typed", CASES)
def test_cli_and_typed_action_send_the_same_request(argv, typed, calls, capsys):
    recruiter_cli.main(["--mode", "mock", *argv])
    typed()
    assert len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize("argv,typed", CASES)
def test_results_are_masked_on_both_paths(argv, typed, calls, capsys):
    result = typed()
    assert result["result"] == {"candidate_name": "<redacted>", "email": "<redacted>",
                                "score": 0.9}
    recruiter_cli.main(["--mode", "mock", *argv])
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("key", [
    "email", "Email", "emails", "email_address", "candidate_email", "name", "first_name",
    "lastName", "full-name", "username", "candidate_name", "phone", "phone_number", "mobile",
    "candidate_phone", "address", "home_address", "linkedin_url", "cv", "cv_url", "resume_text",
    "dob", "date_of_birth", "nric", "passport_no", "photo", "gender", "nationality"])
def test_pii_keys_are_masked_by_name_and_pattern(key):
    assert recruiter_cli._mask_pii({"scores": [{key: "x", "app_id": 11}]}) == {
        "scores": [{key: "<redacted>", "app_id": 11}]}


@pytest.mark.parametrize("key", [
    "job_name_similarity", "job_name", "job_title", "jobTitle", "job_description", "profile_id",
    "match_ids", "match_id", "matches", "job_id", "app_id", "overall_score", "skill_score", "composite_score", "briq",
    "summary", "skills", "industry", "lang", "status", "hotel", "result"])
def test_other_keys_are_kept(key):
    assert recruiter_cli._mask_pii({key: 7}) == {key: 7}


def test_emails_and_phones_inside_text_are_masked():
    text = ("Jane (jane.doe@example.com; +65 9123 4567; (555) 123-4567; 9123 4567) fits job "
            "123456, noted 2026-09-30, score 0.91 for applicants 11, 12 and 13, v1.20.4567. "
            "Call 555-123-4567.")
    out = recruiter_cli._mask_pii({"insights": [{"summary": text}]})["insights"][0]["summary"]
    assert out.count("<redacted-email>") == 1 and out.count("<redacted-phone>") == 4
    assert "jane.doe" not in out and "123-4567" not in out and "9123 4567" not in out
    for kept in ("fits job 123456", "2026-09-30", "0.91", "11, 12 and 13", "v1.20.4567"):
        assert kept in out
    assert out.endswith("Call <redacted-phone>.")          # a number ending a sentence


def test_uuids_are_kept_whole_and_phones_beside_them_still_masked():
    import uuid
    ids = [str(uuid.uuid4()) for _ in range(2000)] + ["b5f234c4-8969-4271-a0f3-7459042f100b"]
    assert [recruiter_cli._mask_pii({"agentTaskGroupUuid": u})["agentTaskGroupUuid"] for u in ids] == ids
    out = recruiter_cli._mask_pii({"m": "group b5f234c4-8969-4271-a0f3-7459042f100b, call +65 9123 4567"})["m"]
    assert out == "group b5f234c4-8969-4271-a0f3-7459042f100b, call <redacted-phone>"


@pytest.mark.parametrize("action", [
    lambda: recruiter_cli.find_talents(list(range(1, 52)), mode="mock"),
    lambda: recruiter_cli.find_talents([0], mode="mock"),
    lambda: recruiter_cli.score_candidates(["123.."], mode="mock"),
    lambda: recruiter_cli.candidate_insights([11], [True], mode="mock"),
    lambda: recruiter_cli.get_match_id_from_profile_id(0, [900001], mode="mock"),
    lambda: recruiter_cli.get_match_id_from_profile_id(123, list(range(1, 52)), mode="mock"),
    lambda: recruiter_cli.generate_jd("x" * 201, mode="mock"),
    lambda: recruiter_cli.generate_jd("Dev", lang="en;x", mode="mock"),
    lambda: recruiter_cli.generate_jd("Dev", skills=["s"] * 31, mode="mock"),
    lambda: recruiter_cli.generate_jd("Dev", job_id=-1, mode="mock"),
])
def test_bad_input_is_refused_before_anything_is_sent(action, calls, audit_log):
    result = action()
    assert result["status"] == "error" and result["result"]["message"]
    assert calls == [] and read_audit(audit_log) == []


def test_the_cli_reports_a_typo_in_ids(calls, capsys):
    recruiter_cli.main(["--mode", "mock", "find-talents", "--job-ids", "123.."])
    assert json.loads(capsys.readouterr().out)["result"]["message"] == (
        "job_ids: ids are positive integers")
    assert calls == []


def test_audit_line_masks_the_request_body(calls, audit_log):
    recruiter_cli.execute("find-talents", "fast_retargeting", {},
                          {"email": "jane@example.com", "job_ids": [1]}, mode="mock")
    [line] = read_audit(audit_log)
    assert line["command"] == "find-talents"
    assert line["body"] == {"email": "<redacted>", "job_ids": [1]}
    assert line["status"] == "ok"


def test_mode_is_required():
    with pytest.raises(TypeError):
        recruiter_cli.find_talents([123])
    with pytest.raises(SystemExit) as exc:                 # the CLI has no default either
        recruiter_cli.main(["find-talents", "--job-ids", "123"])
    assert exc.value.code == 2


def test_mock_mode_end_to_end(audit_log):
    result = recruiter_cli.find_talents([123], mode="mock")
    assert result["status"] == "ok"
    assert [p["profile_id"] for p in result["result"]["suggested_profiles"]] == [
        900001, 900002, 900003]
    assert [a["command"] for a in read_audit(audit_log)] == ["find-talents"]


def test_mock_match_id_lookup_is_per_job_and_never_a_profile_id(audit_log):
    result = recruiter_cli.get_match_id_from_profile_id(123, [900001, 900002], mode="mock")
    assert result["status"] == "ok"
    assert result["result"]["matches"] == [{"profile_id": 900001, "match_id": 123900001},
                                           {"profile_id": 900002, "match_id": 123900002}]
    other_job = recruiter_cli.get_match_id_from_profile_id(124, [900001], mode="mock")
    assert other_job["result"]["matches"][0]["match_id"] == 124900001
    assert [a["command"] for a in read_audit(audit_log)] == ["get-match-id-from-profile-id"] * 2


def test_confirm_gate_blocks_before_any_call(calls, audit_log, monkeypatch, capsys):
    monkeypatch.setattr(recruiter_cli, "NEEDS_CONFIRM", {"score-candidates"})

    result = recruiter_cli.score_candidates([11], mode="mock")
    assert result["status"] == "needs_confirmation"
    assert calls == [] and read_audit(audit_log) == []

    with pytest.raises(SystemExit) as exc:
        recruiter_cli.main(["--mode", "mock", "score-candidates", "--app-ids", "11"])
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "needs_confirmation"
    assert calls == []

    assert recruiter_cli.score_candidates([11], mode="mock", confirmed=True)["status"] == "ok"
    assert len(calls) == 1


def test_failed_calls_are_audited_and_the_cli_prints_one_json_line(monkeypatch, audit_log, capsys):
    def down(*args):
        raise ConnectionError("http://vira.internal:8080/v1 refused")
    monkeypatch.setattr(recruiter_cli, "_call", down)
    with pytest.raises(ConnectionError):                  # vira_tools turns this into a result
        recruiter_cli.find_talents([123], mode="mock")
    with pytest.raises(SystemExit) as exc:
        recruiter_cli.main(["--mode", "mock", "find-talents", "--job-ids", "123"])
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert json.loads(out) == {"status": "error", "message": "recruiter_cli failed (ConnectionError)"}
    assert "vira.internal" not in out + err and "Traceback" not in err
    assert [(a["status"], a["error"]) for a in read_audit(audit_log)] == [
        ("exception", "ConnectionError")] * 2


def test_audit_line_masks_the_query(calls, audit_log):
    recruiter_cli.execute("find-talents", "fast_retargeting", {"email": "jane@example.com"}, {},
                          mode="mock")
    assert read_audit(audit_log)[0]["query"] == {"email": "<redacted>"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_audit_log_is_owner_only(audit_log):
    recruiter_cli.find_talents([123], mode="mock")
    assert stat.S_IMODE(audit_log.stat().st_mode) == 0o600
    audit_log.chmod(0o644)                                # a log from before this change
    recruiter_cli.find_talents([124], mode="mock")
    assert stat.S_IMODE(audit_log.stat().st_mode) == 0o600
    assert len(read_audit(audit_log)) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_a_shared_dotenv_is_reported_by_mode_bits_only(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text("VIRA_API_KEY=value-that-must-not-be-printed\n")
    path.chmod(0o600)
    recruiter_cli._warn_if_shared(path)
    assert capsys.readouterr().err == ""
    path.chmod(0o644)
    recruiter_cli._warn_if_shared(path)
    err = capsys.readouterr().err
    assert "chmod 600" in err and "value-that-must-not-be-printed" not in err


# --- real mode: what may leave the machine ----------------------------------------
class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self.ok, self._payload, self.text = status, status < 400, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


@pytest.fixture
def session(monkeypatch):
    """Stand-in for requests.Session: records how a real-mode call is configured."""
    import requests

    class FakeSession:
        made: list = []
        response = FakeResponse(200, {"ok": True})

        def __init__(self):
            self.trust_env, self.verify, self.posts = True, True, []
            FakeSession.made.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, **kwargs):
            self.posts.append((url, kwargs))
            return FakeSession.response

    monkeypatch.setattr(requests, "Session", FakeSession)
    monkeypatch.setattr(recruiter_cli, "VIRA_BASE_URL", "https://vira.example.test/v1")
    for header, value in (("x-api-key", "k"), ("x-client-name", "c"), ("x-user-id", "u")):
        monkeypatch.setitem(recruiter_cli.HEADERS, header, value)
    return FakeSession


def test_real_calls_ignore_proxy_env_and_follow_no_redirects(session):
    assert recruiter_cli.find_talents([123], mode="real")["status"] == "ok"
    [made] = session.made
    [(url, kwargs)] = made.posts
    assert url == "https://vira.example.test/v1/fast_retargeting"
    assert made.trust_env is False and made.verify is True
    assert kwargs["allow_redirects"] is False and kwargs["timeout"] == 30
    session.response = FakeResponse(307)
    result = recruiter_cli.find_talents([124], mode="real")
    assert result["status"] == "error" and result["http_status"] == 307


@pytest.mark.parametrize("url,allowed", [
    ("https://vira.example.test/v1", True), ("http://localhost:8080/v1", True),
    ("http://127.0.0.1:8080/v1", True), ("http://[::1]:8080/v1", True),
    ("http://vira.example.test/v1", False), ("ftp://vira.example.test/v1", False),
    ("vira.example.test/v1", False),
])
def test_plain_http_only_to_loopback(session, monkeypatch, url, allowed):
    monkeypatch.setattr(recruiter_cli, "VIRA_BASE_URL", url)
    result = recruiter_cli.find_talents([123], mode="real")
    assert (result["status"] == "ok") is allowed and bool(session.made) is allowed
    assert "vira.example.test" not in json.dumps(result)


@pytest.fixture
def engine(session, monkeypatch):
    monkeypatch.setattr(recruiter_cli, "VIRA_ACTUAL_LOCATION", "https://engine.example.test/agent/task-group")
    monkeypatch.setattr(recruiter_cli, "VIRA_XRTOKEN", "xr-test")
    return session


def task_group(**kwargs):
    return recruiter_cli.execute("probe", recruiter_cli.TASK_GROUP_PATH, {}, {"tasks": [1]},
                                 mode="real", **kwargs)


def test_task_groups_go_to_the_engine_with_only_the_users_xrtoken(engine):
    assert task_group()["status"] == "ok"
    recruiter_cli.find_talents([123], mode="real")
    (url, kwargs), = engine.made[0].posts
    assert url == "https://engine.example.test/agent/task-group"
    assert kwargs["headers"] == {"xrtoken": "xr-test", "Content-Type": "application/json"}
    assert kwargs["allow_redirects"] is False and engine.made[0].trust_env is False
    (url, kwargs), = engine.made[1].posts            # the other endpoints keep their own auth
    assert url == "https://vira.example.test/v1/fast_retargeting"
    assert "xrtoken" not in kwargs["headers"] and kwargs["headers"]["x-api-key"] == "k"


def test_a_relative_engine_location_is_under_vira_base_url(engine, monkeypatch):
    monkeypatch.setattr(recruiter_cli, "VIRA_ACTUAL_LOCATION", "/agent/task-group")
    task_group()
    (url, _), = engine.made[0].posts
    assert url == "https://vira.example.test/v1/agent/task-group"


@pytest.mark.parametrize("setting,value,named", [
    ("VIRA_ACTUAL_LOCATION", "", "VIRA_ACTUAL_LOCATION"),
    ("VIRA_ACTUAL_LOCATION", "http://engine.example.test/agent", "VIRA_ACTUAL_LOCATION must be https"),
    ("VIRA_XRTOKEN", "", "VIRA_XRTOKEN"),
])
def test_the_engine_fails_closed_without_its_settings(engine, monkeypatch, setting, value, named):
    monkeypatch.setattr(recruiter_cli, setting, value)
    result = task_group()
    assert result["status"] == "error" and named in result["result"]["message"]
    assert engine.made == [] and "engine.example.test" not in json.dumps(result)


def test_missing_credentials_fail_closed(session, monkeypatch):
    monkeypatch.setitem(recruiter_cli.HEADERS, "x-api-key", "")
    result = recruiter_cli.find_talents([123], mode="real")
    assert result["status"] == "error" and "VIRA_API_KEY" in result["result"]["message"]
    assert session.made == []


def test_non_json_replies_are_capped(session):
    session.response = FakeResponse(502, None, "x" * 10_000)
    result = recruiter_cli.find_talents([123], mode="real")
    assert result["status"] == "error" and len(result["result"]["raw"]) == recruiter_cli.RAW_LIMIT


def test_non_json_reply_text_is_scrubbed(session):
    session.response = FakeResponse(500, None, "no profile for jane@example.com, +65 9123 4567")
    raw = recruiter_cli.find_talents([123], mode="real")["result"]["raw"]
    assert raw == "no profile for <redacted-email>, <redacted-phone>"
