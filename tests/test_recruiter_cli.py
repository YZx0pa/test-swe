"""recruiter_cli: the CLI and the typed actions share one guarded path."""
import json

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
