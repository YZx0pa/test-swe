"""Completed-task memory: list results keep every row's ids, never one row's values."""
import json
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import task_memory

TOKEN_A, TOKEN_B = "<email:aaaaaaaaaaaa>", "<email:bbbbbbbbbbbb>"


def _apps(n, start=674500):
    return [{"appId": start + i, "profileId": 155000 + i, "companyId": 5143, "jobId": 683801,
             "appStatus": "Screening In Progress" if i % 2 else "Shortlisted",
             "briqScore": 80, "summary": "long free text " * 20}
            for i in range(n)]


def _task(tool, args, payload, question="find applicants for job 683801"):
    return [HumanMessage(content=question),
            AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": "c1"}]),
            ToolMessage(content=json.dumps(payload), tool_call_id="c1"),
            AIMessage(content="Found them.")]


def _values(payload, args=None):
    record = task_memory.extract_record(
        _task("get_applications", args or {"job_id": 683801}, payload),
        frozenset({"get_applications"}))
    return record, record["events"][0]["result"]["values"]


def test_list_result_keeps_every_id_not_the_first_row():
    _, values = _values({"applications": _apps(42), "total": 42})
    assert values["app_ids"] == [674500 + i for i in range(42)]
    assert values["profile_ids"] == [155000 + i for i in range(42)]
    assert values["returned"] == 42 and values["total"] == 42
    assert "app_id" not in values and "profile_id" not in values
    # Mixed per-row statuses mean nothing without their row; a shared value is kept once.
    assert "app_status" not in values
    assert values["company_id"] == 5143
    assert "job_id" not in values                      # already shown in the call's args


def test_task_group_envelope_and_snake_case_ids():
    payload = {"request_status": "ok", "subtasks": [{"status": "completed", "result": {
        "applications": [{"app_id": 1, "profile_id": 11}, {"app_id": 2, "profile_id": 12}]}}]}
    _, values = _values(payload)
    assert values["app_ids"] == [1, 2] and values["profile_ids"] == [11, 12]


def test_rows_with_ids_drop_emails():
    rows = [{"appId": 1, "candidateEmail": TOKEN_A}, {"appId": 2, "candidateEmail": TOKEN_B}]
    _, values = _values({"applications": rows})
    assert values["app_ids"] == [1, 2] and "emails" not in values


def test_rows_without_ids_keep_email_tokens_raw_addresses_never():
    rows = [{"name": "<redacted-name>", "email": TOKEN_A},
            {"name": "<redacted-name>", "email": TOKEN_B},
            {"name": "<redacted-name>", "email": "leak@example.com"}]
    _, values = _values({"users": rows})
    assert values["emails"] == [TOKEN_A, TOKEN_B]
    assert "leak@example.com" not in json.dumps(values)


def test_single_record_still_keeps_singular_fields():
    _, values = _values({"appId": 674517, "profileId": 155218, "appStatus": "Shortlisted"},
                        args={"app_id": 674517})
    assert values == {"profile_id": 155218, "app_status": "Shortlisted"}


def test_digest_shows_counts_and_long_id_lists():
    record, _ = _values({"applications": _apps(60), "total": 60})
    record["ts"] = time.time()
    digest = task_memory.memory_digest([record])
    line = next(line for line in digest.splitlines() if "Resolve:" in line)
    assert "-> returned=60, total=60, app_ids=[674500, 674501" in line
    assert "674549, … +10 more]" in line
    assert "long free text" not in digest
