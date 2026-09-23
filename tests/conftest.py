"""Hermetic test setup: no .env, no real VIRA, no real audit log, no real LLM.

The environment is fixed at module top, before any project module is imported,
because recruiter_cli calls load_dotenv() and reads EVENTS_LOG / the VIRA
headers at import time.  Tests must never import run_mini (it loads .env and
minisweagent prints a banner).
"""
import json
import os

os.environ["PYTHON_DOTENV_DISABLED"] = "1"              # every load_dotenv() is a no-op
os.environ["JENI_MODE"] = "mock"
os.environ["VIRA_BASE_URL"] = "http://127.0.0.1:9/v1"    # dead port: real mode can't reach VIRA
for _key in ("VIRA_API_KEY", "VIRA_CLIENT_NAME", "VIRA_USER_ID"):
    os.environ[_key] = ""
os.environ["EVENTS_LOG"] = os.devnull                   # the fixture below points it at tmp_path
os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest  # noqa: E402

import recruiter_cli  # noqa: E402


@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    """Every test gets its own audit file; the real events.jsonl is never touched."""
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(recruiter_cli, "EVENTS_LOG", str(path))
    return path


def read_audit(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
