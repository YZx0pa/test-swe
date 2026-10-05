"""Hermetic test setup: no .env, no real VIRA, no real audit log, no real LLM.

The environment is fixed at module top, before any project module is imported,
because recruiter_cli calls load_dotenv() and reads EVENTS_LOG / the VIRA
headers at import time.  Tests must never import run_mini (it reads .env and
pulls in litellm); mini_env is importable, with minisweagent kept quiet below.
"""
import json
import os
import tempfile

os.environ["PYTHON_DOTENV_DISABLED"] = "1"              # every load_dotenv() is a no-op
os.environ["MSWEA_SILENT_STARTUP"] = "1"                # mini_env imports minisweagent: no banner
os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = tempfile.mkdtemp(prefix="mswea-")   # not ~/.config
os.environ["JENI_MODE"] = "mock"
os.environ["VIRA_BASE_URL"] = "http://127.0.0.1:9/v1"    # dead port: real mode can't reach VIRA
for _key in ("VIRA_API_KEY", "VIRA_CLIENT_NAME", "VIRA_USER_ID", "VIRA_ACTUAL_LOCATION", "VIRA_XRTOKEN"):
    os.environ[_key] = ""
os.environ["EVENTS_LOG"] = os.devnull                   # the fixture below points it at tmp_path
os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
os.environ.pop("TRON_POSTGRES_DSN", None)               # jeni_db never reaches a real database
os.environ.pop("VIRA_RESULT_SOURCE", None)              # nor does reading a task group back
os.environ.pop("VIRA_RESULT_LOCATION", None)
os.environ.pop("cutqueque", None)                       # nor asks the engine to run one
os.environ.pop("VIRA_result_trigger", None)
os.environ["JENI_STAGE_LOG"] = ""                       # the runners' setup() writes no logs/ file
# Jeni's real task catalog is internal (config/README.md): tests use a synthetic one.
os.environ["JENI_TASKS_FILE"] = os.path.join(os.path.dirname(__file__), "fixtures", "jeni_tasks.json")
for _key in ("LANGSMITH_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_key] = "false"

import pytest  # noqa: E402

import recruiter_cli  # noqa: E402
import vira_tools  # noqa: E402


@pytest.fixture(autouse=True)
def audit_log(tmp_path, monkeypatch):
    """Every test gets its own audit file; the real events.jsonl is never touched."""
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(recruiter_cli, "EVENTS_LOG", str(path))
    return path


@pytest.fixture(autouse=True)
def vira_mode():
    """The VIRA mode back as it was after each test: a runner's main() sets it from --mode,
    whose default is real."""
    mode = vira_tools.current_mode()
    yield
    vira_tools.configure(mode)


def done(result: dict) -> bool:
    """A projected Jeni result (jeni_tools.project) that was sent and whose sub-tasks all completed."""
    subs = result.get("subtasks") or []
    return result.get("request_status") == "ok" and bool(subs) and all(
        s.get("status") == "completed" for s in subs)


def reply(result: dict, i: int = 0):
    """The i-th sub-task's result in a projected Jeni result."""
    return result["subtasks"][i].get("result")


def read_audit(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
