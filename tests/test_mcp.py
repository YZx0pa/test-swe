"""vira_mcp: the typed VIRA tools over MCP, masked, with stdout reserved for JSON-RPC."""
import asyncio
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import mcp.types
from mcp.shared.memory import create_connected_server_and_client_session

import recruiter_cli
import vira_mcp
import vira_tools
from conftest import read_audit

vira_tools.configure("mock")
ROOT = Path(__file__).resolve().parent.parent


def session_run(server, scenario):
    async def main():
        async with create_connected_server_and_client_session(server) as client:
            return await scenario(client)
    return asyncio.run(main())


def test_tools_annotations_and_masked_results(monkeypatch, audit_log):
    monkeypatch.setattr(recruiter_cli, "_call", lambda path, q, b, mode: {
        "status": "ok", "http_status": 200,
        "result": {"name": "Jane Doe", "email": "jane@example.com", "profile_id": 900001}})

    async def scenario(client):
        tools = {t.name: t for t in (await client.list_tools()).tools}
        return tools, await client.call_tool("find_talents", {"job_ids": [123]})

    tools, result = session_run(vira_mcp.build_server(), scenario)
    assert set(tools) == vira_tools.NAMES
    assert tools["find_talents"].description == vira_tools.description(vira_tools.find_talents)
    assert tools["find_talents"].annotations.readOnlyHint is True
    assert tools["score_candidates"].annotations.readOnlyHint is False     # triggers scoring
    assert tools["score_candidates"].annotations.idempotentHint is True
    assert tools["score_candidates"].annotations.destructiveHint is True    # overwrites scores
    assert tools["find_talents"].annotations.destructiveHint is False
    assert tools["find_talents"].inputSchema["required"] == ["job_ids"]

    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["result"] == {"name": "<redacted>", "email": "<redacted>", "profile_id": 900001}
    assert [a["command"] for a in read_audit(audit_log)] == ["find-talents"]


def test_invalid_arguments_are_an_mcp_error(audit_log):
    async def scenario(client):
        return await client.call_tool("find_talents", {"job_ids": []})

    result = session_run(vira_mcp.build_server(), scenario)
    assert result.isError is True
    assert read_audit(audit_log) == []


def test_confirm_gated_tools_are_not_exposed(monkeypatch):
    monkeypatch.setattr(recruiter_cli, "NEEDS_CONFIRM", {"score-candidates"})

    async def scenario(client):
        return {t.name for t in (await client.list_tools()).tools}

    assert "score_candidates" not in session_run(vira_mcp.build_server(), scenario)


def tool_names(server):
    async def scenario(client):
        return {t.name for t in (await client.list_tools()).tools}
    return session_run(server, scenario)


def test_real_mode_lists_only_reads_unless_side_effects_are_allowed(monkeypatch):
    monkeypatch.setattr(vira_tools, "_MODE", "real")
    assert tool_names(vira_mcp.build_server()) == vira_tools.READ_ONLY
    assert tool_names(vira_mcp.build_server(allow_side_effects=True)) == vira_tools.NAMES


def test_the_server_stops_calling_vira_after_its_budget(audit_log):
    async def scenario(client):
        return [await client.call_tool("find_talents", {"job_ids": [job]}) for job in (1, 2, 3)]

    results = session_run(vira_mcp.build_server(max_calls=2), scenario)
    payloads = [json.loads(r.content[0].text) for r in results]
    assert [p["status"] for p in payloads] == ["ok", "ok", "error"]
    assert "budget of 2 VIRA calls" in payloads[2]["message"]
    assert [a["body"]["job_ids"] for a in read_audit(audit_log)] == [[1], [2]]


# --- the real process over stdio ---------------------------------------------
def _read_line(proc, timeout=30) -> str:
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    assert ready, "MCP server did not answer in time"
    return proc.stdout.readline()


def test_stdio_server_speaks_only_json_rpc(tmp_path):
    audit = tmp_path / "stdio-events.jsonl"
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path)),
           "PYTHON_DOTENV_DISABLED": "1", "EVENTS_LOG": str(audit)}    # explicit, minimal
    proc = subprocess.Popen([sys.executable, str(ROOT / "vira_mcp.py"), "--mode", "mock"],
                            cwd=tmp_path, env=env, text=True, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = []

    def request(msg_id, method, params=None):
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method,
                                     "params": params or {}}) + "\n")
        proc.stdin.flush()
        lines.append(_read_line(proc))
        return json.loads(lines[-1])

    try:
        init = request(1, "initialize", {
            "protocolVersion": mcp.types.LATEST_PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "vira"
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        listed = request(2, "tools/list")
        assert {t["name"] for t in listed["result"]["tools"]} == vira_tools.NAMES
        called = request(3, "tools/call", {"name": "find_talents",
                                           "arguments": {"job_ids": [123]}})
        payload = json.loads(called["result"]["content"][0]["text"])
        assert payload["status"] == "ok" and payload["result"]["job_id"] == 123
    finally:
        rest, _ = proc.communicate(timeout=30)     # closes stdin: the server exits on EOF
    lines += rest.splitlines()
    for line in filter(str.strip, lines):          # nothing but JSON-RPC on stdout
        assert json.loads(line)["jsonrpc"] == "2.0"
    assert [a["command"] for a in read_audit(audit)] == ["find-talents"]
    assert list(tmp_path.glob("events.jsonl")) == []
