#!/usr/bin/env python3
"""VIRA as an MCP server (stdio): the bridge to every other agent framework.

Any MCP client (Claude Code / Claude Desktop, the OpenAI Agents SDK, PydanticAI,
LangGraph via langchain-mcp-adapters, ...) gets the same four typed tools as
run_langgraph.py (vira_tools.py) and the same guarded path underneath: confirm
gate, input limits, audit log, PII masking.  Credentials stay in this process: it
loads .env itself (through recruiter_cli), so no client config ever carries a VIRA key.

There is no ToolCallGuard here (a server can't see the client's conversation), so:
  * in --mode real, only the read-only tools are listed unless --allow-side-effects
    is given; then approving each score/insights call is up to the client;
  * the process makes at most --max-calls VIRA calls (default 50), then refuses.

    claude mcp add vira -- /abs/path/.venv/bin/python /abs/path/vira_mcp.py --mode mock

stdout is the JSON-RPC channel: nothing in this process may print to it.
"""
import argparse
import functools
import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

import recruiter_cli
import vira_tools

HERE = Path(__file__).parent
MAX_CALLS = 50


class CallBudget:
    """VIRA calls this server process may still make."""

    def __init__(self, limit: int) -> None:
        self.limit, self._used, self._lock = limit, 0, threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self._used >= self.limit:
                return False
            self._used += 1
            return True


def _budgeted(fn, budget: CallBudget):
    @functools.wraps(fn)                  # FastMCP builds the schema from fn's signature
    def tool(*args, **kwargs):
        if not budget.take():
            return {"status": "error", "message": f"Refused: this server's budget of "
                    f"{budget.limit} VIRA calls is used up."}
        return fn(*args, **kwargs)
    return tool


def build_server(*, allow_side_effects: bool = False, max_calls: int = MAX_CALLS) -> FastMCP:
    server = FastMCP("vira")
    budget = CallBudget(max_calls)
    real = vira_tools.current_mode() == "real"
    for fn in vira_tools.TOOLS:
        name = fn.__name__
        read_only = name in vira_tools.READ_ONLY
        if name.replace("_", "-") in recruiter_cli.NEEDS_CONFIRM:
            # Approval over MCP is up to the client (annotations are only hints and
            # elicitation needs client support), so confirm-gated tools stay off MCP.
            continue
        if real and not read_only and not allow_side_effects:
            continue                        # score/insights change data on VIRA
        server.add_tool(
            _budgeted(fn, budget), description=vira_tools.description(fn),
            structured_output=False,        # plain JSON text: every client sees the same result
            annotations=ToolAnnotations(readOnlyHint=read_only,
                                        destructiveHint=not read_only,   # recalculation overwrites
                                        idempotentHint=True, openWorldHint=True))
    return server


def main(argv=None):
    p = argparse.ArgumentParser(description="VIRA tools over MCP (stdio).")
    p.add_argument("--mode", choices=["real", "mock"], required=True,
                   help="mock: local fake VIRA; real: call VIRA at $VIRA_BASE_URL")
    p.add_argument("--allow-side-effects", action="store_true",
                   help="real mode: also list score/insights, which trigger calculations on "
                        "VIRA; the client must then ask before each call")
    p.add_argument("--max-calls", type=int, default=MAX_CALLS,
                   help=f"VIRA calls this process makes before refusing (default {MAX_CALLS})")
    args = p.parse_args(argv)
    vira_tools.configure(args.mode)
    os.chdir(HERE)          # a relative EVENTS_LOG lands where the CLI writes it
    build_server(allow_side_effects=args.allow_side_effects,
                 max_calls=args.max_calls).run()    # stdio transport


if __name__ == "__main__":
    main()
