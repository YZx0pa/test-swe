#!/usr/bin/env python3
"""VIRA as an MCP server (stdio): the bridge to every other agent framework.

Any MCP client (Claude Code / Claude Desktop, the OpenAI Agents SDK, PydanticAI,
LangGraph via langchain-mcp-adapters, ...) gets the same four typed tools as
run_langgraph.py (vira_tools.py) and the same guarded path underneath: confirm
gate, audit log, PII masking.  Credentials stay in this process: it loads .env
itself (through recruiter_cli), so no client config ever carries a VIRA key.

    claude mcp add vira -- /abs/path/.venv/bin/python /abs/path/vira_mcp.py --mode mock

stdout is the JSON-RPC channel: nothing in this process may print to it.
"""
import argparse
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

import recruiter_cli
import vira_tools

HERE = Path(__file__).parent


def build_server() -> FastMCP:
    server = FastMCP("vira")
    for fn in vira_tools.TOOLS:
        name = fn.__name__
        if name.replace("_", "-") in recruiter_cli.NEEDS_CONFIRM:
            # Approval over MCP is up to the client (annotations are only hints and
            # elicitation needs client support), so confirm-gated tools stay off MCP.
            continue
        server.add_tool(
            fn, description=vira_tools.description(fn),
            structured_output=False,        # plain JSON text: every client sees the same result
            annotations=ToolAnnotations(readOnlyHint=name in vira_tools.READ_ONLY,
                                        destructiveHint=False, idempotentHint=True,
                                        openWorldHint=True))
    return server


def main(argv=None):
    p = argparse.ArgumentParser(description="VIRA tools over MCP (stdio).")
    p.add_argument("--mode", choices=["real", "mock"], required=True,
                   help="mock: local fake VIRA; real: call VIRA at $VIRA_BASE_URL")
    args = p.parse_args(argv)
    vira_tools.configure(args.mode)
    os.chdir(HERE)          # a relative EVENTS_LOG lands where the CLI writes it
    build_server().run()    # stdio transport


if __name__ == "__main__":
    main()
