"""The demo's own routes on the LangGraph server (langgraph.json "http.app"): the data panel.

    GET  /demo          the panel: jobs, their teams and applications, and what reached VIRA
    GET  /demo/state    the same as JSON (mock_jeni.State.snapshot)
    POST /demo/reset    back to the mock's fixtures
    GET  /demo/prompts  each act's first prompt (demo/script.py), for the chat UI's starters

It shows the mock's kept State, the one demo/jeni_graph.py's agent changes, and nothing
else: no real data can reach it.  The endpoints are plain functions, so Starlette runs them
on its thread pool: the State's lock is never waited on in the event loop.
"""
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

import jeni_tools
import mock_jeni
from demo import script

PANEL = (Path(__file__).parent / "panel.html").read_text(encoding="utf-8")
NO_STORE = {"Cache-Control": "no-store"}


def panel(request):
    return HTMLResponse(PANEL, headers=NO_STORE)


def state(request):
    snap = mock_jeni.remember_changes().snapshot()
    for entry in snap["activity"]:
        entry["kind"] = "read" if entry["task"] in jeni_tools.READ_ONLY else "write"
    return JSONResponse(snap, headers=NO_STORE)


def reset(request):
    mock_jeni.remember_changes().reset()
    return JSONResponse({"status": "reset"}, headers=NO_STORE)


def prompts(request):
    return JSONResponse(script.prompts(), headers=NO_STORE)


app = Starlette(routes=[Route("/demo", panel), Route("/demo/state", state),
                        Route("/demo/reset", reset, methods=["POST"]),
                        Route("/demo/prompts", prompts)])
