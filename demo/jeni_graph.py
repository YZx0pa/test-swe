"""Jeni v2 for the demo: the LangGraph agent that `langgraph dev` serves (langgraph.json).

    demo/run.sh                                     # the server and the chat UI together
    .venv/bin/langgraph dev --no-browser --no-reload  # the server alone: 127.0.0.1:2024, graph "jeni"

Mock VIRA and the mock database only.  The mock remembers changes
(mock_jeni.remember_changes), so a change shows when it is read back and in the data panel
(demo/app.py).  Routine writes run as soon as the agent calls them; the high-stakes ones in
agent_kit.ALWAYS_CONFIRM (shortlist, reject, share, transfer ownership) wait for an approval
card, as they do in every mode.  GATE_WRITES = True puts every write behind a card, as real
mode does.  The server keeps the threads, so the graph has no checkpointer of its own.  The
mode is fixed to mock before any tool exists, so nothing here reaches VIRA or a real
database.
"""
import asyncio
import dataclasses

import agent_kit
import mock_jeni
import run_langgraph
import vira_tools
from db_queries import fake_db_queries

COMPANY_ID = 5143        # the demo's tenant, bound into the db tools; never from the model
# False: only ALWAYS_CONFIRM tools ask.  True: every write asks, as in real mode.
GATE_WRITES = False

# A chat window shows the reply as written; the SUMMARY: a | b | c format is for grading.
SUMMARY_RULE = """\
- When you are done, reply in plain text without calling a tool. If the task is only
  partially done or cannot be fully completed, start that reply with
  SUMMARY: <what succeeded> | <what failed or is missing> | <why>
"""
CHAT_RULE = """\
- When you are done, reply without calling a tool, briefly, for a chat window: the outcome
  first, in a sentence or two, and a short markdown list only for several items. Don't recount
  the tools you called. If the task is only partially done or cannot be fully completed, say
  what succeeded, what failed or is missing, and why.
"""
# The terminal REPL reads a STATUS line to remember finished tasks (agent_kit.read_status); the
# server never does, and in a chat window it shows as a stray "STATUS: done".
STATUS_RULE = """\
- End EVERY reply that has no tool call with a status line as its LAST line, one of:
    STATUS: done                 - the task is finished (succeeded or cannot proceed)
    STATUS: needs_user: <what>   - you must get something from the user to continue
  Use needs_user only when you are genuinely blocked on the user (e.g. a value no tool
  can supply). Otherwise, if more tool calls are needed, make them instead of replying.
"""
NO_STATUS_RULE = """\
- If more tool calls are needed, make them instead of replying.
"""
CHAT_REPLACEMENTS = ((SUMMARY_RULE, CHAT_RULE), (STATUS_RULE, NO_STATUS_RULE))


def demo_toolset(state: mock_jeni.State) -> agent_kit.Toolset:
    """jeni_db on the kept mock State, with the closing rules written for a chat window."""
    toolset = agent_kit.toolset("jeni_db",
                                query_tools=fake_db_queries(state.db_fixtures(COMPANY_ID)),
                                context={"auth_profile": {"company_id": COMPANY_ID}})
    prompt = toolset.prompt
    for rule, replacement in CHAT_REPLACEMENTS:
        if rule not in prompt:
            raise RuntimeError("a closing rule of agent_kit.SYSTEM_PROMPT changed: update "
                               "CHAT_REPLACEMENTS in demo/jeni_graph.py to match")
        prompt = prompt.replace(rule, replacement)
    return dataclasses.replace(toolset, prompt=prompt)


def build(model=None, gate_writes: bool = GATE_WRITES):
    """The demo agent: mock mode, changes kept, no checkpointer; every write gated only with
    gate_writes.  `model` is for tests; by default it is $CHAT_MODEL, as in the runners."""
    agent_kit.set_tracing(False)
    vira_tools.configure("mock")
    return run_langgraph.build_agent(model=model, toolset=demo_toolset(mock_jeni.remember_changes()),
                                     gate_writes=gate_writes, own_checkpointer=False)


_graph = None
_building = asyncio.Lock()


async def make_graph():
    """langgraph.json's factory, called for every request: builds once, off the event loop
    (reading the catalog there would block it)."""
    global _graph
    async with _building:
        if _graph is None:
            _graph = await asyncio.to_thread(build)
    return _graph
