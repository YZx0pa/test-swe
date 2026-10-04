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

import agent_kit
import mock_jeni
import pii_vault
import run_langgraph
import vira_tools
from db_queries import fake_db_queries

COMPANY_ID = 5143        # the demo's tenant, bound into the db tools; never from the model
USER_EMAIL = "sam.lee@example.com"   # the signed-in user, for "me": Sam Lee (804) in the mock
# False: only ALWAYS_CONFIRM tools ask.  True: every write asks, as in real mode.
GATE_WRITES = False

def demo_toolset(state: mock_jeni.State, query_tools=None) -> agent_kit.Toolset:
    """jeni_db on the kept mock State."""
    query_tools = query_tools or fake_db_queries(state.db_fixtures(COMPANY_ID))
    return agent_kit.toolset(
        "jeni_db",
        query_tools=query_tools,
        context={"auth_profile": {"company_id": COMPANY_ID}},
    )


def build(model=None, gate_writes: bool = GATE_WRITES):
    """The demo agent: mock mode, changes kept, no checkpointer; every write gated only with
    gate_writes.  `model` is for tests; by default it is $CHAT_MODEL, as in the runners."""
    agent_kit.set_tracing(False)
    vira_tools.configure("mock")
    pii_vault.VAULT.user_email = USER_EMAIL
    state = mock_jeni.remember_changes()
    context = {"auth_profile": {"company_id": COMPANY_ID}}
    query_tools = fake_db_queries(state.db_fixtures(COMPANY_ID))
    return run_langgraph.build_agent(model=model, toolset=demo_toolset(state, query_tools),
                                     gate_writes=gate_writes, own_checkpointer=False,
                                     query_tools=query_tools, context=context,
                                     task_memory=True)


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
