#!/usr/bin/env python3
"""VIRA agent on LangGraph, via LangChain's create_agent (a tool-calling loop
compiled to a LangGraph graph: model node <-> tools node, state in a checkpointer).

    python run_langgraph.py                         # interactive, mock VIRA
    python run_langgraph.py --task "Find talents for job 123"
    python run_langgraph.py --tools jeni --task "Assign job 7001 to Bob as a team member"

  jeni_db = jeni tools PLUS read-only db lookup/validation tools, so the agent can
  resolve a job by title and validate ids before acting.  A Postgres DSN
  ($TRON_POSTGRES_DSN or --dsn) uses the real db; with none it falls back to offline
  fake db queries so the toolset still runs:

    TRON_POSTGRES_DSN=postgres://... python run_langgraph.py --tools jeni_db
    python run_langgraph.py --tools jeni_db --task "add python, sql to data scientist job"
"""
import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

import agent_kit

# Toolsets whose tools are async (db handlers) -> the graph must be driven with ainvoke.
ASYNC_TOOLSETS = {"db", "jeni_db"}


def build_agent(*, approve_all: bool = False, step_limit: int = 12, model=None,
                toolset: agent_kit.Toolset = agent_kit.VIRA):
    gated = agent_kit.interrupt_on(approve_all, toolset=toolset)
    approval = [HumanInTheLoopMiddleware(interrupt_on=gated)] if gated else []
    return create_agent(
        model or agent_kit.build_chat_model(),
        toolset.tools(),
        system_prompt=toolset.prompt,
        # HITL first = outermost, so the guard sees a reviewer's edited args
        middleware=[*approval, *agent_kit.middleware(step_limit, toolset=toolset)],
        checkpointer=InMemorySaver(),   # needed to pause at an interrupt and resume
        name="vira-langgraph",
    )


def _fake_query_tools(company_id):
    """Offline db: in-memory fixtures so find_/validate_ tools return something sensible."""
    from db_queries import fake_db_queries
    fixtures = {
        "company_id": company_id,
        "jobs": [{"jobId": 501, "jobName": "Data Scientist", "company_id": company_id},
                 {"jobId": 502, "jobName": "Senior Data Scientist", "company_id": company_id}],
        "users": [{"userId": 9, "firstname": "Bob", "lastname": "Lee",
                   "email": "bob@example.com", "company_id": company_id}],
        "applications": [{"app_id": 11, "job_id": 501}, {"app_id": 12, "job_id": 501}],
    }
    print("(no DSN: using offline fake db queries)")
    return fake_db_queries(fixtures)


async def _open_pool(dsn):
    """Open the asyncpg pool and prove it works, with clear errors instead of tracebacks."""
    try:
        import asyncpg
    except ModuleNotFoundError:
        raise SystemExit("jeni_db real mode needs asyncpg: pip install asyncpg "
                         "(into the venv you launch with).")
    try:
        pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10, command_timeout=30)
    except Exception as exc:                     # bad DSN, host down, auth, TLS, ...
        raise SystemExit(f"could not connect to Postgres with the given DSN "
                         f"({type(exc).__name__}: {exc}). Check TRON_POSTGRES_DSN / --dsn.")
    try:                                         # fail fast now, not on the first query
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1;")
    except Exception as exc:
        await pool.close()
        raise SystemExit(f"connected but a test query failed ({type(exc).__name__}: {exc}).")
    return pool


async def amain(args):
    """Async entry for db/jeni_db: ONE event loop owns the asyncpg pool and every ainvoke.
    asyncpg binds a pool to the loop it was created on, so the pool is created here."""
    agent_kit.setup(args)
    context = {"auth_profile": {"company_id": args.company_id}}
    pool = None
    if args.dsn:
        from db_queries import build_db_queries
        print(f"(connecting to Postgres for --tools {args.tools} ...)")
        pool = await _open_pool(args.dsn)
        print("(db connected)")
        query_tools = build_db_queries(pool)
    else:
        query_tools = _fake_query_tools(args.company_id)
    try:
        toolset = agent_kit.cli_toolset(args, query_tools=query_tools, context=context)
        agent = build_agent(approve_all=args.approve_all, step_limit=args.step_limit,
                            toolset=toolset)
        await agent_kit.arepl("LangGraph", agent, args, toolset)
    finally:
        if pool is not None:
            await pool.close()


def main(argv=None):
    args = agent_kit.parser("VIRA agent on LangGraph (create_agent).").parse_args(argv)
    if args.tools in ASYNC_TOOLSETS:
        asyncio.run(amain(args))            # db tools are async: everything on one loop
        return
    agent_kit.setup(args)
    agent = build_agent(approve_all=args.approve_all, step_limit=args.step_limit,
                        toolset=agent_kit.cli_toolset(args))
    agent_kit.repl("LangGraph", agent, args)


if __name__ == "__main__":
    main()
