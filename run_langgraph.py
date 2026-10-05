#!/usr/bin/env python3
"""VIRA agent on LangGraph, via LangChain's create_agent (a tool-calling loop
compiled to a LangGraph graph: model node <-> tools node, state in a checkpointer).

    python run_langgraph.py                         # interactive, mock VIRA
    python run_langgraph.py --task "Find talents for job 123"
    python run_langgraph.py --tools jeni --task "Assign job 7001 to Bob as a team member"

  jeni_db (the default) = jeni tools PLUS read-only db lookup/validation tools, so the
  agent can resolve a job by title and validate ids before acting.  The db follows
  --mode, like VIRA: mock answers from mock VIRA's synthetic data, so the ids agree;
  real uses Postgres at $TRON_POSTGRES_DSN (or --dsn):

    python run_langgraph.py --task "add Kubernetes to the backend engineer job"
    python run_langgraph.py --mode real        # every write pauses for approval
"""
import asyncio

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

import agent_kit

# Toolsets whose tools are async (db handlers) -> the graph must be driven with ainvoke.
ASYNC_TOOLSETS = {"db", "jeni_db"}


def build_agent(*, approve_all: bool = False, step_limit: int = 12, model=None,
                toolset: agent_kit.Toolset = agent_kit.VIRA, gate_writes: bool = False,
                own_checkpointer: bool = True, query_tools=None, context=None,
                task_memory: bool = False):
    """gate_writes: every write pauses for approval, as in real mode, whatever the mode (the
    demo).  own_checkpointer=False: a server keeps the threads (langgraph dev), and it
    refuses a graph that brings its own checkpointer."""
    gated = agent_kit.interrupt_on(approve_all or gate_writes,
    toolset=toolset)
    approval = [HumanInTheLoopMiddleware(interrupt_on=gated)] if gated else []
    memory = [agent_kit.TaskMemoryMiddleware(toolset.read_only)] if task_memory else []
    agent_kit.stage("MEMORY_MIDDLEWARE_CONFIGURED", enabled=task_memory)
    import entities_db_validation
    entity_pre, entity_execute = entities_db_validation.stages(
        query_tools, context,
        action_names=toolset.names,
        read_only=toolset.read_only,
        pre_approval_names=gated,
    )
    return create_agent(
        model or agent_kit.build_chat_model(),
        toolset.tools(),
        system_prompt=toolset.prompt,
        response_format=agent_kit.TerminalResponse,
        # Pre-card validation is outermost; execution validation is inside HITL so it
        # sees reviewer-edited arguments, then ToolCallGuard performs provenance checks.
        middleware=[*memory, *entity_pre, *approval,
                    *agent_kit.middleware(step_limit, toolset=toolset,
                                          execution_validation=(entity_execute[0]
                                                                if entity_execute else None))],
        # needed to pause at an interrupt and resume
        checkpointer=InMemorySaver() if own_checkpointer else None,
        name="vira-langgraph",
    )


def _mock_query_tools(company_id, dsn_ignored: bool):
    """Mock mode: the lookups answer from mock VIRA's own data.  A real database's ids would
    all be "not found" by the Jeni task mock, so a DSN is used in real mode only."""
    import mock_jeni
    from db_queries import fake_db_queries
    note = "; the DSN is used with --mode real only" if dsn_ignored else ""
    print(f"(mock mode: db lookups use mock VIRA's synthetic jobs, users and applications{note})")
    return fake_db_queries(mock_jeni.db_fixtures(company_id))


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
    if args.mode == "real":
        if not args.dsn:
            raise SystemExit(f"--mode real --tools {args.tools} needs a Postgres DSN: set "
                             f"TRON_POSTGRES_DSN in .env or pass --dsn.")
        from db_queries import build_db_queries
        print(f"(connecting to Postgres for --tools {args.tools} ...)")
        pool = await _open_pool(args.dsn)
        print("(db connected)")
        query_tools = build_db_queries(pool)
    else:
        query_tools = _mock_query_tools(args.company_id, dsn_ignored=bool(args.dsn))
    try:
        toolset = agent_kit.cli_toolset(args, query_tools=query_tools, context=context)
        agent = build_agent(approve_all=args.approve_all, step_limit=args.step_limit,
                            toolset=toolset, query_tools=query_tools, context=context,
                            task_memory=args.task_memory)
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
                        toolset=agent_kit.cli_toolset(args), task_memory=args.task_memory)
    agent_kit.repl("LangGraph", agent, args)


if __name__ == "__main__":
    main()
