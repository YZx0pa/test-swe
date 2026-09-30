#!/usr/bin/env python3
"""VIRA agent on LangGraph, via LangChain's create_agent (a tool-calling loop
compiled to a LangGraph graph: model node <-> tools node, state in a checkpointer).

Same job as run_mini.py, different shape:
  * the model gets the 4 typed VIRA tools (vira_tools.py), not a bash shell,
    so it can't read files or env vars and never picks real vs mock;
  * the loop ends when the model answers in prose;
  * middleware adds a model-call cap, duplicate-call refusal and, with
    --approve-all, a human approval step (interrupt) before every VIRA call.

    python run_langgraph.py                         # interactive, mock VIRA
    python run_langgraph.py --task "Find talents for job 123"
    python run_langgraph.py --approve-all --mode real
    python run_langgraph.py --tools jeni --task "Assign job 7001 to Bob as a team member"
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

import agent_kit


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


def main(argv=None):
    args = agent_kit.parser("VIRA agent on LangGraph (create_agent).").parse_args(argv)
    agent_kit.setup(args)
    agent = build_agent(approve_all=args.approve_all, step_limit=args.step_limit,
                        toolset=agent_kit.cli_toolset(args))
    agent_kit.repl("LangGraph", agent, args)


if __name__ == "__main__":
    main()
