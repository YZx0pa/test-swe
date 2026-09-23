#!/usr/bin/env python3
"""VIRA agent on deepagents: LangGraph + planning, a virtual filesystem and subagents.

deepagents wraps LangChain's create_agent in a harness for longer, multi-step work:
  * write_todos (TodoListMiddleware): the agent plans and tracks its steps;
  * a virtual filesystem (ls / read_file / write_file / edit_file / ...) kept in
    the graph state, for notes and reports.  It is NOT the host disk;
  * subagents behind a `task` tool: each runs in its own context and hands back
    only its final answer, so per-job sourcing doesn't flood the main context.

Security: the backend is deepagents' default StateBackend (in memory, per
thread).  Never pass FilesystemBackend or LocalShellBackend here: they read and
write the host disk / run shell commands, and this directory holds .env.

    python run_deepagent.py --task "For jobs 101 and 102, find suggested talents and \
write a short report to /report.md"
"""
from deepagents import create_deep_agent
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver

import agent_kit
import vira_tools

WORKING_STYLE = """
Working style:
- For a task with several steps or several jobs, plan it first with write_todos and keep
  the list up to date.
- Delegate with the task tool: sourcing-analyst to find or score talents and get candidate
  insights (one job per call; independent jobs can run in parallel), jd-writer for job
  descriptions. Pass every id and value a subagent needs: it cannot see this conversation.
- Files (write_file etc.) live in a scratch space inside this conversation, not on a real
  disk. Write one only when the user asks for a file or report, and give its path in your
  final answer.
"""

SOURCING_PROMPT = """You source and score talent through the VIRA tools, for exactly the job and
ids you are given. Answer compactly: the ids you found or scored with their key scores, and any
failure.
"""

JD_PROMPT = """You draft job descriptions with the generate_jd tool, using only the title, skills,
language and details you are given. Answer with the job description text.
"""


def subagents(tools: dict, step_limit: int) -> list[dict]:
    # Custom middleware is not inherited by subagents, so each gets its own guard + cap.
    return [
        {"name": "sourcing-analyst",
         "description": ("Finds suggested talents for ONE job, scores applicants or suggested "
                         "talents, and gets candidate insights. Give it the job id and any "
                         "application or match ids."),
         "system_prompt": SOURCING_PROMPT + agent_kit.SYSTEM_PROMPT,
         "tools": [tools[n] for n in ("find_talents", "score_candidates", "candidate_insights")],
         "middleware": agent_kit.middleware(step_limit)},
        {"name": "jd-writer",
         "description": ("Drafts a job description. Give it the job title, skills, language "
                         "and any other details from the user."),
         "system_prompt": JD_PROMPT + agent_kit.SYSTEM_PROMPT,
         "tools": [tools["generate_jd"]],
         "middleware": agent_kit.middleware(step_limit)},
        # Replaces the auto-added general-purpose subagent, which would otherwise run
        # the VIRA tools without our guard, call cap or domain rules.
        {**GENERAL_PURPOSE_SUBAGENT,
         "system_prompt": GENERAL_PURPOSE_SUBAGENT["system_prompt"] + "\n\n" + agent_kit.SYSTEM_PROMPT,
         "middleware": agent_kit.middleware(step_limit)},
    ]


def build_agent(*, approve_all: bool = False, step_limit: int = 12, model=None):
    tools = vira_tools.langchain_tools()
    return create_deep_agent(
        model=model or agent_kit.build_chat_model(),
        tools=tools,
        system_prompt=agent_kit.SYSTEM_PROMPT + WORKING_STYLE,
        subagents=subagents({t.name: t for t in tools}, step_limit),
        middleware=[TodoListMiddleware(), *agent_kit.middleware(step_limit)],
        interrupt_on=agent_kit.interrupt_on(approve_all) or None,   # subagents inherit it
        checkpointer=InMemorySaver(),
        name="vira-deepagent",
    )


def show_files(result: dict) -> None:
    """Print whatever the agent wrote to its virtual filesystem."""
    for path, data in sorted((result.get("files") or {}).items()):
        print(f"\n=== virtual file {path} ===")
        print(data.get("content", "") if isinstance(data, dict) else data)


def main(argv=None):
    args = agent_kit.parser("VIRA agent on deepagents.").parse_args(argv)
    agent_kit.setup(args)
    agent = build_agent(approve_all=args.approve_all, step_limit=args.step_limit)
    agent_kit.repl("deepagents", agent, args, after=show_files)


if __name__ == "__main__":
    main()
