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
from terminal import printable

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

JENI_WORKING_STYLE = """
Working style:
- For a request with several steps or several jobs, plan it first with write_todos and keep
  the list up to date.
- Independent per-job work can be delegated with the task tool (general-purpose). Pass every
  id and value it needs: it cannot see this conversation.
- Files (write_file etc.) live in a scratch space inside this conversation, not on a real
  disk. Write one only when the user asks for a file or report.
"""

SOURCING_PROMPT = """You source and score talent through the VIRA tools, for exactly the job and
ids you are given. Answer compactly: the ids you found or scored with their key scores, and any
failure.
"""

JD_PROMPT = """You draft job descriptions with the generate_jd tool, using only the title, skills,
language and details you are given. Answer with the job description text.
"""


def subagents(tools: dict, step_limit: int, ledger: agent_kit.CallLedger) -> list[dict]:
    # Custom middleware is not inherited by subagents, so each gets its own guard + cap,
    # all sharing one ledger: a subagent can't repeat a VIRA call the parent already made.
    return [
        {"name": "sourcing-analyst",
         "description": ("Finds suggested talents for ONE job, looks up their match ids, scores "
                         "applicants or suggested talents, and gets candidate insights. Give it "
                         "the job id and any application, profile or match ids."),
         "system_prompt": SOURCING_PROMPT + agent_kit.SYSTEM_PROMPT,
         "tools": [tools[n] for n in ("find_talents", "get_match_id_from_profile_id",
                                      "score_candidates", "candidate_insights")],
         "middleware": agent_kit.middleware(step_limit, ledger)},
        {"name": "jd-writer",
         "description": ("Drafts a job description. Give it the job title, skills, language "
                         "and any other details from the user."),
         "system_prompt": JD_PROMPT + agent_kit.SYSTEM_PROMPT,
         "tools": [tools["generate_jd"]],
         "middleware": agent_kit.middleware(step_limit, ledger)},
        general_purpose(step_limit, ledger, agent_kit.VIRA),
    ]


def general_purpose(step_limit: int, ledger: agent_kit.CallLedger,
                    toolset: agent_kit.Toolset) -> dict:
    """Replaces the auto-added general-purpose subagent, which would otherwise run the
    tools without our guard, call cap or domain rules.  It gets the main agent's tools."""
    return {**GENERAL_PURPOSE_SUBAGENT,
            "system_prompt": GENERAL_PURPOSE_SUBAGENT["system_prompt"] + "\n\n" + toolset.prompt,
            "middleware": agent_kit.middleware(step_limit, ledger, toolset)}


def build_agent(*, approve_all: bool = False, step_limit: int = 12, model=None,
                toolset: agent_kit.Toolset = agent_kit.VIRA):
    tools = toolset.tools()
    ledger = agent_kit.CallLedger()
    if toolset is agent_kit.VIRA:
        style, helpers = WORKING_STYLE, subagents({t.name: t for t in tools}, step_limit, ledger)
    else:            # Jeni: no specialists yet, just the guarded general-purpose subagent
        style, helpers = JENI_WORKING_STYLE, [general_purpose(step_limit, ledger, toolset)]
    return create_deep_agent(
        model=model or agent_kit.build_chat_model(),
        tools=tools,
        system_prompt=toolset.prompt + style,
        subagents=helpers,
        middleware=[TodoListMiddleware(), *agent_kit.middleware(step_limit, ledger, toolset)],
        # subagents inherit it
        interrupt_on=agent_kit.interrupt_on(approve_all, toolset=toolset) or None,
        checkpointer=InMemorySaver(),
        name="vira-deepagent",
    )


def show_files(result: dict) -> None:
    """Print whatever the agent wrote to its virtual filesystem."""
    for path, data in sorted((result.get("files") or {}).items()):
        print(printable(f"\n=== virtual file {path} ==="))
        print(printable(data.get("content", "") if isinstance(data, dict) else data))


def main(argv=None):
    # no jeni_db: its db tools are async, and this runner drives the graph synchronously
    args = agent_kit.parser("VIRA agent on deepagents.", toolsets=("jeni", "vira"),
                            default_tools="jeni").parse_args(argv)
    agent_kit.setup(args)
    agent = build_agent(approve_all=args.approve_all, step_limit=args.step_limit,
                        toolset=agent_kit.cli_toolset(args))
    agent_kit.repl("deepagents", agent, args, after=show_files)


if __name__ == "__main__":
    main()
