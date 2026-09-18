#!/usr/bin/env python3
"""Run recruiter tasks through real mini-swe-agent.

The API/bash-command descriptions live in commands.md (edit ONLY that file to
add or change commands). This runner merges commands.md into the fixed
system-prompt scaffolding (rules + format that rarely change).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")   # load .env beside this file, dir-independent

import yaml
import minisweagent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment
from minisweagent.agents.default import DefaultAgent

MODE = os.environ.get("JENI_MODE", "real")   # real | mock

# ---- fixed scaffolding (rules/format — rarely changes) ----------------------
RULES = """\
Rules:
- Do ONLY what the user's task asks. Do not add extra steps.
- Issue exactly ONE command per response, then read its JSON output before the next.
- match_id, app_id, profile_id, and job_id are DISTINCT id types. Never pass one
  where another is expected, try to call "get-*" API to get the correct id.
- Never invent ids, titles, or skills. Application ids and emails must come from the user.
- A result may say status 'ok' but contain an error message in its fields
  (e.g. "No match_id is obtained from db"). Treat such messages as FAILURES and
  correct your next command; do not assume success from status alone.
- If the task cannot be done with these commands, run
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT and state plainly you cannot do it.
- When the whole task is done, run echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT by itself.
"""

def build_system_prompt() -> str:
    """Merge commands.md (the part you edit) into the fixed scaffolding."""
    commands = (HERE / "commands.md").read_text(encoding="utf-8")
    header = (
        f"You operate a recruiter platform by running "
        f"`python3 recruiter_cli.py --mode {MODE} <subcommand> <flags>` via the bash tool.\n"
        f"Run one command, read its JSON output, then decide the next.\n"
    )
    return f"{header}\n{RULES}\nCommands:\n\n{commands}\n"

SYSTEM = build_system_prompt()
INSTANCE = ("Recruiter task {{task}}. Use python3 recruiter_cli.py. "
            "Do not analyze any codebase. Issue one bash tool call now.")


def build_agent():
    cfg_path = Path(minisweagent.__file__).parent / "config" / "mini.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    agent_cfg = cfg.get("agent", {})
    env_cfg = cfg.get("environment", {})
    model_cfg = cfg.get("model", {})

    agent_cfg["system_template"] = SYSTEM
    agent_cfg["instance_template"] = INSTANCE
    agent_cfg["mode"] = "yolo"
    agent_cfg["step_limit"] = 12

    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    model_kwargs["parallel_tool_calls"] = False   # one command per turn
    model = LitellmModel(model_name=os.environ.get("CHAT_MODEL", "gpt-4o"),
                         model_kwargs=model_kwargs)
    env = LocalEnvironment(env=env_cfg["env"]) if env_cfg.get("env") else LocalEnvironment()
    return DefaultAgent(model, env, **agent_cfg)


def show(messages):
    import json
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                print(f"[{i}] assistant → run: {cmd}")
        elif role == "tool":
            print(f"[{i}] tool      → {m.get('content','')}")
        elif role == "user":
            print(f"[{i}] user      : {m.get('content','')}")
        elif role == "system":
            print(f"[{i}] system    : (instructions)")
        elif role == "exit":
            print(f"[{i}] exit      : {m.get('extra',{}).get('exit_status')}")


def main():
    print(f"Recruiter agent (mini, mode={MODE}). Type a task, or 'quit'.")
    while True:
        try:
            task = input("\ninput task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        if not task or task.lower() in {"quit", "exit"}:
            print("bye"); return
        agent = build_agent()
        try:
            result = agent.run(task)
            print("\n=== result ===")
            print(result)
            show(agent.messages)
        except Exception as exc:
            print(f"[error] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
