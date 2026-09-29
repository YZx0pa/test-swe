#!/usr/bin/env python3
"""Run recruiter tasks through real mini-swe-agent.

The API/bash-command descriptions live in commands.md (edit ONLY that file to
add or change commands). This runner merges commands.md into the fixed
system-prompt scaffolding (rules + format that rarely change).

The model's "bash" tool is mini_env.RecruiterEnvironment, not a shell: it runs
only `echo …` and `python3 recruiter_cli.py <subcommand> …`, with the mode set
here (JENI_MODE, default mock) and a minimal environment.  In real mode,
score-candidates and candidate-insights wait for your y/N first.
"""
import os
import shlex
from pathlib import Path

from dotenv import dotenv_values
from platformdirs import user_config_dir

HERE = Path(__file__).parent


def _dotenv_disabled() -> bool:
    return os.environ.get("PYTHON_DOTENV_DISABLED", "").casefold() in {"1", "true", "t", "yes", "y"}


def _read_dotenv() -> dict[str, str]:
    """mini's global .env (where `mini-extra config` puts keys), then this directory's, which wins."""
    if _dotenv_disabled():
        return {}
    mini_global = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or user_config_dir("mini-swe-agent"))
    values: dict[str, str] = {}
    for path in (mini_global / ".env", HERE / ".env"):          # later files win
        values |= {k: v for k, v in dotenv_values(path).items() if v is not None}
    return values                                               # read here, never printed


_DOTENV = _read_dotenv()
# Before minisweagent and litellm are imported: both call load_dotenv() on import, and
# litellm's search walks up from site-packages into this directory's .env.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")   # no cost-map download at import

# VIRA credentials go only to recruiter_cli's process; everything else (CHAT_MODEL,
# the model provider's key, JENI_MODE, EVENTS_LOG) to this one.  A variable already in
# the environment wins over .env, as with load_dotenv().
VIRA_KEYS = ("VIRA_BASE_URL", "VIRA_API_KEY", "VIRA_CLIENT_NAME", "VIRA_USER_ID")
VIRA_ENV = {k: os.environ[k] if k in os.environ else _DOTENV[k]
            for k in VIRA_KEYS if k in os.environ or k in _DOTENV}
for _key, _value in _DOTENV.items():
    if _key not in VIRA_KEYS:
        os.environ.setdefault(_key, _value)

import yaml  # noqa: E402
import minisweagent  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402

from mini_env import RecruiterEnvironment  # noqa: E402

MODE = os.environ.get("JENI_MODE", "mock")   # real | mock
if MODE not in ("real", "mock"):
    raise SystemExit(f"JENI_MODE must be 'real' or 'mock', not {MODE!r}")

# ---- fixed scaffolding (rules/format — rarely changes) ----------------------
RULES = """\
Rules:
- Do ONLY what the user's task asks. Do not add extra steps.
- Issue exactly ONE command per response, then read its JSON output before the next.
- match_id, app_id, profile_id, job_id are DISTINCT. Never pass one where another is
  expected. If you need an id type you don't have, you MUST call the matching get-*
  command (only one listed above; don't invent names). If none exists, run
  echo Not_Able_to_obtain_the_correct_id ONCE, then finish as described below.
- Never invent any field value. Every value must come from either the OUTPUT of a
  previous command, or explicit user input. If a required value is available from
  neither, do not guess — finish as described below and say what was missing.
- Never issue a command you have already run with the same arguments, and never
  repeat a command that failed the same way. If you cannot make progress, finish.
- "status": "ok" only means the API call was received — NOT that the operation
  succeeded. Always read the result fields for the real outcome. If a result field
  contains an error or a message saying nothing was found/obtained/processed, treat
  it as a FAILURE even though status is "ok", and correct your next command.
- If the task is only partially done or cannot be fully completed, before finishing
  run it as ONE quoted argument, exactly like this (the quotes matter — an unquoted
  | is refused):
  echo "SUMMARY: <what succeeded> | <what failed or is missing> | <why>"
  then run echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT by itself.
- When the whole task is fully done, run echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT by itself.
"""

def build_system_prompt() -> str:
    """Merge commands.md (the part you edit) into the fixed scaffolding."""
    commands = (HERE / "commands.md").read_text(encoding="utf-8")
    header = (
        "You operate a recruiter platform by running "
        "`python3 recruiter_cli.py <subcommand> <flags>` via the bash tool.\n"
        "The bash tool is not a shell: it runs only that command and `echo`, one per call "
        "(no pipes, redirection, `;`, `&&` or variables). The host sets real or mock mode.\n"
        "Run one command, read its JSON output, then decide the next.\n"
    )
    return f"{header}\n{RULES}\nCommands:\n\n{commands}\n"

SYSTEM = build_system_prompt()
INSTANCE = ("Recruiter task {{task}}. Use python3 recruiter_cli.py. "
            "Do not analyze any codebase. Issue one bash tool call now.")


def approve(args: list[str]) -> bool:
    """Real mode: score/insights trigger calculations on VIRA, so a person approves each."""
    print(f"\n[approval] recruiter_cli {' '.join(shlex.quote(a) for a in args)}")
    try:
        return input("approve? [y/N] > ").strip().lower().startswith("y")
    except EOFError:
        return False


def build_agent():
    cfg_path = Path(minisweagent.__file__).parent / "config" / "mini.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    agent_cfg = cfg.get("agent", {})
    env_cfg = cfg.get("environment", {})
    model_cfg = cfg.get("model", {})

    # The system prompt is passed as a template variable, so Jinja never parses
    # commands.md (a stray {{ … }} there would otherwise be evaluated).
    agent_cfg["system_template"] = "{{ system_prompt }}"
    agent_cfg["instance_template"] = INSTANCE
    agent_cfg.pop("mode", None)   # DefaultAgent has no confirm step; RecruiterEnvironment is the gate
    agent_cfg["step_limit"] = 12

    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    model_kwargs["parallel_tool_calls"] = False   # one command per turn
    # Reasoning models (gpt-5*) otherwise answer some turns in prose, which mini
    # rejects as a format error — and the model then retries by re-issuing its
    # previous command, firing real API calls a second time.  Forcing a tool call
    # every turn removes that failure mode; the agent always ends on a command
    # (echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT), so nothing needs a prose turn.
    model_kwargs["tool_choice"] = "required"
    model = LitellmModel(model_name=os.environ.get("CHAT_MODEL", "gpt-5-mini"),
                         model_kwargs=model_kwargs)
    env = RecruiterEnvironment(mode=MODE, env=env_cfg.get("env") or {}, secrets=VIRA_ENV,
                               approve=approve if MODE == "real" else None)
    agent = DefaultAgent(model, env, **agent_cfg)
    agent.extra_template_vars["system_prompt"] = SYSTEM
    return agent


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
