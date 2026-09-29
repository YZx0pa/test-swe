# Running the VIRA agent on LangGraph, deepagents and other frameworks

This branch (`feat/agent-frameworks`) runs the same recruiter agent on three runtimes side by
side (mini-swe-agent, LangGraph and deepagents) and exposes VIRA over MCP for everything else.
All of them go through one guarded path to VIRA, so masking, auditing and the confirm gate
behave the same everywhere.

## Recommendation

| Use | When | File |
|---|---|---|
| **LangGraph agent** (`create_agent`) | Default for today's VIRA: short tasks over 4 read/compute endpoints. Typed tools, no shell, fewest tokens. | `run_langgraph.py` |
| **LangGraph workflow** (`StateGraph`) | Recruiter pipelines whose steps are known in advance. Code picks every step, so an approval can't be skipped and id types can't be mixed up. | `run_workflow.py` |
| **deepagents** | Long, multi-part work: many jobs, reports, research. Planning, subagents and scratch files are worth their token cost there, and only there. | `run_deepagent.py` |
| **MCP server** | Reaching any other framework or tool (Claude Code, OpenAI Agents SDK, PydanticAI, …) without rewriting the tools. | `vira_mcp.py` |
| mini-swe-agent | Baseline only. Its "bash" now runs just `recruiter_cli` and `echo`, without a shell (next section). | `run_mini.py` |

## 1. Where we started: mini-swe-agent + bash

```
run_mini.py ─ DefaultAgent (yolo) ─ model writes a shell string ─ LocalEnvironment (shell=True)
                                         └─ python3 recruiter_cli.py --mode … <subcommand> --flags ─ VIRA / MockVira
```

`recruiter_cli.py` already carries the domain weight: endpoint routing, the query-vs-body
split, auth headers from env, `_mask_pii`, `_audit` and `NEEDS_CONFIRM`. The trouble was the bash
tool itself. Each problem below and how it is closed now (§11 lists the tests):

- **Secrets were one command away.** mini's `LocalEnvironment` runs every command through
  `subprocess(shell=True)` with `env=os.environ | …`, in the repo directory, and nothing ever asked
  for confirmation (`mode: yolo` isn't a `DefaultAgent` setting at all). `env`, `cat .env`, or
  `VIRA_BASE_URL=http://elsewhere python3 recruiter_cli.py …` would have put or sent credentials
  out. litellm and minisweagent also call `load_dotenv()` on import, so the whole `.env` landed in
  `os.environ` even without run_mini's own call.
  **Now:** `run_mini.py` gives the model `mini_env.RecruiterEnvironment`. It accepts only what
  `mini_policy.parse()` allows: `echo …` (answered in Python) and
  `python3 recruiter_cli.py <subcommand> <flags>`, which runs as an argv list with this interpreter,
  without a shell. Everything else, including `;`, pipes, redirection, `VAR=` prefixes and newlines,
  comes back `Refused: …` and never runs. The child gets `PATH`, `LANG`, `EVENTS_LOG` and the
  `VIRA_*` values only; the model provider's key and your shell's variables stay out. `.env`
  loading is disabled before minisweagent and litellm are imported, and `run_mini` reads `.env`
  itself.
- **The model picked real vs mock.** It wrote `--mode` itself and the CLI defaulted to `real`.
  **Now:** `--mode` is required by the CLI, and the environment adds the host's mode
  (`JENI_MODE`, default `mock`). A different `--mode` from the model is refused. In real mode,
  `score-candidates` and `candidate-insights` wait for a y/N at the terminal.
- **Prompt rules that exist only to tame bash:** one command per turn, quoting `SUMMARY` (an unquoted
  `|` is now refused rather than read as a pipe), `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, and
  `tool_choice="required"` to stop prose turns that re-fired real API calls. These remain; they
  are the cost of the bash-shaped interface.
- **The confirm gate has no human in it.** It told the model to retry with `--confirmed`, a flag
  that doesn't exist. **Now:** the message says the agent can't approve, and `mini_policy` refuses
  any `--confirmed`. A write command still needs an approval design (§10).
- **Jinja saw the environment.** mini renders its prompt templates with `os.environ` among the
  variables, so a `{{ VIRA_API_KEY }}` typed into `commands.md` would have put the key in the
  prompt. **Now:** the system prompt is passed as a template variable, so `commands.md` is never
  parsed, and the environment's template variables no longer include `os.environ`.
- **Masking was by exact key name,** so `first_name`, `phone_number`, an email inside an insight
  summary, or a non-JSON real-mode reply (`{"raw": text}`) passed through. **Now:** keys are
  normalised (camelCase, `-`) and matched against `PII_KEYS` plus word patterns, and every string
  value has emails and grouped or `+`-prefixed phone numbers replaced. Names inside free text
  still aren't detected.

## 2. One tool contract, many runtimes

```
recruiter_cli.execute(…, mode=…)   gate → _call → _audit → _mask_pii      (one guarded path)
   ├─ recruiter_cli CLI ────────── run_mini.py            (mini-swe-agent, via mini_env: no shell)
   ├─ vira_tools.py (typed) ────┬─ run_langgraph.py       LangGraph agent loop (create_agent)
   │                            ├─ run_deepagent.py       deepagents harness
   │                            └─ vira_mcp.py (stdio)    any MCP client
   └─ run_workflow.py (via vira_tools)                    explicit LangGraph graph
```

| File | Role |
|---|---|
| `recruiter_cli.py` | `execute()` plus four typed actions own routing and the query/body split. The CLI wraps them; its output is byte-identical to before. |
| `vira_tools.py` | The model-facing contract: four typed functions whose `Annotated[…, Field(…)]` signatures and docstrings **are** the tool schema. Edit them the way you'd edit `commands.md`. |
| `agent_kit.py` | Shared prompt, model factory, middleware, approval loop and REPL for the LangChain-based runners. |
| `run_langgraph.py` / `run_workflow.py` / `run_deepagent.py` | The runtimes. New runners default to `--mode mock`; `--mode real` reaches VIRA. |
| `vira_mcp.py` | The same tools over MCP. |
| `compare_agents.py` | Live side-by-side on mock VIRA (section 7): `--model`, `--repeat`, `--json` traces. |
| `grounding.py` | Traces every tool argument and answer id/score to the task or an earlier result, and flags ids of the wrong kind ([format](agent-frameworks-changes.md#6-trace-and-grounding-json)). |
| `mini_policy.py` | What mini's model may run: `echo …` or one `python3 recruiter_cli.py <subcommand>` call; everything else is refused. Stdlib only, shared by `mini_env`, `grounding` and `compare_agents`. |
| `mini_env.py` | `RecruiterEnvironment`, mini's "bash" tool: runs what `mini_policy` allows as an argv list, with the host's mode and a minimal environment. |
| `tests/` | 165 offline tests: mock VIRA, a scripted fake model, `.env` disabled, audit log in `tmp_path`. |

Guarantees that hold in every new runtime:
- Results are masked by `_mask_pii` before they reach the model, the graph state or the checkpointer:
  sensitive keys by name and pattern, and emails and phone numbers inside any string.
- Every VIRA call lands in the audit log in the same format.
- The host sets real/mock once (`vira_tools.configure`); the model never sees a mode parameter.
- Tools pass `confirmed=False`, so a confirm-gated command can't run from them (see follow-ups).
- A failure comes back as `{"status": "error", …}` with the exception type only, never hosts or URLs.
  It is audited too (`status: "exception"`, `error: <type>`), and the CLI prints one JSON line
  instead of a traceback.
- Real calls go only to an `https://` `VIRA_BASE_URL` (plain http only to loopback), never follow
  redirects, ignore proxy and CA variables from the environment (`VIRA_CA_BUNDLE` sets a CA), and
  aren't sent at all when a `VIRA_*` credential is empty.
- The audit log is owner-only (0600), and its `query` is masked like the body.

How mini's bash-era prompt rules became structure:

| run_mini.py rule | New runtimes |
|---|---|
| "Use `--mode mock`" (model-written) | Mode is fixed by the host; no tool parameter for it |
| One command per response; quote `SUMMARY`; `echo COMPLETE_TASK…`; `tool_choice="required"` | Gone: typed calls, parallel calls allowed, the loop ends when the model answers in prose |
| "Never repeat a command with the same arguments" | `ToolCallGuard` refuses exact repeats (normalised args) without calling VIRA, across the main agent and its subagents |
| "match_id, app_id, profile_id, job_id are DISTINCT" | `ToolCallGuard` refuses an id passed as a different kind than it came back as (e.g. a `profile_id` sent as `match_ids`) |
| `step_limit: 12` | `ModelCallLimitMiddleware(thread_limit=12)`, one fresh thread per task |
| "Tell the user … retry with `--confirmed`" | `--approve-all`: a LangGraph interrupt pauses before the call; approve, edit or reject |
| Arg validation by argparse (strings) | Pydantic schema: `job_ids` ≥1, `job_title` non-empty, and so on; the model gets the error and retries |

## 3. LangGraph

LangGraph is a runtime for graphs of steps over a shared **state**:
- **nodes** are Python functions that return state updates;
- **edges**, fixed or conditional, choose what runs next;
- a **checkpointer** (`InMemorySaver`, `SqliteSaver`, …) saves the state after every step under a
  `thread_id`;
- **`interrupt(value)`** pauses a node; `invoke(Command(resume=…))` on the same thread continues it.

That last pair is what makes human approval a structural step rather than a prompt request.

### 3a. The agent loop: `run_langgraph.py`

`langchain.agents.create_agent` (which replaced `langgraph.prebuilt.create_react_agent`) compiles
a model node ↔ tools node loop into a LangGraph graph. Behaviour is added as middleware:

```python
create_agent(build_chat_model(), vira_tools.langchain_tools(),
             system_prompt=SYSTEM_PROMPT,
             middleware=[HumanInTheLoopMiddleware(...),   # only with --approve-all; outermost
                         ToolCallGuard(),                   # refuse repeats, contain crashes
                         ModelCallLimitMiddleware(thread_limit=12, exit_behavior="end")],
             checkpointer=InMemorySaver())
```

- `create_agent`'s recursion limit is now 9,999, so the call-limit middleware is the real step cap.
- A string model is built with no kwargs; pass an instance to control options.
  `build_chat_model()` maps litellm-style `CHAT_MODEL` ids (`openai/…` → `openai:…`) and uses Chat
  Completions for OpenAI, like mini's litellm path. The Responses API stores responses server-side
  by default.
- `@wrap_tool_call` on a plain function registers only the sync hook, so it would fail under
  `ainvoke`. Subclass `AgentMiddleware` and implement both hooks, as `ToolCallGuard` does.
- LangChain's `PIIMiddleware` is not a substitute for `_mask_pii`. It scrubs tool results after the
  raw message is already checkpointed, ignores artifacts, has no phone or name detector, and its
  email `mask` keeps the local part.

### 3b. The explicit workflow: `run_workflow.py`

```
START → score → shortlist (top-k, in code) → approve ⏸ interrupt → insights → summarize (1 LLM call) → END
               └─ VIRA error → END                     └─ rejected ───────────┘
```

The model never chooses a tool, so it can't skip the approval, pass a profile id as a match id,
or repeat a call. On resume the `approve` node re-runs from the top, so nothing before
`interrupt()` may have side effects. Prefer this shape for any pipeline you can draw on a
whiteboard.

### Human in the loop

`--approve-all` gates every VIRA tool. Each model turn raises one interrupt that batches all gated
calls. The runner answers with one decision per call (`approve`, `edit` with new args, `reject`)
via `Command(resume={"decisions": [...]})` on the same thread. A rejected call never reaches VIRA
(tested). Interrupts need a checkpointer; `InMemorySaver` covers a single process.

## 4. deepagents: `run_deepagent.py`

deepagents 0.7 is a harness on top of `create_agent`. It adds:
- **planning**: `write_todos` via `TodoListMiddleware`, opt-in since 0.7;
- **a virtual filesystem**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` and
  `delete`, held in graph state by `StateBackend`, with tool results over 20k tokens evicted to it;
- **subagents** behind a `task` tool: each runs in its own context and returns only its final
  answer;
- automatic **summarization** near the context limit.

Our configuration:
- **Subagents:** `sourcing-analyst` gets find, score and insights; `jd-writer` gets `generate_jd`.
  We also pass our own `general-purpose` spec, replacing the auto-added one.
- **Middleware:** `TodoListMiddleware` plus the same guard and call cap as the LangGraph agent.
- **Approval:** `interrupt_on` is only set with `--approve-all`; subagents inherit it (tested).

Security: keep the default **`StateBackend`**, where files live in graph state per thread and never
touch the host disk. Never use `FilesystemBackend` (reads and writes real files, defaulting to the
current directory, which holds `.env`) or `LocalShellBackend` (runs `subprocess(shell=True)`). The
`execute` tool is registered but is never offered to the model unless the backend is a sandbox, and
a hallucinated call is refused at runtime (both tested).

Gotchas we hit:
- **Subagents have their own message history,** so a guard that reads history can't see the
  parent's calls. A subagent re-sent the parent's exact `generate_jd` call until the guard got a
  shared, thread-scoped `CallLedger` (section 7).
- **Subagents do not inherit custom middleware.** The auto-added general-purpose subagent only
  inherits middleware that replaces one of its default slots. Without our override it would run
  the VIRA tools with no guard, no call cap and no domain rules.
- **`model=None` is deprecated** and falls back to Claude Sonnet, so always pass a model.
  gpt-5-mini has no deepagents harness profile and isn't in its evaluated-model list.
- **Planning costs tokens,** and the harness likes to retry and delegate when a result looks
  incomplete: 27–38k tokens where LangGraph used 3–4k (section 7).

Use it when a task needs many steps or several jobs, benefits from delegating per-job work to
isolated subagents, or should produce a written artifact (a report file in the virtual FS).

## 5. MCP server: `vira_mcp.py`

One stdio server exposes the four tools to any MCP client. It uses `mcp==1.30` (`mcp.server.fastmcp`);
mcp 2.x renamed `FastMCP` to `MCPServer` and would conflict with `langchain-mcp-adapters`
(`mcp<2`) in the same venv.

```bash
# Claude Code, from the repo root:
claude mcp add vira -- "$PWD/.venv/bin/python" "$PWD/vira_mcp.py" --mode mock
```

- **Credentials stay server-side.** A stdio child gets only a minimal environment from the
  client; the server loads `.env` itself, found next to `recruiter_cli.py` whatever the working
  directory. No client config ever carries a VIRA key.
- `--mode` is required. stdout is the JSON-RPC channel, and a test checks that every stdout line
  is JSON-RPC.
- Annotations: find/JD are `readOnlyHint`; score/insights trigger calculations, so they're
  `idempotentHint` only. They are hints, not enforcement.
- Confirm-gated tools are never exposed. MCP approval depends on the client (elicitation needs
  client support), so a write tool needs an approval design first.

Client snippets (not run in this branch; versions per PyPI metadata):

```python
# LangGraph via langchain-mcp-adapters (async-only tools → use ainvoke)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
client = MultiServerMCPClient({"vira": {"transport": "stdio", "command": sys.executable,
                                        "args": ["vira_mcp.py", "--mode", "mock"]}})
async with client.session("vira") as session:
    tools = await load_mcp_tools(session)
    agent = create_agent(model, tools, ...)            # then: await agent.ainvoke(...)

# OpenAI Agents SDK: 0.20.0 is the last release on openai 2.x (litellm keeps us there)
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
async with MCPServerStdio(params={"command": sys.executable,
                                  "args": ["vira_mcp.py", "--mode", "mock"]}) as vira:
    agent = Agent(name="Recruiter", model="gpt-5-mini", mcp_servers=[vira])
    result = await Runner.run(agent, "Find talents for job 123")

# PydanticAI: 2.31.1 (or 1.107.6) is the last on openai 2.x
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport
agent = Agent("openai:gpt-5-mini",
              toolsets=[MCPToolset(StdioTransport(command=sys.executable,
                                                  args=["vira_mcp.py", "--mode", "mock"]))])
```

## 6. Other frameworks at a glance

Checked against PyPI metadata on 2026-09-23 (not install-tested unless noted). "Our pins" means
python 3.11, pydantic 2.13.5, openai 2.54.0 (held below 3 by litellm), httpx 0.28.1.

| Framework | Latest | Models | Approval (HITL) | MCP | Works with our pins | Notes |
|---|---|---|---|---|---|---|
| LangGraph + LangChain | langgraph 1.2.12 / langchain 1.4.2 | any | `interrupt()`; `HumanInTheLoopMiddleware` | via adapters | yes (installed) | Graph runtime + checkpointing; the base for deepagents |
| deepagents | 0.7.18 (beta, near-daily patches) | any (defaults to Claude) | `interrupt_on` | via adapters | yes (installed) | Pulls in langchain-anthropic, langchain-google-genai, langsmith |
| MCP SDK | 1.30.0 (v1) / 2.2.0 | n/a | client-side | is MCP | 1.30 installed | v2: `MCPServer`, spec 2026-07-28 |
| OpenAI Agents SDK | 0.22.3 | OpenAI-first; LiteLLM beta | `needs_approval` → `RunState.approve/reject` | stdio/HTTP | only ≤ 0.20.0 | 0.21+ needs openai 3 |
| PydanticAI | 2.48.0 | any | `requires_approval` → `DeferredToolRequests` | `MCPToolset` | only ≤ 2.31.1 / 1.107.6 | 2.32+ needs openai 3 + httpx2 |
| Claude Agent SDK | 0.2.158 | Claude only | `can_use_tool`, hooks | native | yes | Bundles the Claude Code CLI (~100 MB wheel); `tools=[]` removes built-ins |
| Microsoft Agent Framework | 1.19.0 | any | `approval_mode="always_require"` | stdio/HTTP | yes | Successor to AutoGen / Semantic Kernel |
| Google ADK | 2.9.2 | Gemini-first; LiteLLM | `require_confirmation` (experimental) | `McpToolset` | yes | Heavy (FastAPI, OpenTelemetry) |
| smolagents | 1.26.0 | any | step callbacks | `ToolCollection.from_mcp` | yes | `CodeAgent` executes model-written Python and needs a sandbox |
| CrewAI | 1.15.22 | any | `human_input` | adapters | **no** (`pydantic<2.13`) | Heavy; role-based crews + flows |

## 7. Results: the same tasks on three runtimes

`compare_agents.py` on 2026-09-23 with gpt-5-mini against mock VIRA. PASS/FAIL is judged on the
audit log, i.e. what actually reached VIRA, not on the model's prose. Each cell is one run and
LLM runs aren't deterministic, so treat this as anecdotal. Cells show model calls · tokens ·
wall time; costs are litellm price-table estimates.

Run 2, with the current tool contract. Both runs predate the security hardening in §11 (mini's
confined environment, and the guards and approvals added after it); re-run before comparing.

| task | check | mini | LangGraph | deepagents |
|---|---|---|---|---|
| find | one find-talents for job 123 | FAIL · 4 · 8.5k · 48s | PASS · 2 · 2.6k · 10s | PASS · 2 · 9.3k · 11s |
| jd_ar | one generate-jd: lang=ar, given title + skills | PASS · 2 · 3.5k · 20s | PASS · 2 · 4.1k · 32s | FAIL · 8 · 34.7k · 90s |
| score_insights | score 11,12, then insights for 11 | PASS · 3 · 5.3k · 29s | PASS · 3 · 3.6k · 9s | PASS · 3 · 14.0k · 14s |
| id_trap | find, then stop (no match_ids exist) | PASS · 4 · 8.9k · 59s | PASS · 2 · 3.1k · 16s | PASS · 6 · 27.1k · 72s |
| no_title | no call: title missing, don't invent one | FAIL · 12 · 37.4k · 215s | PASS · 1 · 1.4k · 5s | PASS · 1 · 4.7k · 5s |
| **total** | | **3/5** · ~$0.069 | **5/5** · ~$0.011 | **4/5** · ~$0.045 |

Run 1, before the contract fix below: mini 5/5 (~$0.033), LangGraph 4/5 (~$0.016),
deepagents 4/5 (~$0.025). All live runs on this branch together cost about $0.25.

What the runs showed:

1. **A typed schema is part of the prompt.** In run 1 both typed runtimes failed `jd_ar`. They
   filled `job_function`, `industry`, `other_requirements` and extra skills nobody asked for,
   re-called `generate_jd`, and then wrote the Arabic JD themselves. mini didn't, because
   `commands.md` never documents those flags. The fix: the field descriptions now say "only what
   the user named; leave empty otherwise", and the prompt says to relay tool results. LangGraph
   then passed.
2. **mini's failures are the bash failure modes.** Its code and prompt were identical in both
   runs, yet run 2 failed twice:
   - A typo, `--job-ids 123..`, crashed the CLI's int parse, and the model gave up.
   - Asked for a JD with no title, it had no way to ask the user, since every turn must be a
     command. It looped through echo, `--help` and `--job-title ""` (which reached VIRA with an
     empty title), plus one off-policy command, until the 12-call limit. The typed runtimes
     answered that case in one call.
3. **deepagents' subagents got past the duplicate guard.** In run 2, after the mock's
   placeholder JD, the main agent delegated to `jd-writer` twice. Each subagent re-sent the
   identical call, because each has its own message history. This is now fixed with a
   thread-scoped `CallLedger` shared by the main agent and every subagent (tested). A targeted
   re-run sent no exact repeats. The model varied the arguments instead (adding "Include
   sections…" to `other_requirements`, reordering skills) and made 4 calls. deepagents tends to
   retry and delegate when a result looks incomplete; the mock's placeholder text triggered it
   here.
4. **Cost and speed.** LangGraph was cheapest in almost every cell (1.4–4.1k tokens per task).
   deepagents starts from about 4.7k tokens for a single call because of its filesystem, todo and
   task tool schemas, and reaches 27–38k when it plans and delegates. mini needs extra calls just
   to finish (echo SUMMARY, echo COMPLETE) and was 2–5× slower.

## 8. Adding a VIRA endpoint

1. `recruiter_cli.py`: add a typed action that calls `execute()` (path plus the query/body split),
   and a CLI subcommand that parses flags and calls it.
2. `commands.md`: document the subcommand for mini.
3. `vira_tools.py`: add a typed function (its signature and docstring are the model's contract)
   and append it to `TOOLS`. Add it to `READ_ONLY` if it is a pure read.
4. Writes, notifications and anything irreversible: add the CLI name to `NEEDS_CONFIRM`. Today that
   makes the tool unreachable from `vira_tools` and hides it from MCP, so decide the approval
   design first (follow-ups).
5. New sensitive response fields: extend `PII_KEYS` (or the `_PII_WORD` / `_PII_NAME` patterns
   beside it) and add the key to `test_pii_keys_are_masked_by_name_and_pattern`.
6. `mock_vira.py`: add a mock reply. Tests: add a parity case to `tests/test_recruiter_cli.py`
   and the schema expectation to `tests/test_agents.py`.

## 9. Running it

```bash
uv pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q                                   # offline, 165 tests

JENI_MODE=mock python run_mini.py                                         # mini baseline, no shell
python run_langgraph.py --task "Find potential talents for job 123"       # mock VIRA by default
python run_langgraph.py --approve-all                                     # approve/edit/reject each call
python run_workflow.py --app-ids 11,12,13 --top 2
python run_deepagent.py --task "For jobs 101 and 102, find talents and write /report.md"
python vira_mcp.py --mode mock                                            # for MCP clients
python compare_agents.py --out report.md                                  # live LLM, mock VIRA only
python compare_agents.py --model gpt-4o-mini --repeat 3 --json runs.json   # traces for the page
python run_langgraph.py --mode real --task "…" --trace-json real.json     # one real run, local file
```

LangSmith tracing is forced off in the new runners: `set_tracing(False)` sets all four of
`LANGSMITH_TRACING_V2`, `LANGSMITH_TRACING`, `LANGCHAIN_TRACING_V2` and `LANGCHAIN_TRACING`
(langsmith reads `…_V2` first) and clears langsmith's cached lookup. `--trace` allows it.
Traces would carry prompts and (masked) tool results off the machine.

## 10. Follow-ups

- **Approval design for write tools.** Proposal: the HITL approval *is* the confirmation.
  `vira_tools` passes `confirmed=True` only for tools that `agent_kit` gates with `interrupt_on`,
  asserted at build time for the main agent and every subagent. MCP keeps them off, or uses
  elicitation.
- **get-match-id:** the `id_trap` task shows the gap: without it, scoring suggested talents
  can't be done correctly.
- **Durable approvals:** `SqliteSaver` (langgraph-checkpoint-sqlite) so a paused run survives a
  restart. Encrypt checkpoints at rest (`EncryptedSerializer`).
- **Per-task tool budgets shared across subagents** (e.g. one `generate_jd` per task). Exact-repeat
  refusal doesn't stop retries with tweaked arguments, and `ToolCallLimitMiddleware` counts per
  agent context, so the budget needs the ledger approach.
- **Model choice for deepagents:** gpt-5-mini works, but a stronger model is what deepagents is
  tuned and evaluated on.
- **Trace UI:** LangGraph Studio / LangSmith, if the team wants it and accepts data leaving the
  machine.

## 11. Security model

The threat that shapes everything here is **prompt injection**: the task text, and VIRA results
built from candidate data (CV-derived insight summaries, generated JD text), reach the model. So
every control sits in code the model can't reach, and each one has a test.

| Threat | Control | Tests |
|---|---|---|
| mini's model runs arbitrary shell commands (`env`, `cat .env`, `VAR=… python3 recruiter_cli.py`, chained or multi-line commands) | `mini_env.RecruiterEnvironment` runs only what `mini_policy.parse()` accepts: `echo` (answered in Python) and `python3 recruiter_cli.py <subcommand>` as an argv list, with no shell; the rest is refused before anything starts | `test_mini_env.py`: `test_everything_else_is_refused`, `test_refused_commands_never_start_a_process` |
| Secrets reach mini's commands or prompt templates | The child env is `PATH`/`LANG`/`EVENTS_LOG` plus `VIRA_*` passed as `secrets` (kept off the config); template variables exclude `os.environ`; `.env` loading is disabled before minisweagent and litellm import, and `run_mini` reads `.env` itself | `test_cli_runs_as_argv_with_the_host_mode_and_a_minimal_env`, `test_secrets_stay_out_of_templates_and_serialisation` |
| The model chooses real mode | `--mode` is required by the CLI; mini's environment adds the host's mode (`JENI_MODE`, default mock) and refuses a different one, including argparse abbreviations | `test_mode_is_required`, `test_allowed_commands`, `test_everything_else_is_refused` |
| VIRA output ends mini's run early | `COMPLETE_TASK…` is honoured only from `echo`, never from recruiter_cli output | `test_echo_complete_ends_the_run_but_vira_output_cannot` |
| Side effects in real mode (mini) | `score-candidates`/`candidate-insights` wait for a y/N; with no approver they are refused | `test_real_mode_side_effects_wait_for_approval`, `test_real_mode_without_an_approver_refuses_side_effects` |
| The model self-confirms a gated command | `mini_policy` refuses `--confirmed`; the gate's message says the agent can't approve | `test_everything_else_is_refused`, `test_confirm_gate_blocks_before_any_call` |
| Tracebacks put hosts and paths into mini's context | Only argparse's error line is returned on failure | `test_argparse_errors_reach_the_model_but_tracebacks_do_not` |
| A chained command's output lands in traces | `grounding.parse_cli` uses `mini_policy`, so such a call is off-policy and its output hidden | `test_a_vira_call_chained_to_another_command_is_off_policy` |
| `commands.md` parsed as a template (SSTI, key injection) | The system prompt is passed as a template variable; Jinja never parses `commands.md` | smoke-tested; `run_mini` is kept out of the offline suite |
| The API key follows a redirect, or goes through an env-configured proxy | `requests.Session` with `trust_env=False` and `allow_redirects=False` (requests strips only `Authorization` on a cross-host redirect); a 3xx is an error result | `test_real_calls_ignore_proxy_env_and_follow_no_redirects` |
| The key is sent in cleartext, or sent empty | `https://` required except for loopback hosts; an empty `VIRA_*` credential returns an error before any request | `test_plain_http_only_to_loopback`, `test_missing_credentials_fail_closed` |
| Error text carries hosts or large bodies into model context | Exceptions become `{"status": "error"}` with the type only; a non-JSON reply is capped at 500 characters | `test_failed_calls_are_audited_and_the_cli_prints_one_json_line`, `test_non_json_replies_are_capped`, `test_vira_failure_becomes_an_error_result_without_hosts` |
| Failed calls leave no trace | `execute()` audits the exception type, then re-raises | `test_failed_calls_are_audited_and_the_cli_prints_one_json_line` |
| Other local users read the audit log or `.env` | The audit log is created 0600 (and tightened if older); recruiter_cli warns on stderr when `.env` is group- or world-readable, checking mode bits only | `test_audit_log_is_owner_only`, `test_a_shared_dotenv_is_reported_by_mode_bits_only` |
| `.env` from a parent directory gets loaded | recruiter_cli loads the `.env` next to it, by path | — |
| Candidate PII reaches the model (and the model provider) | `_mask_pii`: normalised keys matched against `PII_KEYS` and word patterns (`first_name`, `phone_number`, `linkedin_url`, …, but not `job_name_similarity`); emails and phone numbers replaced inside every string, including a non-JSON `raw` reply. Names in free text are not detected | `test_pii_keys_are_masked_by_name_and_pattern`, `test_other_keys_are_kept`, `test_emails_and_phones_inside_text_are_masked`, `test_non_json_reply_text_is_scrubbed` |
| Prompts and tool results shipped to LangSmith | All four tracing variables are set, and langsmith's cached lookup cleared | `test_tracing_stays_off_even_with_langsmith_tracing_v2_set` |
