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
| `recruiter_cli.py` | `execute()` plus the typed actions (four VIRA endpoints and the assumed `get_match_id_from_profile_id` lookup) own routing and the query/body split. The CLI wraps them; its output is byte-identical to before. |
| `vira_tools.py` | The model-facing contract: typed functions whose `Annotated[…, Field(…)]` signatures and docstrings **are** the tool schema. Edit them the way you'd edit `commands.md`. |
| `agent_kit.py` | Shared prompt, model factory, middleware, approval loop and REPL for the LangChain-based runners. |
| `run_langgraph.py` / `run_workflow.py` / `run_deepagent.py` | The runtimes. New runners default to `--mode mock`; `--mode real` reaches VIRA. |
| `vira_mcp.py` | The same tools over MCP. |
| `compare_agents.py` | Live side-by-side on mock VIRA (section 7): `--model`, `--repeat`, `--json` traces. |
| `grounding.py` | Traces every tool argument and answer id/score to the task or an earlier result, and flags ids of the wrong kind ([format](agent-frameworks-changes.md#6-trace-and-grounding-json)). |
| `mini_policy.py` | What mini's model may run: `echo …` or one `python3 recruiter_cli.py <subcommand>` call; everything else is refused. Stdlib only, shared by `mini_env`, `grounding` and `compare_agents`. |
| `terminal.py` | `printable()`: strips control, bidi and zero-width characters from model- or VIRA-written text before it is printed. |
| `mini_env.py` | `RecruiterEnvironment`, mini's "bash" tool: runs what `mini_policy` allows as an argv list, with the host's mode and a minimal environment. |
| `jeni_tools.py` / `mock_jeni.py` | Jeni's own 22 tasks as typed tools, one task per call, on a mock of VIRA's task-group API (§12). `--tools jeni` in the LangGraph and deepagents runners. The task catalog is internal and read from `config/jeni_tasks.json`, outside git (`config/README.md`). |
| `db_queries.py` / `db_lookup.py` / `db_tools.py` | Read-only lookups in Jeni's database (§13): a job by title, a user by name, a job's applications, and checks of ids and emails, scoped to the company the runner sets. `--tools jeni_db` (LangGraph's default) adds them to Jeni's tasks. |
| `demo/`, `langgraph.json` | The demo (§14): the Jeni agent on LangGraph's dev server, on the mock with its changes kept and every write gated, plus a live data panel. |
| `tests/` | 335 offline tests: mock VIRA, a scripted fake model, `.env` disabled, audit log in `tmp_path`. |

Guarantees that hold in every new runtime:
- Results are masked by `_mask_pii` before they reach the model, the graph state or the checkpointer:
  sensitive keys by name and pattern, and emails and phone numbers inside any string.
- Every VIRA call lands in the audit log in the same format.
- The host sets real/mock once (`vira_tools.configure`); the model never sees a mode parameter.
- Tools pass `confirmed=False`, so a confirm-gated command can't run from them (see follow-ups).
- In real mode, `score_candidates` and `candidate_insights` (they trigger calculations on VIRA)
  always pause for a person's approval in the LangGraph and deepagents runners, subagents
  included. Over MCP they are listed only with `--allow-side-effects`.
- A failure comes back as `{"status": "error", …}` with the exception type only, never hosts or URLs.
  It is audited too (`status: "exception"`, `error: <type>`), and the CLI prints one JSON line
  instead of a traceback.
- Real calls go only to an `https://` `VIRA_BASE_URL` (plain http only to loopback), never follow
  redirects, ignore proxy and CA variables from the environment (`VIRA_CA_BUNDLE` sets a CA), and
  aren't sent at all when a `VIRA_*` credential is empty.
- The audit log is owner-only (0600), and its `query` is masked like the body.
- Inputs are bounded before anything is sent: at most 50 ids per argument (positive integers),
  200 characters per text value, 30 entries per list, and `lang` must look like a language code.
  The tool schemas carry the same limits (`recruiter_cli.MAX_*`), so the model sees them.
- `ToolCallGuard` refuses an id that is in neither the task nor an earlier tool result (a
  reviewer's `edit` counts as user input), as well as ids of the wrong kind.

How mini's bash-era prompt rules became structure:

| run_mini.py rule | New runtimes |
|---|---|
| "Use `--mode mock`" (model-written) | Mode is fixed by the host; no tool parameter for it |
| One command per response; quote `SUMMARY`; `echo COMPLETE_TASK…`; `tool_choice="required"` | Gone: typed calls, parallel calls allowed, the loop ends when the model answers in prose |
| "Never repeat a command with the same arguments" | `ToolCallGuard` refuses exact repeats (normalised args) without calling VIRA, across the main agent and its subagents. A read, or a call that failed, may run again once a write ran after it; a failed or refused call also once the user wrote again |
| "match_id, app_id, profile_id, job_id are DISTINCT" | `ToolCallGuard` refuses an id passed as a different kind than it came back as (e.g. a `profile_id` sent as `match_ids`), and points at `get_match_id_from_profile_id`, the one tool that turns profile ids into match ids |
| "Never invent any field value" (for ids) | `ToolCallGuard` refuses an id found in neither anything the user wrote (any turn) nor an earlier tool result |
| `step_limit: 12` | `ModelCallLimitMiddleware(run_limit=12)`: 12 model calls per user turn (and per resume after an approval), so a conversation can go on |
| "Tell the user … retry with `--confirmed`" | A LangGraph interrupt pauses before the call (approve, edit or reject): for score/insights always in real mode, for every tool that changes data with `--approve-all` |
| Arg validation by argparse (strings) | Pydantic schema: `job_ids` 1–50 positive ids, `job_title` 1–200 characters, `lang` a language code, and so on; the model gets the error and retries. The typed actions check the same limits, so the CLI and mini are covered too |

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
             middleware=[HumanInTheLoopMiddleware(...),   # real mode or --approve-all; outermost
                         ToolCallGuard(),                   # refuse repeats and unsourced ids, contain crashes
                         ModelCallLimitMiddleware(run_limit=12, exit_behavior="end")],
             checkpointer=InMemorySaver())
```

- `create_agent`'s recursion limit is now 9,999, so the call-limit middleware is the real step cap.
- **Conversations.** The interactive REPL (LangGraph and deepagents) keeps one thread per session,
  so when the agent asks for something (an email it may not guess, which applicants) your reply
  continues the same task. `new` starts a fresh conversation. Every user turn counts as user input
  for `ToolCallGuard`. The call cap is per run, i.e. one turn or one resume after an approval, each
  started by a person; a per-thread cap would end the conversation after 12 calls in total. A
  repeat of an earlier call in the same conversation is still refused, unless a write ran
  since and the earlier call was a read or failed: "show job 7001", "add Kafka to it", "show it
  again" reads it twice, and "publish it to LinkedIn" (fails: private), "make it public, then
  publish" retries the publish. A call that failed or was refused may also run again once the
  user has written since. A write that succeeded never runs twice. deepagents' shared
  `CallLedger` still refuses these repeats.
  `--task`, `compare_agents.py` and mini run every task on a fresh thread.
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

`agent_kit.interrupt_on()` decides which tools pause. `--approve-all` gates every tool that
changes data; reads never pause. In real mode, `score_candidates` and `candidate_insights` are
gated even without it, because they trigger calculations on VIRA; the reads (`find_talents`,
`generate_jd`) run straight through. Jeni's shortlist, reject, share and transfer-ownership tasks
(`ALWAYS_CONFIRM`) pause in every mode, mock included. The mode comes from
`vira_tools.configure()`, so a real-mode agent can't be built without the gate.
Each model turn raises one interrupt that batches all gated calls. The runner answers with one decision per call (`approve`, `edit` with new args, `reject`)
via `Command(resume={"decisions": [...]})` on the same thread. A rejected call never reaches VIRA
(tested). At the CLI, `ask_human` takes Enter or `y` to approve, `n` to reject, or a change in words
("only 12"), which a small confirmation model (`CONFIRM_MODEL`, default `gpt-4o-mini`) turns into
edited args. Only values the original call offered survive, the change is shown, and nothing runs
until Enter or `y`. Interrupts need a checkpointer; `InMemorySaver` covers a single process.

**Each card says what the call would do, in plain words.** `interrupt_on()` gives every gated tool
a `description`, `jeni_tools.summary(name, args)`: "Shortlist applications 5102 and 5103.",
"Transfer ownership of job 7003 to alice.johnson@example.com.", "Add user 802 to the hiring team
of job 7001 as a team member." That replaces the middleware's own "Tool execution requires
approval / Tool: … / Args: {…}". Any other tool gets its name and its values in words, never
JSON, and a long note is cut to 120 characters.

**A reviewer's edit or rejection stands until the user writes again.** HumanInTheLoopMiddleware
keeps the model's own call in its message and marks the decision only at the start of the tool
result ("Note: a human reviewer replaced this tool call …", "User rejected the tool call for …").
Asked to shortlist the top two, with the reviewer editing the card to one, gpt-5-mini then
proposed the other applicant in a new call: a second card for what the reviewer had just
removed. A prompt rule ("report what ran as the outcome … don't redo what they removed or
rejected, or offer to") didn't stop it: the demo's rehearsal still saw it in 2 of 3 runs. So it
is structural now. `agent_kit.overruled(messages)` names the tools a reviewer edited or rejected
since the user's last message; `interrupt_on()` gives HITL a `when` predicate so a new call to
such a tool gets no card, and `ToolCallGuard` refuses it ("the reviewer already edited or
rejected … that decision stands"). The next message from the user lifts it, so "shortlist 5103
too" works.

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
- **Approval:** `interrupt_on` gates every write with `--approve-all`, and score/insights in real
  mode; subagents inherit it (both tested).

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
  incomplete: 3–6× LangGraph's tokens per task (section 7).

Use it when a task needs many steps or several jobs, benefits from delegating per-job work to
isolated subagents, or should produce a written artifact (a report file in the virtual FS).

## 5. MCP server: `vira_mcp.py`

One stdio server exposes the tools to any MCP client. It uses `mcp==1.30` (`mcp.server.fastmcp`);
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
- Annotations: find, the match-id lookup and JD are `readOnlyHint`; score/insights trigger calculations that overwrite
  scores, so they're `destructiveHint` (and `idempotentHint`). They are hints, not enforcement.
- Confirm-gated tools are never exposed. MCP approval depends on the client (elicitation needs
  client support), so a write tool needs an approval design first.
- The server can't see the client's conversation, so `ToolCallGuard` doesn't apply. Instead:
  - in `--mode real`, only the reads (find, match-id lookup, JD) are listed unless you pass
    `--allow-side-effects`, and then
    approving each score/insights call is the client's job;
  - the process makes at most `--max-calls` VIRA calls (default 50), then answers `Refused`;
  - the input limits of §2 are part of the tool schemas, so an oversized call is an MCP error.

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

`compare_agents.py --repeat 3` on 2026-09-30 with gpt-5-mini against mock VIRA, after the
security hardening in §11 (at `6d27247`). PASS/FAIL is judged on the audit log, i.e. what actually
reached VIRA, not on the model's prose. Cells show passes out of 3 · median model calls · median
tokens · median wall time. Costs are litellm price-table estimates for all 18 runs of a runtime.

The four endpoints are stand-ins chosen to compare runtimes, not Jeni's production task set.

| task | check | mini | LangGraph | deepagents |
|---|---|---|---|---|
| find | one find-talents for job 123 | 3/3 · 2 · 4.0k · 18s | 3/3 · 2 · 2.9k · 7s | 3/3 · 2 · 9.7k · 8s |
| jd_ar | one generate-jd: lang=ar, given title + skills | 3/3 · 2 · 4.1k · 22s | 3/3 · 2 · 3.5k · 19s | 3/3 · 2 · 10.2k · 17s |
| score_insights | score 11,12, then insights for 11 | 3/3 · 3 · 5.8k · 22s | 3/3 · 3 · 4.2k · 8s | 3/3 · 3 · 14.8k · 10s |
| top_pick | score 11,12,13, then insights only for the top scorer (12) | 3/3 · 3 · 4.8k · 18s | 3/3 · 3 · 4.3k · 6s | 3/3 · 3 · 14.9k · 15s |
| id_trap | find, then stop (no match_ids existed; see the note below) | 3/3 · 4 · 9.8k · 55s | 3/3 · 2 · 3.6k · 11s | 3/3 · 4 · 21.5k · 36s |
| no_title | no call: title missing, don't invent one | 1/3 · 12 · 37.8k · 169s | 3/3 · 1 · 1.7k · 5s | 3/3 · 1 · 5.0k · 12s |
| **total** | | **16/18** · ~$0.167 | **18/18** · ~$0.037 | **18/18** · ~$0.096 |

`id_trap` was measured before the assumed `get_match_id_from_profile_id` lookup existed. Its
check is now find → look up match ids for job 123 → score those match ids, so this row needs a
re-run.

No wrong-kind id reached VIRA in any run, and the guard had none to refuse. The grounding check
flagged two values, both in deepagents' `no_title` answer: "5+ years" and "~300 words" in the
example it offered while asking for the title.

Earlier single runs on 2026-09-23, before the hardening and without `top_pick`: run 1 mini 5/5
(~$0.033), LangGraph 4/5 (~$0.016), deepagents 4/5 (~$0.025); run 2 mini 3/5 (~$0.069),
LangGraph 5/5 (~$0.011), deepagents 4/5 (~$0.045).

**Multi-job check** (by hand, not in the harness): "For jobs 101, 102 and 103, find suggested
talents and write a short report with the top 2 talents for each job and their match scores",
twice per runtime. Both runtimes made one `find_talents` per job and reported the right profiles
and scores. LangGraph: 4 model calls, ~6.4k tokens. deepagents: 6 calls, ~31k tokens; it wrote
todos but delegated nothing. MockVira answers only the first id in `job_ids` and returns the same
profiles for every job, so this can't catch one job's talents being reported under another.

What the runs showed:

1. **A typed schema is part of the prompt.** In the first single run both typed runtimes failed
   `jd_ar`. They filled `job_function`, `industry`, `other_requirements` and extra skills nobody
   asked for, re-called `generate_jd`, and then wrote the Arabic JD themselves. mini didn't,
   because `commands.md` never documents those flags. The fix: the field descriptions now say
   "only what the user named; leave empty otherwise", and the prompt says to relay tool results.
   Both typed runtimes now pass it 3/3.
2. **mini's failures are the bash failure modes.** Asked for a JD with no title, it has no way to
   ask the user, since every turn must be a command. In both failed runs it echoed "please provide
   the job title" until the 12-call limit. Nothing reached VIRA; before the hardening, one run sent
   `--job-title ""`. The typed runtimes answer that case in one call. The earlier typo crash
   (`--job-ids 123..`) now comes back as a clear error (§2), though no run hit it this time.
3. **deepagents' subagents got past the duplicate guard.** In the second single run, the main
   agent delegated to `jd-writer` twice after the mock's placeholder JD, and each subagent re-sent
   the identical call, because each has its own message history. This is fixed with a
   thread-scoped `CallLedger` shared by the main agent and every subagent (tested). deepagents
   still tends to retry and delegate when a result looks incomplete, varying the arguments
   instead of repeating them.
4. **Cost and speed.** LangGraph was cheapest in every cell and fastest in all but `jd_ar`
   (19s against deepagents' 17s): 1.7–4.3k tokens and 5–19s per task. deepagents starts at about 5k tokens for a single call because of its
   filesystem, todo and task tool schemas, and used 3–6× LangGraph's tokens per task (about 2.6×
   the cost overall). mini needs extra calls just to finish (echo SUMMARY, echo COMPLETE), cost
   about 4.5× LangGraph, and was the slowest in every cell.

### Jeni's own tasks (`--suite jeni`)

`compare_agents.py --suite jeni --repeat 3` on 2026-09-30 with gpt-5-mini against `mock_jeni`
(§12): seven Jeni requests (`jeni_eval.py`). Writes must match exactly, reads are free. mini
isn't in this suite, because its CLI doesn't cover Jeni's tasks.

| task | check | LangGraph | deepagents |
|---|---|---|---|
| job_details | one get_single_job_details for job 7001 | 3/3 · 2 · 7.1k · 5s | 3/3 · 2 · 13.6k · 4s |
| create_then_skill | create the job, then add Spark using the new job's id | 3/3 · 3 · 10.4k · 4s | 3/3 · 3 · 20.4k · 6s |
| assign_team_member | search for Bob, then add user 802 as a team member (role 5), no transfer | 3/3 · 3 · 10.5k · 5s | 3/3 · 3 · 20.3k · 6s |
| shortlist_top2 | read job 7001's applications, then shortlist the top two (5102, 5103) | 3/3 · 3 · 10.8k · 5s | 3/3 · 3 · 20.7k · 6s |
| add_candidate | one create_application_to_job on job 7001 with the name and email given | 3/3 · 2 · 6.9k · 4s | 3/3 · 2 · 13.5k · 4s |
| share_no_email | no share: the recipient's email wasn't given, so ask for it | 3/3 · 1 · 4.2k · 6s | 3/3 · 1 · 7.3k · 5s |
| unsupported | nothing changed: sharing a job isn't supported | 3/3 · 1 · 4.0k · 5s | 3/3 · 1 · 7.4k · 6s |
| **total** | | **21/21** · ~$0.060 | **21/21** · ~$0.101 |

- **Real outputs chained in one request.** Every run took the new job's id from `create_job`'s
  result, Bob's user id from `search_users`, and the top two applicants from `get_applications`,
  and passed them on. v1 can't do the first of these in one request (§12).
- **No guessing.** Every run chose role 5 for "team member" and none transferred ownership. With
  no recipient email, every run asked for one instead of sharing, and "share job 7001 on
  Facebook" was declined with what is supported instead. The guard never had to refuse a call.
- **Cost.** The 22 tool schemas put LangGraph at 4–11k tokens per request, against 1.7–4.3k with
  the five sample tools. deepagents used about twice LangGraph's tokens and 1.7× its cost, with no
  gain on these single-job requests.
- **Grounding** flagged 7 values per runtime, none of them ids: `search_users`' `limit: 10`,
  `role_id: 5` (from "team member" via the field description), a page size of 50, and a
  "1–2000 characters" hint in an answer.
- Not covered yet: requests over several jobs, and a clarifying answer continuing the same
  request (the runners start a new thread per task).

## 8. Adding a VIRA endpoint

1. `recruiter_cli.py`: add a typed action that calls `execute()` (path plus the query/body split),
   and a CLI subcommand that parses flags and calls it.
2. `commands.md`: document the subcommand for mini. Add it to `mini_policy.SUBCOMMANDS`, and to
   `mini_policy.SIDE_EFFECTS` unless it is a pure read; a test checks both against the CLI and
   `vira_tools.READ_ONLY`.
3. `vira_tools.py`: add a typed function (its signature and docstring are the model's contract)
   and append it to `TOOLS`. Add it to `READ_ONLY` if it is a pure read.
4. Writes, notifications and anything irreversible: add the CLI name to `NEEDS_CONFIRM`. Today that
   makes the tool unreachable from `vira_tools` and hides it from MCP, so decide the approval
   design first (follow-ups).
5. Bounds: check the new arguments with `_input_problem()` in the typed action, and put the same
   `MAX_*` / `LANG_RE` limits in the `vira_tools` signature.
6. New sensitive response fields: extend `PII_KEYS` (or the `_PII_WORD` / `_PII_NAME` patterns
   beside it) and add the key to `test_pii_keys_are_masked_by_name_and_pattern`.
7. `mock_vira.py`: add a mock reply. Tests: add a parity case to `tests/test_recruiter_cli.py`
   and the schema expectation to `tests/test_agents.py`.
8. `grounding.py`: add the name to `VIRA_TOOLS` so traces and `compare_agents` grounding include
   its calls, and any new id argument to `ID_KINDS` so `ToolCallGuard` checks it.
   `run_deepagent.py`: subagent tool lists are explicit, so add the tool to the subagent that
   needs it.

## 9. Running it

```bash
uv pip install -r requirements-dev.txt        # or, exact pins with hashes:
# uv pip install --require-hashes -r requirements.lock.txt
.venv/bin/pip-audit -r requirements.lock.txt --disable-pip      # known vulnerabilities in the pins
.venv/bin/python -m pytest -q                                   # offline, 335 tests

JENI_MODE=mock python run_mini.py                                         # mini baseline, no shell
python run_langgraph.py --tools vira --task "Find potential talents for job 123"   # mock VIRA by default
python run_langgraph.py --tools vira --approve-all                        # approve, change or reject each write
python run_langgraph.py --tools jeni                                      # one conversation: answer its questions; 'new' starts over
python run_workflow.py --app-ids 11,12,13 --top 2
python run_deepagent.py --tools vira --task "For jobs 101 and 102, find talents and write /report.md"
python run_langgraph.py --tools jeni --task "Assign job 7001 to Bob as a team member"   # Jeni's tasks (§12)
python jeni_tools.py                                                      # list them: read or write, and their fields
python run_langgraph.py --task "Add Kubernetes to the backend engineer job"      # jeni_db: looks the job up first (§13)
demo/run.sh                                                               # the demo (§14): chat on :3000, data panel beside it
.venv/bin/python -m demo.rehearse                                         # play the demo script against it and check the data
python vira_mcp.py --mode mock                                            # for MCP clients
python vira_mcp.py --mode real --max-calls 20                             # reads only; add --allow-side-effects for score/insights
python compare_agents.py --out traces/report.md                           # live LLM, mock VIRA only
python compare_agents.py --model gpt-4o-mini --repeat 3 --json traces/runs.json   # traces for the page
python run_langgraph.py --tools vira --mode real --task "…" --trace-json traces/real.json   # one real run, local file
```

LangSmith tracing is forced off in the new runners: `set_tracing(False)` sets all four of
`LANGSMITH_TRACING_V2`, `LANGSMITH_TRACING`, `LANGCHAIN_TRACING_V2` and `LANGCHAIN_TRACING`
(langsmith reads `…_V2` first) and clears langsmith's cached lookup. `--trace` allows it.
Traces would carry prompts and (masked) tool results off the machine.

Local traces and reports (`--trace-json`, `--json`, `--out`) are written owner-only (0600) with
emails and phone numbers scrubbed, since they hold the task text and final answers. `traces/` and
`*.jsonl` are gitignored, along with `.env.*`, key files and credential stores.

## 10. Follow-ups

- **Approval design for write tools.** Real mode already pauses score/insights (§3); a write
  tool in `NEEDS_CONFIRM` still can't run anywhere. Proposal: the HITL approval *is* the confirmation.
  `vira_tools` passes `confirmed=True` only for tools that `agent_kit` gates with `interrupt_on`,
  asserted at build time for the main agent and every subagent. MCP keeps them off, or uses
  elicitation.
- **get-match-id on VIRA:** `get_match_id_from_profile_id` is an assumed endpoint that only mock
  VIRA answers. Real mode sends `POST get_match_id_from_profile_id` with
  `{"job_id", "profile_ids"}` and gets VIRA's error until the backend adds it. Confirm the real
  path and body, then re-run `id_trap` for §7.
- **Durable approvals:** `SqliteSaver` (langgraph-checkpoint-sqlite) so a paused run, and a
  REPL conversation, survives a restart. Encrypt checkpoints at rest (`EncryptedSerializer`).
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
| Candidate PII reaches the model (and the model provider) | `_mask_pii`: emails in results become `pii_vault` tokens (`<email:…>`, a keyed hash; the address kept AES-GCM-encrypted in this process only) and other personal values are redacted; normalised keys matched against `PII_KEYS` and word patterns (`first_name`, `phone_number`, `linkedin_url`, …, but not `job_name_similarity`); emails and phone numbers replaced inside every string, including a non-JSON `raw` reply; uuids are kept whole (their digit groups looked like phone numbers to 6% of random uuids). Names in free text are not detected | `test_pii_keys_are_masked_by_name_and_pattern`, `test_other_keys_are_kept`, `test_emails_and_phones_inside_text_are_masked`, `test_non_json_reply_text_is_scrubbed` |
| Cost or DoS amplification on VIRA's LLM endpoints; oversized text injected into VIRA's own JD prompt | Limits in the typed actions (`_input_problem`) and in the tool schemas: ≤50 positive ids, ≤200 characters per text, ≤30 list entries, `lang` a language code | `test_bad_input_is_refused_before_anything_is_sent`, `test_the_cli_reports_a_typo_in_ids`, `test_schemas_reject_oversized_or_malformed_input` |
| An injected prompt sprays invented ids at VIRA | `ToolCallGuard` refuses id arguments found in neither the user's messages (any turn) nor an earlier result, before VIRA is called; reviewer edits count as user input; `grounding.summary` reports them as `ungrounded_blocked` | `test_an_invented_id_is_refused_before_vira`, `test_ids_from_an_earlier_result_are_not_invented`, `test_edit_runs_the_reviewers_args`, `test_an_invented_id_the_guard_refused_counts_as_blocked` |
| An agent triggers calculations on VIRA (recal_briq) without a person | Real mode gates score/insights with a LangGraph interrupt in the LangGraph and deepagents runners (subagents inherit it) and with a y/N in mini; `interrupt_on()` reads the configured mode | `test_real_mode_always_gates_the_calls_that_change_vira`, `test_a_real_mode_agent_pauses_before_scoring_but_not_before_a_read`, `test_real_mode_subagents_pause_before_scoring`, `test_real_mode_side_effects_wait_for_approval` |
| An MCP client (or an injected prompt in it) drives VIRA under the service identity | Real mode lists only the reads (find, match-id lookup, JD) unless `--allow-side-effects`; a per-process budget (`--max-calls`, default 50); score/insights annotated `destructiveHint` | `test_real_mode_lists_only_reads_unless_side_effects_are_allowed`, `test_the_server_stops_calling_vira_after_its_budget`, `test_tools_annotations_and_masked_results` |
| Secrets, audit logs or traces committed by accident | `.gitignore` covers `.env`/`.env.*` (except `.env.example`), `*.env`, key and certificate files, `credentials*`, `secrets*`, `.netrc`/`.npmrc`/`.pypirc`, `*.jsonl` and `traces/` | `git ls-files -ci --exclude-standard` is empty |
| Traces or reports readable by other users, or carrying the task's PII | `agent_kit.write_private()`: 0600 files (a missing directory is created 0700), emails and phone numbers scrubbed | `test_trace_json_is_owner_only_and_scrubbed` |
| Model or VIRA text drives your terminal (ESC/OSC sequences, clipboard writes, bidi tricks in an approval prompt) | `terminal.printable()` on every trajectory, approval prompt, virtual-file and workflow print, in all runners | `test_printable_strips_terminal_escapes_and_bidi_controls`, `test_trajectories_print_without_escapes` |
| A mistyped edit at the approval prompt crashes the task | `ask_human` asks again until it gets a JSON object | `test_a_bad_edit_is_asked_again_not_a_crash` |
| A tampered or vulnerable dependency | `requirements.lock.txt` pins all 139 packages with hashes (`--require-hashes` installs); `pip-audit` is in the dev requirements. pyjwt is at 2.15.1 for CVE-2026-101918 (an uncaught `RecursionError` on a deeply nested token); no known vulnerabilities on 2026-10-01 | `pip-audit -r requirements.lock.txt --disable-pip` |
| Prompts and tool results shipped to LangSmith | All four tracing variables are set, and langsmith's cached lookup cleared | `test_tracing_stays_off_even_with_langsmith_tracing_v2_set` |
| Jeni: a guessed candidate email or share recipient reaches VIRA (a CV sent to the wrong person) | `ToolCallGuard` refuses a `USER_ONLY` value (candidate name and email, new owner's email, share recipients) that isn't in the user's words, without echoing it; masked values from results fail the email pattern | `test_an_email_the_user_never_gave_is_refused`, `test_an_email_the_user_gave_is_used`, `test_invalid_input_never_reaches_vira` |
| A token sends a CV or a job to the wrong person (a candidate's email from a result used as a recipient) | Only a colleague's token (from the user directory) or "me" (the signed-in user, set by the host) may stand in for a share recipient or new owner; a candidate's token, an unknown token or "me" with nobody signed in is refused, and candidate fields take a typed address only. Tokens are swapped back to addresses in `recruiter_cli._call`, after the guard and the approval; the audit log stays redacted | `tests/test_pii.py` |
| Jeni: an agent changes jobs, applications or ownership without a person | Real mode gates all 16 tasks that change data; only the 6 reads run unasked | `test_real_mode_gates_every_task_that_changes_data`, `test_a_real_mode_agent_asks_before_a_write_but_not_before_a_read` |
| The engine's `xrtoken` leaks or is sent elsewhere | `VIRA_XRTOKEN` is read from the environment and sent only as the `xrtoken` header to `VIRA_ACTUAL_LOCATION`, never to the other endpoints; the audit log records commands and masked bodies, not headers; no redirects are followed and proxy variables are ignored | `test_task_groups_go_to_the_engine_with_only_the_users_xrtoken` |
| A task group sent with a missing or wrong engine setting | No location, plain http to a non-loopback host, or an empty token: nothing is sent, and the error names the setting, never the host | `test_the_engine_fails_closed_without_its_settings` |
| An agent redoes what a reviewer removed (the applicant edited off a shortlist, a rejected change with other args) | `agent_kit.overruled()`: until the user writes again, a new call to a tool the reviewer edited or rejected gets no card (HITL's `when`) and `ToolCallGuard` refuses it | `test_a_reviewers_edit_stands_until_the_user_writes_again`, `test_a_rejection_stands_and_gets_no_second_card` |
| Jeni: candidate names in task payloads reach the audit log, or the creator's name reaches the model | `_mask_pii` masks a `field_value` whose `field_name` is sensitive, and creator/owner names; the model gets only the sub-task result | `test_candidate_details_are_masked_in_the_audit_log`, `test_creator_and_owner_names_are_masked`, `test_the_model_sees_only_the_sub_task_result` |
| Jeni: a collaborator silently added as administrator | `role_id` has no default and takes only 1 (administrator) or 5 (team member); `is_private` is set by the task | `test_role_has_no_default_and_visibility_is_set_by_the_task` |
| Jeni's internal task catalog published with the code | `config/*` is gitignored (only its README is tracked); `jeni_tools` reads the catalog from there or `JENI_TASKS_FILE`; the tests use a synthetic fixture and pass without the real file | `test_the_catalog_is_read_from_the_configured_file`, `test_a_missing_catalog_is_a_clear_error_and_only_for_jeni` |
| Jeni: a user id used as an application id, or an invented user id | `ID_KINDS` covers `user_ids` and `app_id`, and VIRA's camelCase keys (`userId`, `appId`, `jobId`) | `test_a_user_id_is_not_an_application_id`, `test_an_invented_user_id_is_refused`, `test_camel_case_result_keys_count_as_id_kinds` |
| jeni_db: the model reads another company's data | The company comes from the runner and is bound when the tools are built; no tool takes it as an argument, and a query without it is an error. Every query filters by it, `list_job_applications` included (staging holds 169 companies) | `test_the_model_supplies_search_terms_and_ids_but_never_the_company`, `test_a_query_without_a_tenant_is_an_error_not_a_guess`, `test_another_companys_job_lists_no_applications_and_fails_validation`, `test_the_real_application_list_is_scoped_to_the_tenant` |
| jeni_db: people's emails from the database reach the model | db results go through `_mask_pii` like VIRA's; a user search's email labels arrive redacted | `test_user_search_results_reach_the_model_masked` |
| jeni_db: a search term widens the query (`%`, `_`) | Terms are escaped before `ILIKE`, and every value is a bound parameter | `test_the_real_title_search_is_literal_and_asks_for_one_more_row`, `test_the_real_user_search_is_literal_and_tenant_scoped` |
| jeni_db: a database error crashes the run or puts a host into context | `ToolCallGuard` covers the db tools: an exception becomes `tool failed (<type>)` | `test_a_failing_query_is_an_error_result_not_a_crash` |

## 12. Jeni's own tasks: `jeni_tools.py`

Sections 1–11 use four sample AI endpoints, chosen to compare runtimes. Jeni v1, the prototype,
runs a different set: the 22 recruiter tasks in its `XAGENT_SUBTASK_API_CONFIG_SCHEMA`.
`jeni_tools.py` turns each task into a typed tool, and `--tools jeni` gives them to the LangGraph
and deepagents runners.

The catalog is internal, so it isn't in the repository. It is shared as a file, read from
`config/jeni_tasks.json` (gitignored) or wherever `JENI_TASKS_FILE` points, and only when a Jeni
tool is first needed: nothing else depends on it, and without it `--tools jeni` stops with a
message saying where to put it. `python jeni_tools.py --from-js tasks.js` builds it from v1's
file, and `python jeni_tools.py` checks it. The offline tests use a synthetic 15-task catalog in
v1's format (`tests/fixtures/jeni_tasks.json`).

```
v1  request → one LLM call plans every task as JSON ({{output_from:…}} for later values) → VIRA runs the group
v2  request → agent loop: one tool call = one task → VIRA runs a one-task group → sub-task result → next step
```

| Area | Read only (6) | Changes data (16) |
|---|---|---|
| Jobs | `get_single_job_details` | `create_job`, `edit_job`, `add_job_skills`, `remove_job_skills`, `clone_job`, `make_job_private`, `make_job_public`, `make_job_closed`, `make_job_open`, `publish_job_to_linkedin` |
| People on a job | `search_users` | `add_job_collaborators`, `transfer_job_ownership` |
| Applications | `get_applications`, `get_single_application_details` | `create_application_to_job`, `shortlist_multiple_application`, `reject_multiple_application`, `share_application` |
| Candidates | `get_suggested_candidates_for_a_job`, `get_self_sourcing_candidates_for_a_job` | |

How a call works:
- **Schema from the catalog.** Mandatory fields are required, and types follow tasks.js
  (`NUMBER_FIELDS`, `STRING_ARRAY_FIELDS`, …) with v2's limits: positive ids, at most 50 per list,
  200 characters per value (2,000 for descriptions and messages), email patterns, `role_id` 1 or 5,
  and no unknown fields. `run()` validates again, so a bad call never reaches VIRA whoever makes it.
- **One task per group.** A call sends a task group holding only that task, in v1's payload
  format (`handleGenerateViraPayload`), through `recruiter_cli.execute()`: audit log, PII mask,
  mode set by the host. Real mode sends it to the VIRA engine (§15); `mock_jeni.py` answers it
  in the shape of a completed VIRA reply.
- **Values typed in lower case are tidied** before they are sent, and on the approval card
  (`tidy_case`): job titles, countries, regions and candidate names in title case ("backend
  engineer" became the job "backend engineer" in the demo; now "Backend Engineer"), each skill
  in its usual spelling ("sql" → "SQL", "node.js" → "Node.js"). A value with any capital is left
  exactly as typed ("iOS Developer", "Maya de Souza"); emails never change. The field
  descriptions ask the model for the same, so the tidy-up is a safety net.
- **Only the sub-task result reaches the model**: `{status, task_status, failed_reason, result}`.
  The group's uuids, timestamps and creator name stay out. A task can be `completed` with items in
  `failedArr` (one of two collaborators not found), and the prompt says to read it.
- **Approvals.** Real mode and `--approve-all` pause all 16 writes, never the 6 reads.
  Shortlist, reject, share and transfer ownership pause in mock mode too.
- **Asking.** `RULES` tells the model to ask only for what no tool can give it, not to have the
  user confirm values they already gave, to do each step once it has what that step needs, and,
  after a failure, to report the reason rather than offer a retry or another tool's workaround.
  To compare or rank applicants it points at `get_applications`, which returns their match
  scores in one call (the rehearsal saw four one-by-one reads otherwise).
- **Personal values as tokens** (§16): an email in a result reaches the model as `<email:…>`; a colleague's token or "me" may be a recipient, a candidate's may not.
- **Ids and personal values.** Ids must come from the user or an earlier result and keep their
  kind (a `userId` from `search_users` can't be sent as `app_ids`). Candidate names and emails
  must be exactly what the user wrote. A new owner or share recipient may also be a colleague's
  `<email:…>` token from the user directory, or "me" (§16); a candidate's token never.

What v1's catalog shows, and what v2 changes (proposals for engineering):
- Every task has one sub-task, `level: 1` and `task_output: []`, so v1's rule for chaining a
  later task onto an earlier one's output never applies, and v1 strips the `{{output_from:…}}`
  placeholders before sending. "Create a job and add Spark to it" can't be one request in v1. In
  v2 the agent reads the new job's id from `create_job`'s result and passes it on.
- `add_job_collaborators.role_id` defaults to 1 (administrator) in v1. v2 gives it no default,
  and the model asks when the user didn't say.
- `make_job_private` and `make_job_public` both take `is_private`, so v1's model could send "make
  public" with `is_private: true`. v2 sets it from the task.
- `publish_job_to_linkedin`'s description tells the model to ask whether the job is already open
  and public before publishing. VIRA checks that itself (the mock returns "Job must be open and
  public before publishing to LinkedIn"), so the question could go: the agent would publish and
  report the refusal. v2 keeps the question for now, as the catalog asks.
- Kept as v1 has them, but worth fixing in the catalog: `create_job`'s description names title,
  skills and experience as the minimum while only the title is mandatory; `get_applications`
  calls `job_id` mandatory but marks it optional; `task_get_jobs` is commented out, so no task
  finds a job by name.
- v1's response schema accepts any `task_name` and any field in any task. v2's tools are exactly
  the 22 tasks with their own fields.

The mock (`mock_jeni.py`) holds synthetic jobs (7001–7003), users (801–803), applications
(5101–5104, 5201–5202) and suggested and self-sourcing candidates. By default it is stateless:
every task group starts from those fixtures, a created job's id comes from its title
(`created_job_id`), and changes are reported but not remembered, so repeated and parallel runs
agree (tests, `compare_agents`). It enforces one VIRA rule the catalog states in prose:
publishing to LinkedIn needs an open, public job.

`mock_jeni.remember_changes()` switches it to one `State` that lives across calls, for the demo
(§14): skills, visibility, status, teams, owners, stages and LinkedIn postings stick, a created
job or application gets an id of its own and is found by the db lookups
(`State.db_fixtures()` lists are refilled in place), and `reset()` goes back to the fixtures.
The kept State also logs every sub-task it ran (`activity`) with ids, skills, titles and flags
only, never names, emails or search text; `snapshot()` is what the demo's data panel shows.
`forget_changes()` makes it stateless again.

Open points:
- **People are masked in results,** so after `search_users` the agent picks a person by id. With
  two matches it can't tell them apart by name. Colleague directory data may deserve a different
  policy from candidate data.
- **Clarifying answers start a new thread** in the runners (one thread per task), so asking and
  then continuing isn't exercised yet.
- **22 tool schemas cost tokens:** a live "assign job 7001 to Bob as a team member" took 3 model
  calls and about 10.5k tokens on LangGraph.

## 13. Database lookups: `--tools jeni_db`

Jeni's tasks take ids, but people name things: "the data scientist job", "Bob". v1 looks names
up in Jeni's database before it plans (`handleSearchJobs`, `handleSearchUsers`). `jeni_db` gives
the agent the same lookups as read-only tools next to the 22 tasks, so it turns a name into an
id, checks the ids the user typed, and only then acts. It is the default toolset of
`run_langgraph.py`.

| Tool | Does | Returns |
|---|---|---|
| `find_job_by_title` | Jobs whose title contains the text, newest first | one: `{"status": "resolved", "job_id"}`; several: `ambiguous`, up to 10 candidates labelled with their open date; none: `not_found` |
| `find_user` | Active recruiters (role 4) by name or email | the same, with `user_id`; the labels are emails, so they arrive masked |
| `list_job_applications` | A job's applications (up to 200) | the app ids, in `applications` and `selection.candidates` |
| `validate_job_id(s)`, `validate_app_ids`, `validate_email(s)` | Checks ids or emails the user gave | `resolved`, or `not_found` with `invalid_values` |

Files:
- `db_lookup.py`: v1's Node queries on asyncpg (`handle_search_jobs`, `handle_search_users`),
  with bound parameters and a `limit`.
- `db_queries.py`: each lookup as a `QueryTool` returning the resolved / ambiguous / not_found /
  validated contract, backed by an asyncpg pool (`build_db_queries`) or by in-memory fixtures
  (`fake_db_queries`) with the same 0 / 1 / many mapping.
- `db_tools.py`: the `QueryTool`s as LangChain tools, and the prompt's `RULES`.

How a call works:
- **The company comes from the runner.** `context = {"auth_profile": {"company_id": …}}` is
  bound when the tools are built (`--company-id` here, the authenticated profile in
  production). The tool schemas hold only search terms and ids. A query without a company
  returns an error, and every query filters by it: a production DSN holds one company, but
  staging holds 169.
- **Async tools on one event loop.** asyncpg ties a pool to the loop that created it, so
  `run_langgraph.py` runs `jeni_db` inside one `asyncio.run(amain())`: it opens the pool, checks
  it with `SELECT 1`, builds the agent and drives every turn with `ainvoke`
  (`agent_kit.arun_task`, `arepl`). The deepagents runner is synchronous and doesn't offer
  `jeni_db`.
- **The database follows `--mode`, like VIRA.** Mock mode answers the lookups from mock VIRA's
  own jobs, users and applications (`mock_jeni.db_fixtures`), so every id they return is one the
  Jeni task mock knows; a DSN is ignored, with a note. Real mode connects to `--dsn` (default
  `$TRON_POSTGRES_DSN`) and stops with a message without one. Mixing them fails every write: a
  staging job the database confirms is "not found" by the mock.
- **What the model sees.** `resolved` is flattened to the id; `ambiguous` keeps the candidates, each labelled with its name and id and, when known, its open date ("Senior Backend Engineer (job 7001, opened 2026-08-21)"); the rules say to list them that way, since a copy of a job has the same title, and to accept an answer by name, detail or id. Before, the agent asked "tell me the job_id: 7001 or 8001" with no names. The prompt says
  to ask the user rather than pick. A search with more than 10 matches says
  so and asks for the id or a narrower title. Results go through `_mask_pii`. `%` and `_` in a
  search term match literally. A title that finds nothing is tried once more without a trailing
  "job", "role" or "position" (models often search for "data scientist job").
- **What the prompt adds.** Look names up instead of asking for ids; validate ids the user typed
  (ids a tool returned are valid already); "the applicants" of a job means all of them, from
  `list_job_applications`; on `ambiguous`, ask which one and nothing else.
- **Guarded like the tasks.** The db tools are in the toolset's names, so `ToolCallGuard` refuses
  invented ids and exact repeats and turns a query exception into an error result. They are
  reads, so real mode doesn't pause for them.

On TRON staging (company 5143), "data scientist" matches 27 jobs, nine of the first ten titled
exactly "Data Scientist"; the open date is what tells them apart.

Live, on staging with gpt-5-mini ("add python, sql to data scientist job, shortlist the
applicants for it and publish the job to linkedin", then the job id): before the prompt rules,
the agent answered the id with the same two questions again ("all or specific applicants?", "is
the job public?") and changed nothing. With them, it adds the skills, lists the 11 applicants and
shortlists them in that turn, and holds only the LinkedIn step for the catalog's question. Over
four samples of the first turn, all four listed the candidates; three asked only which job, one
also asked the catalog's open-and-public question.

Open points:
- **Jeni's tasks on real VIRA** go to the engine now, which queues them and runs them later; see
  §15 for what works and what doesn't yet.

## 14. The demo: `demo/` on `langgraph dev`

A demo of Jeni v2 needs three things the terminal runners don't give it: a chat window with
approval cards, a way to see that a change really happened, and data that keeps the change. The
agent is the same `create_agent` graph as `run_langgraph.py`; LangGraph's own API server
(`langgraph dev`, from `requirements-demo.txt`) serves it, with threads, streaming and resume.

```
browser ─ chat UI ───────────── langgraph dev (127.0.0.1:2024) ─ graph "jeni" = demo/jeni_graph.make_graph
        └ data panel (/demo) ─┘        │                           jeni_db tools → recruiter_cli.execute (mock)
                                       └─ demo/app.py: /demo, /demo/state, /demo/reset ─ mock_jeni's kept State
```

| File | Role |
|---|---|
| `langgraph.json` | Graph `jeni` from `demo/jeni_graph.py:make_graph`, the panel's routes from `demo/app.py:app`, env from `.env` (the model key; nothing reaches VIRA). |
| `demo/jeni_graph.py` | `build()`: tracing off, `vira_tools.configure("mock")`, `mock_jeni.remember_changes()`, then `jeni_db` on the kept State's fixtures (`fake_db_queries`, company 5143) and `run_langgraph.build_agent(gate_writes=GATE_WRITES, own_checkpointer=False)`, with `GATE_WRITES = False`. `make_graph()` is the async factory: it builds once, on a worker thread. |
| `demo/app.py`, `demo/panel.html` | The data panel: jobs with their status, visibility, LinkedIn posting, skills, team and applicants, and "What reached VIRA", every sub-task the mock ran, reads and writes. It polls `/demo/state` every second and highlights what changed; **Reset data** posts `/demo/reset`. `/demo/prompts` gives the chat its starter cards. |
| `demo/script.py`, `demo/SCRIPT.md` | The script: eight acts, each with its prompts, what to do at each approval card, and a check over the data. `SCRIPT.md` is the presenter's copy, with talking points and recovery; a test keeps its prompts identical. |
| `demo/rehearse.py` | Plays the script against the running server through `langgraph_sdk`, the API the chat uses, and checks the data after each act. |
| `demo/run.sh`, `demo/README.md` | The launcher (preflight, UI build when needed, both servers, the agent built and the data reset before anyone types) and the setup guide. |
| `requirements-demo.txt`, `requirements-demo.lock.txt` | `requirements.txt` plus `langgraph-cli[inmem]==0.4.32`. The lock keeps every pin of `requirements.lock.txt` and adds 27 packages (langgraph-api 0.15.1, the in-memory runtime, uvicorn extras, OpenTelemetry), all with hashes; no known vulnerabilities on 2026-10-01. |

What differs from `run_langgraph.py`, and why:
- **Mock only, with memory.** The mode is set before any tool exists, and the db lookups answer
  from the mock's own data, so their ids agree with the tasks (§13). `remember_changes()` makes
  "make it public", then "publish it" work, and puts a created job where `find_job_by_title`
  finds it (§12).
- **Routine writes run straight away; high-stakes ones ask.** With `GATE_WRITES = False` the demo
  gets mock mode's rule: only the `ALWAYS_CONFIRM` tools (shortlist, reject, share an
  application, transfer ownership) pause, as they do in every mode, and reads and lookups never
  do. The script needs nothing more: its two cards with a decision (a rejected transfer, an
  edited shortlist) are both `ALWAYS_CONFIRM`. `GATE_WRITES = True` makes `build_agent(gate_writes=True)`
  pass `mode="real"` to `interrupt_on()`, so all 16 writes pause, as on real VIRA.
- **No checkpointer of its own.** `langgraph dev` refuses a graph that brings one ("persistence is
  handled automatically by the platform") and keeps threads itself, in memory, flushed to
  `.langgraph_api/` (gitignored). `build_agent(own_checkpointer=False)`.
- **The closing rules are written for a chat window.** `SYSTEM_PROMPT` ends with the grading format
  `SUMMARY: <what succeeded> | <what failed> | <why>` and a `STATUS: done` / `STATUS: needs_user`
  last line, which only the terminal REPL reads (to remember finished tasks); in the chat both
  showed as stray lines. The demo replaces the first with "the outcome first, in a sentence or
  two … don't recount the tools you called … say what succeeded, what failed or is missing, and
  why", and the second with its one instruction that isn't about the status line ("if more tool
  calls are needed, make them instead of replying"). It fails at build time if either rule it
  replaces has changed (`CHAT_REPLACEMENTS`).
- **Built off the event loop.** The server calls the factory inside its event loop, and runs
  there under blockbuster, which raises on blocking I/O (reading the catalog, for one). The
  factory is `async` and builds with `asyncio.to_thread` once; the panel's endpoints are plain
  functions, so Starlette runs them on its thread pool and the State's lock is never waited on
  in the loop. Tools are sync and already run on worker threads.

The chat UI (`demo/ui/`) is LangChain's [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui)
(MIT) at upstream commit `cf72cb0`, vendored. It talks to the server with `@langchain/langgraph-sdk`,
streams the thread, renders tool calls and results, and turns each HITL interrupt into an
approval card (approve, edit the args, or reject with a reason). Local changes, all in one commit
after the import:
- Jeni's name, mark and page title; no GitHub links; no file upload (Jeni doesn't read files).
- **Show data** / **Hide data**: the server's `/demo` panel in the right-hand column, open by
  default.
- **Edited arguments keep their types.** Upstream resumes with each edited value as the text of
  its box, so a reviewer who edited `app_ids` to `[5102]` sent the string `"[5102]"`, which the
  tool's schema refuses. The card now keeps the proposed args, and `restoreArgTypes()` parses an
  edited value back to JSON wherever the proposed one wasn't a string; a value that doesn't parse
  is an error on the card. Found by driving the real UI in a headless browser; the API-level
  rehearsal couldn't see it, since it sends JSON.
- `pnpm.overrides` for two low-severity advisories in build tooling (`@babel/core`,
  `postcss-selector-parser`); `pnpm audit`: no known vulnerabilities on 2026-10-01.
- **No internal workings on screen** (2026-10-02). Tool calls, their arguments and raw results
  are never rendered (`messages/ai.tsx`), and the "Hide Tool Calls" switch and the API host badge
  are gone. The approval card leads with the server's plain-language summary (§3) under "Needs
  your approval", drops the thread id, Studio, State (the raw thread), Description and Mark as
  Resolved, and shows the call's fields only after **Change details**. Checked in headless
  Chromium: the page has no tool names, JSON or default card text; editing the shortlist to
  `[5102]` shortlists 5102 only; a rejected transfer never reaches the mock.

Security, beyond §11 (each has a test in `tests/test_demo.py` unless noted):

| Threat | Control |
|---|---|
| The demo reaches VIRA or a real database | `build()` fixes mock mode before the tools are built, whatever the process had, and the db tools are `fake_db_queries` over the mock's own data |
| Anyone on the network drives the agent or reads its threads (`langgraph dev` has no auth) | It binds `127.0.0.1` by default; don't pass `--host 0.0.0.0` or `--tunnel`. Show it by sharing your screen (not tested) |
| A job title typed into the chat runs as HTML in the panel | Every value is set with `textContent`; the page has no `innerHTML` |
| Demo conversations stay on disk | `.langgraph_api/` holds the threads (what people typed, masked tool results); it is gitignored. Delete it after a demo (not tested) |
| Prompts and results go to LangSmith | Tracing is forced off in `build()`. Studio (the link `langgraph dev` prints) is a LangSmith page that talks to the local server from your browser; use it with demo data only |

### The script and its rehearsal

Each act of `demo/script.py` shows one thing v2 does differently, on the mock's data. A check
reads the data panel after the act (jobs by id, and the sub-tasks that reached the mock during
it), so a pass means the data changed as the script says. Acts that end with nothing changed,
or with the reviewer's edit, also name the approval card that must have appeared. The rehearsal
answers each card as the script says (approve, reject with a reason, or edit the args), sends a
follow-up turn only while the check still fails ("As a team member."), and deletes its threads
and resets the data at the end.

| Act | Shows | Check |
|---|---|---|
| lookup | the job found by name; the write waits for a card | 7001 gains Kubernetes and Terraform, one write |
| reject | high-stakes changes wait for a person, who can say no | the transfer card appeared and was rejected; nothing reached VIRA |
| chain | a result used in the next step | `add_job_skills` on the id `create_job` returned |
| shortlist | the pick from match scores; the reviewer's edit is what runs | 5102 shortlisted, 5103 not, one write |
| ask | only what can't be looked up is asked | Bob (802) joins 7001 as a team member |
| email | personal values only from the user | one share of 5102, to one recipient |
| recover | LinkedIn needs an open job; both steps when told | 7002 reopened, then published |
| unsupported | no approximation | nothing changed |

`python -m demo.rehearse --repeat 3` on 2026-10-01 with gpt-5-mini, after everything below was
fixed, with every write gated: **24/24**. Medians per act: 1–6 model calls, 5–27k tokens, 8–30
s including the approvals; per pass about 135k tokens, about 2½ minutes and about $0.05
(litellm's price table). Again on 2026-10-02 with `GATE_WRITES = False` (three cards a pass:
the transfer, the shortlist, the share): **24/24**, medians of 1–5 model calls, 5–23k tokens and
8–22 s per act; per pass about 145k tokens, about 2 minutes and about $0.05.

What the rehearsals found, and what changed:
- **The agent went around the reviewer.** Told to shortlist the top two and edited down to one,
  it proposed the other in a new call. The prompt rule alone left it in 2 of 3 runs; now it is
  structural (§3, human in the loop).
- **Re-reads and retries were refused as repeats.** "Show it again" after a change, and
  "reopen it, then publish" after a refused publish, now run (§3a).
- **Ranking applicants one by one** (four reads, 35 s): the rules now point at
  `get_applications`, which returns the scores in one call (§12). The shortlist act went from 7
  model calls to 4.
- **"Add Bob to the backend engineer job"** was once read as adding Bob as a candidate. The
  script now says "to the hiring team".
- **An external recipient was refused.** The db rules validate every email the user gives with
  `validate_emails`, which knows only the company's users, so "share it with
  hm.lee@example.com" stopped. The script shares with a colleague (Priya); whether share
  recipients must be colleagues is a product question (below).
- **The UI sent edited ids as a string** (above). The browser check that found it drives the real
  chat in headless Chromium; it isn't part of the repository.

Open points:
- `langgraph dev` is a development server (in-memory, one process). A hosted pilot would run the
  same graph on a licensed LangGraph deployment or behind our own API with auth.
- **Share recipients.** `validate_emails` checks an address against the company's users, and the
  db rules validate every email before a write, so a CV can't be shared with an outside address.
  If sharing outside the company is allowed, the rule should exempt `share_application`.
- The replies still run long now and then, and sometimes offer to redo what the reviewer removed
  (the code refuses it if asked in the same request).

## 15. The real VIRA engine: Jeni's tasks in `--mode real`

Tried on 2026-10-02 against the staging engine, with read-only task groups only: the engine's own
`sub_task_get_job_description` and v2's `search_users`.

How a call goes now:
- **Route and auth.** Task groups (`jeni_tools.PATH`) go to `VIRA_ACTUAL_LOCATION`, the full
  endpoint URL or a path under `VIRA_BASE_URL`, with one header, `xrtoken: $VIRA_XRTOKEN`. The four
  sample endpoints keep `VIRA_BASE_URL` and their `x-api-key` headers. The protections are the same
  for both (`recruiter_cli._target`, `_real_mode_problem`).
- **The xrtoken is a user's.** v1 makes one per user (`generateXrtokenHelper({ userId })`). Every
  task group runs with that user's rights, in that user's company, so real mode gates every write
  (§3).
- **A session is required.** Without one the engine answers 400: "Missing agent_session_uuid!
  Required: task_group_name, agent_session_uuid, tasks". v1 sends `null` there. v2 sends one per
  conversation, a uuid5 of the thread id (the tools get the thread from their `RunnableConfig`;
  outside a thread, one per process). A fresh uuid was accepted, so the engine doesn't require
  the session to exist.
- **The engine is asynchronous.** The POST answers 200 in about 0.3 s with only
  `{agentTaskGroupUuid, message: "Your tasks have been received. We are processing your tasks.
  You can see task status on the task panel"}`. `jeni_tools.project()` turns that into
  `{"status": "queued", ...}`, and `RULES` tells the model to say the task was submitted, not
  done, and not to use anything it would have returned. Until results can be read back, v2 can't
  chain steps in real mode (a new job's id into the next call).

What the engine runs, from its tables on staging (`hris.agenttaskgroup`, `agenttask`,
`agentsubtask`), last 60 days, statuses and counts only:
- **Vocabulary.** The task and sub-task names in use are v2's catalog names (`task_get_applications`
  / `sub_task_get_applications`, `task_create_job`, …), plus `sub_task_get_jobs`, which v1's
  catalog comments out. The sample's `task_setup_job` with `sub_task_job_create`,
  `sub_task_job_board_linkedin_publish` and a transfer by `new_owner_user_id` appears only in our
  probe: a newer or planned format.
- **Speed.** 130 groups completed in 30 days, with a median of 3 s from creation to done.
- **Failure lives in the sub-tasks.** 404 sub-tasks failed (395 with a `failed_reason`), while every
  task and group ended `completed`, whatever its sub-tasks did.
- **Tasks are independent.** In 117 groups where a task had a failed sub-task and more tasks
  followed, every later task ran: 60 completed fully, 112 had failures of their own, none was left
  unrun.
- **Sub-tasks within a task are chained.** Every sub-task of a multi-sub-task task carries a
  `next_sub_task_payload`, and after a failed sub-task every later one in the same task failed too
  (42 of 42; no success ever followed a failure). Whether they fail because of the first one or for
  reasons of their own needs a controlled run: the reasons are free text from real records and
  weren't read.

So the expected behaviour holds, with one nuance: a failed task shows only in its sub-tasks, and
the task itself reads `completed`.

Why nothing executes (checked 2026-10-02):
- **The endpoint only creates the group.** In v1 the chat controller runs a group itself, in the
  same request (`BhishmaStreamController` → `XagentTaskEngine.handleCreateTaskGroup`, then
  `handleProcesViraTasksAsync`: `getQueuedSubTasks`, `handleSubTask` for each sub-task in order,
  `handleGetTaskGroupInfo` for the result). The engine's reply to us, "We are processing your
  tasks. You can see task status on the task panel", is the progress text v1 shows before that
  loop. Nothing runs the loop for groups created through the endpoint.
- **No worker or Celery.** The engine is v1's Node backend; the staging database has no
  Celery or kombu tables, and 119 groups (back to 2025-01-25, our three probes included) sit
  `queued` forever, most created by the same user as our token, who also has 659 completed ones.
  Having a session or a chat history doesn't decide it (116 of the 119 have a chat history).
- **Not the tunnel or the database.** The tunnel connects in 0.1 s, and the engine wrote our probe
  groups to the same database. No group was created between 2026-09-24 and 2026-10-02 at all,
  which is why the last completed one is from 2026-09-24.
- **Reading results back:** `handleGetTaskGroupInfo` already builds the group with each
  sub-task's status, response and `failedReason`. Exposed as a call by `agentTaskGroupUuid`, it is
  what v2 should read. Until then the same data is in `hris.agentsubtask` for testing.

What engineering needs to provide, in order of preference:
1. Run the group on create (the endpoint calls `handleProcesViraTasksAsync` before it answers, as
   v1's chat does) and return `handleGetTaskGroupInfo`, or
2. a "process" call for an `agentTaskGroupUuid` plus a "get" call for its info, or a worker that
   picks up queued groups;
3. and a word on the 119 stuck groups (and our three probes) on staging: run them or archive them.


- **The controlled run**, once groups execute: one group, two tasks; in the first, a sub-task that
  fails followed by one that would succeed alone; the second task independent. Read-only tasks.

## 16. Personal data: tokens instead of redaction, and "me"

Until 2026-10-02 every email in a result reached the model as `<redacted-email>`. That kept
addresses out of the model's context, but it also meant the agent could never act on one: "share
it with Priya" needed Priya's address typed out, even right after `search_users` had found her.
`pii_vault.py` makes the masking reversible on our side only.

- **Tokens.** `_mask_pii(result, source)` turns each address in a result, under an email key or
  inside text, into `<email:3f9a1c2b4d5e>`: the first 12 hex digits of an HMAC-SHA256 of the
  address under a key the model never sees. The same address gets the same token in a process.
  The vault keeps the address encrypted with AES-GCM (the token as associated data), in memory
  only; the key is `JENI_PII_KEY` (32 bytes, base64) or a random one per process. Nothing is
  written out, and the CLI, one process per command, prints redactions instead.
- **The decrypt step.** `recruiter_cli._call` swaps tokens back to addresses just before a call
  is sent, mock or real: after the guard, after the approval card, after the audit entry (which
  stays redacted). A token the process didn't issue sends nothing.
- **Where a token may go.** Each token remembers where its address was seen. Only a colleague's
  (from the user directory: `search_users`, `find_user`, `validate_email(s)`) may stand in for a
  share recipient or a new owner (`RECIPIENT_FIELDS`). A candidate's, or any other record's, goes
  nowhere, and candidate fields still take a typed address only. `ToolCallGuard` asks
  `pii_vault.VAULT.allowed()` where it checks the user's words.
- **"me".** The signed-in user's own address: `JENI_USER_EMAIL`, or the host's setting (the demo's
  is Sam Lee, 804). "Share it with me" and "transfer it to me" send that address; with nobody
  signed in, Jeni asks for it. In production it comes from the authenticated profile, like the
  company id.
- **On the card** a token shows as a hint, `p•••r@example.com`, and "me" as "you".

Limits: tokens last as long as the process (or the key); a token's address is visible to anyone
who can read the process's memory, and the vault keeps every address the process has seen; the
card's hint (first and last letter, and the domain) is saved with the approval request in the
thread; names inside free text are still not detected.
