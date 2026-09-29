# `feat/agent-frameworks`: what changed and how to run it

This branch moves the VIRA recruiter agent off mini-swe-agent's bash tool and onto four typed
tools. It runs them on three new runtimes (a LangGraph agent, an explicit LangGraph workflow, and
deepagents) and serves them over MCP. It also adds a live comparison harness, a grounding/trace
module and an offline test suite. mini-swe-agent keeps working as the baseline.

The design rationale, framework survey and measured results are in
[agent-frameworks.md](agent-frameworks.md). This page covers the concrete changes and the steps to
set up, run and test them.

---

## 1. What changed

### One guarded path, many runtimes

```
recruiter_cli.execute(cmd, path, query, body, mode=…)    confirm gate → _call → _audit → _mask_pii
   ├─ recruiter_cli CLI ─────────── run_mini.py            mini-swe-agent (bash), baseline
   ├─ vira_tools.py (typed) ─────┬─ run_langgraph.py       LangGraph agent loop (create_agent)
   │                             ├─ run_deepagent.py       deepagents harness + subagents
   │                             └─ vira_mcp.py (stdio)    any MCP client
   └─ run_workflow.py (via vira_tools)                     fixed LangGraph StateGraph
```

Every runtime reaches VIRA through `recruiter_cli.execute()`, so masking, the audit log and the
confirm gate behave the same everywhere.

### Modified files

| File | Change |
|---|---|
| `recruiter_cli.py` | New `execute()`: the single guarded path (gate → call → audit → mask). It returns a dict and never prints. There are four typed, importable actions (`find_talents`, `generate_jd`, `score_candidates`, `candidate_insights`) with a required keyword-only `mode`. CLI subcommands now just parse flags and call those actions, and requests and printed output are unchanged. `_guard` returns a result instead of exiting, and the CLI still exits `2` on `needs_confirmation`. `get-match-id` is commented out. The CLI still defaults to `--mode real`. |
| `run_mini.py` | Prompt rules tightened: echo `Not_Able…` once then finish, never repeat a command, quote the `SUMMARY` echo so `\|` isn't a pipe. `tool_choice="required"` stops reasoning models from answering in prose and then re-firing the previous command. Default `CHAT_MODEL` is `gpt-5-mini`. |
| `mock_vira.py` | Scores and insights are now deterministic per id instead of constant. For example, applicants 11 / 12 / 13 score 0.78 / 0.95 / 0.71, so "pick the top scorer" tasks can be checked. |
| `commands.md` | `get-match-id` and the notes pointing to it are removed. |
| `.env.example` | `CHAT_MODEL=gpt-5-mini`. |
| `requirements.txt` | Adds `langchain==1.4.2`, `langgraph==1.2.12`, `langchain-openai==1.6.4`, `deepagents==0.7.18`, `mcp==1.30.0`. |
| `requirements.lock.txt` | Gains the new transitive pins. Every pin mini-swe-agent already used is unchanged. |
| `.gitignore` | Adds `.pytest_cache/`. |

### New files

| File | What it is |
|---|---|
| `requirements-dev.txt` | `-r requirements.txt` plus `pytest==9.1.1`. |
| `pytest.ini` | `pythonpath = .`, `testpaths = tests`. |
| `vira_tools.py` | The model-facing contract: four typed functions whose `Annotated[…, Field(…)]` signatures and docstrings are the tool schema. `configure(mode)` sets real/mock once per process. Tools always pass `confirmed=False` and turn exceptions into `{"status": "error", …}` with the exception type only. `langchain_tools()` wraps them for LangChain. |
| `agent_kit.py` | Shared by the LangChain runners: `SYSTEM_PROMPT`; `build_chat_model()`, which maps litellm-style ids such as `openai/gpt-5-mini` and uses Chat Completions for OpenAI; `ToolCallGuard` middleware; `CallLedger`; the model-call cap; `--approve-all` interrupts; usage counting; the REPL; and `--trace-json`. |
| `run_langgraph.py` | LangGraph agent loop via `langchain.agents.create_agent`, with an `InMemorySaver` checkpointer. |
| `run_workflow.py` | Explicit `StateGraph`: score → shortlist top-k (in code) → approval interrupt → insights → one-call summary. The model never picks a tool. |
| `run_deepagent.py` | deepagents harness: `write_todos` planning, a virtual filesystem kept in graph state (`StateBackend`, never the host disk), and subagents `sourcing-analyst` and `jd-writer`, plus a guarded `general-purpose`. They share one `CallLedger` per task. |
| `vira_mcp.py` | FastMCP stdio server exposing the four tools. `--mode` is required. It loads `.env` itself, so no client config carries VIRA credentials. Confirm-gated tools are never exposed. |
| `compare_agents.py` | Runs the same tasks through mini, LangGraph and deepagents against mock VIRA only. PASS/FAIL is judged on the audit log, i.e. what actually reached VIRA. Writes a markdown report and optional JSON traces. |
| `grounding.py` | Pure functions that trace every tool argument, and every id or score in the final answer, back to the task or an earlier tool result. They also flag ids passed as the wrong kind (e.g. a `profile_id` sent as `match_ids`). `ToolCallGuard` and `compare_agents.py` use them. |
| `tests/` | 56 offline tests (section 3). |

### Behaviour that holds in every new runtime

- Tool results go through `_mask_pii` before they reach the model, the graph state or the
  checkpointer.
- Every VIRA call writes one audit line in the same format (`ts`, `command`, `query`, masked
  `body`, `status`).
- The host sets real/mock once. The model never sees a mode parameter. The new runners default to
  `--mode mock`.
- Tools pass `confirmed=False`, so a command in `NEEDS_CONFIRM` can't run from an agent, and
  `vira_mcp.py` doesn't list it.
- Errors come back as `{"status": "error", …}` with the exception type only, never hosts or URLs.
- `ToolCallGuard` refuses, without calling VIRA:
  - exact repeats of a call (normalised args), across a deepagents main agent and all its
    subagents;
  - ids passed as a different kind than they came back as.
- `ModelCallLimitMiddleware(thread_limit=12)` caps model calls per task. LangSmith tracing is off
  unless `--trace` is passed.

### Commits (oldest first)

| Commit | Subject |
|---|---|
| `d20fd12` | WIP: drop get-match-id, tighten mini rules, default to gpt-5-mini |
| `3f889dc` | Add LangGraph, deepagents and MCP dependencies |
| `7b249bd` | Expose recruiter_cli's guarded path as typed, importable actions |
| `2a486d4` | Run the VIRA agent on LangGraph: typed tools, agent loop and workflow |
| `bba9fde` | Run the VIRA agent on deepagents |
| `eac1145` | Serve the VIRA tools over MCP |
| `fb1c541` | Fix two agent failures found by the live comparison |
| `69207ce` | Add a live comparison harness and the frameworks write-up |
| `45b9477` | Measure grounding and add a data-driven task to the comparison |
| `dd14e7c` | Catch ids of the wrong kind, parse mini's real transcripts, record traces |
| `e7552a1` | Refuse ids of the wrong kind before they reach VIRA |
| `7fa9583` | Count refused wrong-kind ids separately from ones that reached VIRA |

### Suggested reading order

1. `recruiter_cli.py`: `execute()` and the four typed actions.
2. `vira_tools.py`: the tool contract.
3. `agent_kit.py`: `ToolCallGuard`, `CallLedger`, `middleware()`, `run_task()`.
4. `run_langgraph.py`, then `run_workflow.py`, then `run_deepagent.py`, then `vira_mcp.py`.
5. `grounding.py` and `compare_agents.py`.
6. `tests/conftest.py` and `tests/fakes.py`, then the test files.

---

## 2. Setup

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git checkout feat/agent-frameworks
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt        # or: uv pip install -r requirements.lock.txt  (exact pins, includes pytest)
```

Configuration comes from `.env`, which is gitignored and loaded by the programs themselves.
Every name is listed in `.env.example` with a placeholder:

```bash
cp .env.example .env     # then fill in the values you need
```

| Activity | Needs in `.env` |
|---|---|
| Offline tests (section 3) | nothing: the tests disable `.env` loading |
| `recruiter_cli.py --mode mock`, `run_workflow.py --no-llm` | nothing |
| Anything that calls an LLM: the agent runners, `compare_agents.py`, `run_workflow.py` without `--no-llm` | `OPENAI_API_KEY`; optionally `CHAT_MODEL` (default `gpt-5-mini`) |
| `--mode real` (section 5) | `VIRA_BASE_URL`, `VIRA_API_KEY`, `VIRA_CLIENT_NAME`, `VIRA_USER_ID` |

To check that a value is present without displaying it:

```bash
grep -c '^OPENAI_API_KEY=.' .env     # 1 = set and non-empty, 0 = missing or empty
```

**Keep manual runs out of the real audit log.** Every VIRA call (mock or real) appends a line to
`$EVENTS_LOG` (default `events.jsonl`). A shell variable takes precedence over `.env`, because
`load_dotenv()` doesn't override existing variables. So for the manual steps below, use a
throwaway log and read it afterwards to see exactly what reached VIRA:

```bash
export EVENTS_LOG=/tmp/vira-audit.jsonl
rm -f "$EVENTS_LOG"
```

---

## 3. Offline test suite

No `.env`, no network, no LLM, no VIRA:

```bash
python -m pytest -q
# ........................................................   [100%]
# 56 passed
```

How the tests stay hermetic:
- `tests/conftest.py` fixes the environment before any project module is imported:
  - `PYTHON_DOTENV_DISABLED=1`, so every `load_dotenv()` is a no-op;
  - `JENI_MODE=mock`;
  - a dead `VIRA_BASE_URL` (`127.0.0.1:9`) and blank VIRA credentials;
  - a fake `OPENAI_API_KEY`;
  - tracing off.
- An autouse fixture points the audit log at `tmp_path`.
- `tests/fakes.py` provides `ScriptedModel`, a fake chat model that replays a fixed list of
  `AIMessage`s (tool calls and answers). That lets full LangGraph and deepagents graphs run
  offline.

| File | Tests | Covers |
|---|---|---|
| `tests/test_recruiter_cli.py` | 12 | For all four endpoints, the CLI and the typed action send the identical request, and both mask results. The audit line masks the request body. Typed actions have no default `mode`. Mock mode end to end. The confirm gate blocks before any call, the CLI exits `2`, and `confirmed=True` passes. |
| `tests/test_agents.py` | 21 | Tool schemas are typed and described. Tools use the configured mode and never confirm. Scoring with no ids never calls VIRA. VIRA failures become error results with no host. `build_chat_model` id mapping. The agent sees only masked output. Guard: exact repeats refused, parallel duplicates run once, different args allowed, wrong-kind ids refused before VIRA, ids the user named are allowed. A crashing tool doesn't crash the run. Invalid args come back to the model. The model-call cap ends the run. `--approve-all`: approve runs the call, reject never reaches VIRA, edit runs the edited args. Workflow: waits for approval then runs insights on the shortlist, a rejection skips insights, the summary comes from the model, and it stops on a VIRA error. |
| `tests/test_deepagent.py` | 8 | `StateBackend` only, and `execute` isn't offered. A hallucinated `execute` call runs nothing. The main agent and subagents go through the guarded path, and a subagent returns only its answer. A subagent can't repeat the parent's call. The ledger is per task. Subagents inherit approval. Virtual files stay in graph state and nothing is written to disk. |
| `tests/test_mcp.py` | 4 | Tool list, annotations (`readOnlyHint` / `idempotentHint`) and masked results. Invalid args are an MCP error with no audit line. Confirm-gated tools aren't listed. A real `vira_mcp.py` subprocess over stdio writes only JSON-RPC to stdout and audits to `$EVENTS_LOG`. |
| `tests/test_grounding.py` | 11 | Values traced to the task or earlier results, and invented values flagged. `lang=en` counts as a default, and percentages match scores. Refusals are marked. Mini commands are parsed and off-policy output hidden. Typos stay ungroundable. Real ids of the wrong kind count as misuse, while right-kind ids and task ids are fine. `reground()` matches a fresh trace. Mini's `<returncode>` observation is unwrapped. Wrong-kind ids the guard refused count as blocked, not as reaching VIRA. |

Run a subset:

```bash
python -m pytest tests/test_mcp.py -q
python -m pytest -q -k "approve or reject or edit"
python -m pytest -q -k wrong_kind
```

---

## 4. Manual runs against mock VIRA

Everything here uses the local `MockVira` stand-in. Steps 4.1 and 4.2, and the raw JSON-RPC test
in 4.5, need no LLM and are fully deterministic. The rest call the model named by `CHAT_MODEL` and
need `OPENAI_API_KEY`; their outputs vary from run to run.

After each step, `cat "$EVENTS_LOG"` shows exactly what reached VIRA.

### 4.1 The CLI directly (no LLM)

`recruiter_cli.py` still defaults to `--mode real`, so always pass `--mode mock` here.

```bash
python recruiter_cli.py --mode mock find-talents --job-ids 123
python recruiter_cli.py --mode mock score-candidates --app-ids 11,12,13
python recruiter_cli.py --mode mock candidate-insights --app-ids 11,12
python recruiter_cli.py --mode mock generate-jd --job-title "Data Analyst" --skills "SQL,Python" --lang ar
cat "$EVENTS_LOG"
```

Expect:

```text
{"status": "ok", "http_status": 200, "result": {"job_id": 123, "suggested_profiles": [{"profile_id": 900001, "match_score": 0.91}, {"profile_id": 900002, …}, {"profile_id": 900003, …}], "_note": "SYNTHETIC mock response"}}
{"status": "ok", …, "result": {"scores": [{"app_id": 11, "composite_score": 0.78, "briq": 0.62}, {"app_id": 12, "composite_score": 0.95, "briq": 0.85}, {"app_id": 13, "composite_score": 0.71, "briq": 0.67}], …}}
{"status": "ok", …, "result": {"version": "v3", "insights": [{"app_id": 11, "summary": "good culture add; needs Go ramp-up (mock)"}, {"app_id": 12, "summary": "strong backend fit (mock)"}], …}}
{"status": "ok", …, "result": {"lang": "ar", "job_description": "[ar] Draft JD for 'Data Analyst'. …", "skills_used": ["SQL", "Python"], …}}
```

The audit log has four lines, one per call, showing the query/body split per endpoint. For example:

```text
{"ts": "…", "command": "score-candidates", "query": {"composite_score": "True", "briq": "True", "recal_briq": "True"}, "body": {"app_ids": [11, 12, 13], "match_ids": []}, "status": "ok"}
{"ts": "…", "command": "generate-jd", "query": {"lang": "ar"}, "body": {"job_id": 0, "job_title": "Data Analyst", "skills": ["SQL", "Python"], "job_function": [], "industry": [], "other_requirements": []}, "status": "ok"}
```

### 4.2 The explicit workflow (no LLM with `--no-llm`)

```bash
python run_workflow.py --app-ids 11,12,13 --top 2 --yes --no-llm
```

Expect the top two by `composite_score` (12, then 11) to be shortlisted, approved, and given
insights:

```text
[approval] Run candidate insights for applicants [12, 11]?

=== workflow result ===
scores   : [{"app_id": 11, "composite_score": 0.78, "briq": 0.62}, {"app_id": 12, "composite_score": 0.95, "briq": 0.85}, {"app_id": 13, "composite_score": 0.71, "briq": 0.67}]
shortlist: [12, 11]
approved : true
insights : [{"app_id": 12, "summary": "strong backend fit (mock)"}, {"app_id": 11, "summary": "good culture add; needs Go ramp-up (mock)"}]
summary  : "Shortlisted [12, 11]; insights ran."
```

Now run it without `--yes` and answer `n` at `approve? [y/N] >`. Expect `approved : false`, no
`insights` line, and a summary saying insights didn't run. The audit log gains a
`score-candidates` line but no `candidate-insights` line: the graph paused at the interrupt and a
rejection skipped the call.

Drop `--no-llm` to have one LLM call write the summary from the collected state. That needs
`OPENAI_API_KEY`.

### 4.3 LangGraph agent (`run_langgraph.py`)

```bash
python run_langgraph.py --task "Find potential talents for job 123."
```

Expect a trajectory, then usage:

```text
=== trajectory ===
[0] user      : Find potential talents for job 123.
[1] assistant → call: find_talents({"job_ids": [123]})
[2] tool      → {"status": "ok", "http_status": 200, "result": {"job_id": 123, "suggested_profiles": […]}}
[3] assistant : <prose answer listing 900001–900003 and their scores>

(model calls: 2, tokens in/out: …/…)
```

Try the behaviours the guard and prompt are meant to enforce:

| Task | Expected |
|---|---|
| `"Generate a job description."` | No tool call. The answer says the job title is missing and doesn't invent one. |
| `"Find talents for job 123 and score them."` | One `find_talents` call, then a stop saying match ids are missing, since `find_talents` returns profile ids and no tool converts them. If the model does try `score_candidates` with those ids, the tool result is `Refused: 900001 is a profile_id, not a match_id …` and nothing reaches VIRA. |
| `"Score applicants 11, 12 and 13, then get candidate insights only for the one with the highest composite score."` | `score_candidates({"app_ids": [11, 12, 13]})`, then `candidate_insights({"app_ids": [12]})`, since 12 has the top mock score. |
| The same task twice in one prompt, e.g. `"Find talents for job 123. Then find talents for job 123 again."` | Either the model doesn't repeat the call, or the repeat comes back `Refused: identical to an earlier call in this task`. Either way the audit log has one `find-talents` line. |

**Human approval:**

```bash
python run_langgraph.py --approve-all --task "Find potential talents for job 123."
```

Each VIRA call pauses:

```text
[approval] find_talents({"job_ids": [123]})
approve? [y]es / [n]o / [e]dit args >
```

- `y` runs the call.
- `n` rejects it. VIRA is never called, so no audit line is written, and the model is told the
  user declined.
- `e` prompts `new args as JSON >`. Enter e.g. `{"job_ids": [124]}` and the audit log shows
  `job_ids: [124]`.

**Interactive mode:** run `python run_langgraph.py` with no `--task`. Type tasks at
`input task>`, and `quit` to exit. Each task runs on a fresh thread.

Other flags:
- `--step-limit N`: model-call cap per task (default 12).
- `--trace-json PATH`: write the run's steps and grounding (section 6).
- `--trace`: allow LangSmith tracing, which sends prompts and masked results off the machine.

### 4.4 deepagents (`run_deepagent.py`)

```bash
python run_deepagent.py --task "For jobs 101 and 102, find suggested talents and write a short report to /report.md"
ls report.md        # expect: No such file or directory
```

Expect:
- a `write_todos` call (the plan);
- `task` calls delegating to `sourcing-analyst`, typically one per job, possibly in parallel;
- a `write_file` to `/report.md`;
- a `=== virtual file /report.md ===` block printed after the trajectory.

The trajectory shows only the main agent's messages; each subagent's work comes back as the
result of its `task` call. The audit log has one `find-talents` line per job. The report lives in
graph state only, so nothing is written to the working directory.

`--approve-all` also works here. Subagents inherit it, so every VIRA call made by a subagent
pauses for approval too. Expect noticeably more tokens than LangGraph for the same work; section 7
of [agent-frameworks.md](agent-frameworks.md#7-results-the-same-tasks-on-three-runtimes) has
numbers.

### 4.5 MCP server (`vira_mcp.py`)

`--mode` is required. stdout is the JSON-RPC channel and logs go to stderr. Run by hand, the
server just waits on stdin, so it's easiest to test from a client.

**Raw JSON-RPC smoke test (no LLM):**

```bash
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"find_talents","arguments":{"job_ids":[123]}}}'
  sleep 2; } | python vira_mcp.py --mode mock 2>/dev/null
```

The `sleep` keeps stdin open until the tool call finishes. Without it the server exits at EOF
and drops the `id: 3` reply. Expect three JSON-RPC lines:
- `id: 1`: `serverInfo.name` is `"vira"`.
- `id: 2`: the four tools `find_talents`, `generate_jd`, `score_candidates`,
  `candidate_insights`.
- `id: 3`: the mock `find-talents` result as JSON text.

The audit log gains one `find-talents` line.

**From Claude Code** (run from the repo root):

```bash
claude mcp add vira -- "$PWD/.venv/bin/python" "$PWD/vira_mcp.py" --mode mock
# in a Claude Code session: /mcp shows "vira" connected; ask it to find talents for job 123
claude mcp remove vira
```

The server loads `.env` itself, found next to `recruiter_cli.py` whatever the working directory,
so the client config carries no VIRA credentials. Snippets for LangGraph (via
`langchain-mcp-adapters`), the OpenAI Agents SDK and PydanticAI are in section 5 of
[agent-frameworks.md](agent-frameworks.md#5-mcp-server-vira_mcppy). They weren't install-tested
on this branch.

### 4.6 Side-by-side comparison (`compare_agents.py`)

`compare_agents.py` forces mock VIRA before importing anything:
- it takes only `OPENAI_API_KEY` and `CHAT_MODEL` from `.env`, then disables `.env` loading;
- it blanks the VIRA credentials and uses a dead `VIRA_BASE_URL`;
- it runs mini's bash in a scratch copy of `recruiter_cli.py` + `mock_vira.py`.

Each run gets its own audit log, and PASS/FAIL is judged on that log.

```bash
# cheap check: one runtime, two tasks (well under a cent with gpt-5-mini)
python compare_agents.py --runners langgraph --tasks find,no_title

# full run: 6 tasks × 3 runtimes, markdown report to a file (roughly $0.15 with gpt-5-mini)
python compare_agents.py --out report.md

# repeated runs with another model, plus JSON traces
python compare_agents.py --model gpt-4o-mini --repeat 3 --json runs.json
```

| Option | Values |
|---|---|
| `--runners` | `mini`, `langgraph`, `deepagents` |
| `--tasks` | `find`, `jd_ar`, `score_insights`, `top_pick`, `id_trap`, `no_title` |

Each task's pass condition is written as `check_<task>` in `compare_agents.py`. Progress lines
(`… model · runner · task · run n`) and the audit-log directory go to stderr. The report goes to
stdout:
- a table with one cell per task × runtime: verdict · model calls · tokens · est. cost ·
  seconds;
- a totals row: passes, cost, ungrounded values, wrong-kind ids that reached VIRA, and wrong-kind
  ids the guard refused;
- per-run details for every failure and for run 1 of each cell: tool calls as issued, what
  reached VIRA, and the final answer.

A mini command other than `python3 recruiter_cli.py --mode mock …` or a plain `echo` fails that
run, and its output is never shown. LLM runs aren't deterministic, so use `--repeat` before
reading much into a single cell. `report.md` and `runs.json` aren't gitignored, so don't commit
them.

### 4.7 mini-swe-agent baseline (`run_mini.py`)

```bash
JENI_MODE=mock python run_mini.py        # interactive; type a task, 'quit' to exit
```

mini runs model-written shell commands in yolo mode, in the current directory, with `.env` loaded
into their environment. See section 1 of
[agent-frameworks.md](agent-frameworks.md#1-where-we-started-mini-swe-agent--bash). The mode
comes from `JENI_MODE` (default `real`). To exercise mini safely, prefer
`compare_agents.py --runners mini`, which runs it in a sandbox copy with keys blanked.

---

## 5. Against real VIRA (optional)

This needs a reachable VIRA at `VIRA_BASE_URL` and the four `VIRA_*` values in `.env`. Two things
to know first:
- `score-candidates` and `candidate-insights` trigger calculations on VIRA (e.g. `recal_briq`), so
  they aren't pure reads. Use ids from a test tenant.
- Real calls are audited to `$EVENTS_LOG` like mock ones.

```bash
python recruiter_cli.py --mode real find-talents --job-ids <job_id>                 # direct, no LLM
python run_workflow.py --mode real --app-ids <id>,<id>,<id> --top 2                 # asks before insights
python run_langgraph.py --mode real --approve-all --task "Find potential talents for job <job_id>" \
    --trace-json real.json                                                          # approve each call; local trace
python vira_mcp.py --mode real                                                      # for an MCP client
```

Use `--approve-all` for agent runs in real mode so you see every call before it goes out.
`--trace-json` output contains masked tool results. It stays on the machine, but it isn't
gitignored.

---

## 6. Trace and grounding JSON

`compare_agents.py --json PATH` and the runners' `--trace-json PATH` write the same format:

```text
{
  "model": "gpt-5-mini", "mode": "mock", "repeat": 1, "created": "2026-…Z",
  "tasks": { "<task key or text>": {"text": "…", "check": "…"} },
  "runs": [ {
    "runner", "task", "repeat", "passed", "error",
    "model_calls", "input_tokens", "output_tokens", "seconds", "cost",
    "audit":     [ …audit-log lines… ],
    "steps":     [ …see below… ],
    "grounding": { …summary… },
    "final":     "…final answer…"
  } ]
}
```

From a runner's `--trace-json`, `passed` and `cost` are `null` and `audit` is empty. Only the
comparison harness judges runs.

`steps[]` holds the run in order. Each step has an `i` and a `kind`:

| `kind` | What it holds |
|---|---|
| `note` | Text the model wrote alongside a tool call. |
| `call` | `tool` and `args`. A VIRA call also has `provenance`: for each argument value, its `sources`. Mini calls also keep the `command`. |
| `result` | `tool`, `status` and `content`. `refused: true` when the guard refused the call. |
| `echo` | A mini `echo` (mini runs only). |
| `answer` | The final `text`, plus `numbers`: each id, score or percentage in it, with its `sources`. |

A value's `sources` can be:
- `"task"`: the value was in the user's request;
- `"step N"`: it appeared in an earlier tool result ("chained");
- `"default"`: a documented default (`lang=en`);
- `[]`: ungrounded.

A `misused_as` key marks a real id passed as the wrong kind.

`grounding` summarises the steps:

| Field | Meaning |
|---|---|
| `args` / `args_grounded` / `args_chained` | Tool-argument counts: all, traced, and traced to an earlier result. |
| `numbers` / `numbers_grounded` | Answer-number counts: all and traced. |
| `ungrounded` | Values with no source. |
| `misused` | Wrong-kind ids that reached VIRA. |
| `misused_blocked` | Wrong-kind ids the guard refused. |

Print one line per run:

```bash
python -c "import json,sys; d=json.load(open(sys.argv[1])); [print(r['runner'], r['task'], r['passed'], r['grounding']) for r in d['runs']]" runs.json
```

---

## 7. Known limitations

These are tracked as follow-ups in
[agent-frameworks.md §10](agent-frameworks.md#10-follow-ups):

- **`NEEDS_CONFIRM` is empty.** The confirm gate is exercised only by tests that patch it. A
  gated command would be unreachable from agents and missing from MCP until approval is designed
  for write tools.
- **No `get-match-id`.** Suggested talents can't be scored correctly. The `id_trap` task passes
  only when the agent stops after `find_talents`.
- **mini is unchanged apart from its prompt.** It still has the shell and `.env` exposure
  described in §1.
- **Failed calls aren't audited.** A VIRA connection error raises before `_audit`.
- **Masking is by key name.** A non-JSON real-mode reply (`{"raw": text}`) passes through
  unmasked.
- **Approvals are in memory only.** `InMemorySaver` means a paused approval doesn't survive a
  restart.
- **LLM results vary.** The comparison numbers in §7 of the design doc are single runs.
