# Jeni v2 demo

Jeni v2 (the `jeni_db` agent from `run_langgraph.py`) in a chat window, with a live view of
the data beside it. It runs on mock VIRA and synthetic data only. The presenter's script is
[SCRIPT.md](SCRIPT.md); the design notes are in
[docs/agent-frameworks.md §14](../docs/agent-frameworks.md#14-the-demo-demo-on-langgraph-dev).

```
browser ─ chat UI (demo/ui, :3000) ──────────── langgraph dev (127.0.0.1:2024) ─ graph "jeni" (demo/jeni_graph.py)
        └ data panel (iframe of :2024/demo) ─┘        └ /demo, /demo/state, /demo/reset, /demo/prompts (demo/app.py)
```

## Setup (once)

1. Python: `uv pip install --require-hashes -r requirements-demo.lock.txt` (the main lock plus
   LangGraph's dev server).
2. `.env` with the model key (`OPENAI_API_KEY` for the default `CHAT_MODEL=gpt-5-mini`), and
   Jeni's task catalog at `config/jeni_tasks.json` ([config/README.md](../config/README.md)).
   Nothing else is needed: no VIRA credentials, no database.
3. Node 20 or later (tested with 24.21.0 LTS) and pnpm. Without root, for example:

   ```bash
   cd /tmp && V=v24.21.0 && F=node-$V-linux-x64.tar.xz
   curl -fLO https://nodejs.org/dist/$V/$F && curl -fLO https://nodejs.org/dist/$V/SHASUMS256.txt
   grep " $F\$" SHASUMS256.txt | sha256sum -c           # must print: OK
   mkdir -p ~/.local/opt && tar -xJf $F -C ~/.local/opt && ln -sfn ~/.local/opt/node-$V-linux-x64 ~/.local/opt/node
   ~/.local/opt/node/bin/corepack enable                 # pnpm, at the version demo/ui pins
   ```

   `demo/run.sh` looks for Node in `~/.local/opt/node/bin`; set `NODE_BIN` if it's elsewhere.
4. The UI's packages install on the first `demo/run.sh`, from `demo/ui/pnpm-lock.yaml` with
   `--frozen-lockfile` (integrity hashes checked). pnpm 10 doesn't run their install scripts.
   `cd demo/ui && pnpm audit`: no known vulnerabilities on 2026-10-01.

## Run

```bash
demo/run.sh
```

It checks the setup, builds the UI when its sources changed, starts both servers, builds the
agent (so the first request isn't slow), resets the data, and prints:

- chat: <http://localhost:3000>, with **Show data** / **Hide data** for the panel;
- data: <http://127.0.0.1:2024/demo>, the panel on its own;
- Studio (optional): the LangSmith page that shows the graph, talking to the local server.

On WSL, open the chat in the Windows browser; WSL forwards `localhost`. Ctrl-C stops both
servers. Logs are in `demo/.run/`.

## Rehearse

```bash
.venv/bin/python -m demo.rehearse                 # the whole script once: about 2 minutes, about $0.05
.venv/bin/python -m demo.rehearse --repeat 3      # how reliably each act lands
.venv/bin/python -m demo.rehearse --act shortlist
```

It plays [script.py](script.py) against the running server through the same API the chat uses,
does what the script says at each approval card, and checks the data after each act, not the
reply. It deletes its chats and resets the data at the end. It uses the live model, so results
vary a little from run to run.

## What's where

| File | What it is |
|---|---|
| `../langgraph.json` | The server's config: graph `jeni`, the panel's routes, `.env`. |
| `jeni_graph.py` | The demo agent: mock mode, the mock remembers changes, replies written for a chat window. Routine writes run straight away and the high-stakes ones (shortlist, reject, share, transfer ownership) wait for a card; `GATE_WRITES = True` puts every write behind one, as on real VIRA. |
| `app.py`, `panel.html` | The data panel and its routes, and `/demo/prompts` for the chat's starter cards. |
| `script.py` | The acts: prompts, what to do at each card, and what the data must show afterwards. |
| `SCRIPT.md` | The presenter's copy: what to type, what to click, what to point out. A test keeps its prompts identical to `script.py`. |
| `rehearse.py` | Plays the script against the running server and checks the data. |
| `run.sh` | Starts and stops everything. |
| `ui/` | The chat UI (below). |

## The chat UI (`ui/`)

[Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) (MIT, `ui/LICENSE`) at upstream
commit `cf72cb0f68a04d24db93eb19afb2d46f3a5261d4` (2026-09-28). It is vendored so the demo can't
change when upstream does. It already renders LangChain's approval requests (approve, edit,
reject) and resumes the run with the decision. The first commit of `demo/ui` is upstream's tree
without `.github/`, and the one after it holds every local change (`git log -- demo/ui`):

- Jeni's name and mark, and the page title, instead of Agent Chat's; no GitHub links; no file
  upload (Jeni doesn't read files).
- The data panel: **Show data** / **Hide data**, and a column with the server's `/demo` page, open
  by default (`?data=false` hides it).
- Starter cards on an empty chat, from `/demo/prompts`: all 17 acts, numbered as in `SCRIPT.md`
  and grouped as Jobs, Applicants, Team & sharing and Guardrails. A click fills the input box and
  brings it into view; the presenter sends it. The start screen scrolls to its last card (its top
  offset is padding: a margin pushed the scroll area's bottom out of the window).
- **No internal workings on screen.** Tool calls, their payloads and raw results are never shown,
  and the "Hide Tool Calls" switch and the API host badge are gone. The approval card shows the
  server's plain-language summary ("Shortlist applications 5102 and 5103.") instead of the tool
  name, and has no thread id, Studio link, State or Description panels, or Mark as Resolved; the
  call's fields appear only after **Change details**.
- **Edited arguments keep their types.** Upstream sends each edited value as the text in its box,
  so editing `app_ids` to `[5102]` sent the string `"[5102]"` and the tool refused it. Where the
  proposed value wasn't a string, the edit is now parsed back to JSON before the run resumes, and
  a value that doesn't parse is an error on the card instead
  (`agent-inbox/utils.ts: restoreArgTypes`). Worth reporting upstream.
- `pnpm.overrides` for `@babel/core` ≥ 7.29.6 and `postcss-selector-parser` ≥ 7.1.3, two
  low-severity advisories in build tooling.

## Keep it safe

- **Mock only.** The graph fixes mock mode before its tools exist; nothing reaches VIRA or a
  database.
- **This machine only.** The server binds 127.0.0.1 and has no auth: anyone who reached it could
  read and change every chat. Share your screen, not the URL; don't add `--host 0.0.0.0` or
  `--tunnel`.
- **Chats are saved** in `.langgraph_api/` (gitignored): what people typed and masked tool
  results. Delete the folder after a demo.
- **Tracing is off.** Studio is optional. It is a LangSmith page, so use it with demo data only.

## If something's wrong

| Symptom | Fix |
|---|---|
| `something is already serving http://127.0.0.1:2024` | An earlier server is still running: stop it, or Ctrl-C the other `run.sh`. |
| `the UI build failed` | See `demo/.run/ui-build.log`. |
| An error toast in the chat | The server may have stopped: see `demo/.run/server.log`, then restart `demo/run.sh`. |
| The data looks wrong | **Reset data** in the panel, or `curl -X POST http://127.0.0.1:2024/demo/reset`. |
| The chat's history lists old chats | Stop the demo and delete `.langgraph_api/`. |
