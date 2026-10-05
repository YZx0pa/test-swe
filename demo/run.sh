#!/usr/bin/env bash
# Start the Jeni v2 demo: LangGraph's dev server (the agent and the data panel) and the chat UI.
#
#   demo/run.sh                                     # synthetic data + mock VIRA (default)
#   demo/run.sh --real --company-id 5143             # real VIRA + tenant-scoped Postgres
#   demo/run.sh --real --gate-writes                 # also approve every normal write
#
# Both stay on this machine: the server binds 127.0.0.1 and has no auth, so share your screen,
# not the URL.  Ctrl-C stops both.  Logs go to demo/.run/.  Setup: demo/README.md.
set -euo pipefail
MODE=mock
COMPANY_ID=5143
GATE_WRITES=false

usage() {
  echo "usage: demo/run.sh [--real] [--gate-writes] [--company-id ID]" >&2
  exit 2
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --real) MODE=real; shift ;;
    --gate-writes) GATE_WRITES=true; shift ;;
    --company-id) [ "$#" -ge 2 ] || usage; COMPANY_ID=$2; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done
case "$COMPANY_ID" in *[!0-9]*|'') echo "demo/run.sh: --company-id must be a positive integer" >&2; exit 2 ;; esac
cd "$(dirname "$0")/.."
ROOT=$PWD
RUN=$ROOT/demo/.run
API=http://127.0.0.1:2024
export PATH="${NODE_BIN:-$HOME/.local/opt/node/bin}:$PATH"

die() { echo "demo/run.sh: $*" >&2; exit 1; }

# --- preflight: what's missing, never what a secret is -------------------------------
[ -x .venv/bin/langgraph ] || die "LangGraph's dev server isn't installed: uv pip install --require-hashes -r requirements-demo.lock.txt"
command -v pnpm >/dev/null || die "pnpm not found: install Node 20+ and run 'corepack enable' (demo/README.md), or set NODE_BIN"
[ -f "${JENI_TASKS_FILE:-config/jeni_tasks.json}" ] || die "Jeni's task catalog isn't at config/jeni_tasks.json (config/README.md)"
[ -f .env ] || die "no .env: copy .env.example and fill in the model key"
if curl -sf "$API/ok" >/dev/null 2>&1; then die "something is already serving $API; stop it first"; fi

mkdir -p "$RUN"
if [ ! -d demo/ui/node_modules ]; then
  echo "installing the chat UI (locked, with integrity hashes) ..."
  (cd demo/ui && COREPACK_ENABLE_DOWNLOAD_PROMPT=0 pnpm install --frozen-lockfile) > "$RUN/ui-install.log" 2>&1 \
    || die "pnpm install failed: see $RUN/ui-install.log"
fi

# The UI's settings are public (NEXT_PUBLIC_*) and baked in at build time.
export NEXT_PUBLIC_API_URL=http://localhost:2024 NEXT_PUBLIC_ASSISTANT_ID=jeni NEXT_TELEMETRY_DISABLED=1
if [ ! -f demo/ui/.next/BUILD_ID ] || [ -n "$(find demo/ui/src demo/ui/package.json -newer demo/ui/.next/BUILD_ID -print -quit)" ]; then
  echo "building the chat UI ..."
  (cd demo/ui && pnpm build) > "$RUN/ui-build.log" 2>&1 || die "the UI build failed: see $RUN/ui-build.log"
fi

# --- start both, stop both -------------------------------------------------------------
# Each server runs in its own session, and stopping kills the whole group: `next start`
# leaves a next-server child that outlives its parent otherwise.
groups=()
cleanup() {
  for pgid in "${groups[@]}"; do kill -TERM -- "-$pgid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

JENI_DEMO_MODE="$MODE" JENI_COMPANY_ID="$COMPANY_ID" JENI_GATE_WRITES="$GATE_WRITES" LANGGRAPH_CLI_NO_ANALYTICS=1 setsid .venv/bin/langgraph dev --no-browser --no-reload \
  --host 127.0.0.1 --port 2024 > "$RUN/server.log" 2>&1 &
groups+=($!)
(cd demo/ui && exec setsid pnpm start -H 127.0.0.1 -p 3000) > "$RUN/ui.log" 2>&1 &
groups+=($!)

echo "starting ..."
for _ in $(seq 60); do curl -sf "$API/ok" >/dev/null 2>&1 && break; sleep 0.5; done
curl -sf "$API/ok" >/dev/null 2>&1 || die "the server didn't start: see $RUN/server.log"

# Build the agent now, not on the audience's first request.
assistant=$(curl -sf -X POST "$API/assistants/search" -H 'content-type: application/json' \
  -d '{"graph_id": "jeni"}' | .venv/bin/python -c 'import json, sys; print(json.load(sys.stdin)[0]["assistant_id"])')
curl -sf "$API/assistants/$assistant/schemas" >/dev/null || die "the agent didn't build: see $RUN/server.log"
[ "$MODE" != mock ] || curl -sf -X POST "$API/demo/reset" >/dev/null

for _ in $(seq 60); do curl -sf http://127.0.0.1:3000 >/dev/null 2>&1 && break; sleep 0.5; done
curl -sf http://127.0.0.1:3000 >/dev/null 2>&1 || die "the UI didn't start: see $RUN/ui.log"

cat <<EOF

  Jeni v2 demo is up ($MODE mode; company $COMPANY_ID)
    chat      http://localhost:3000
EOF
if [ "$MODE" = mock ]; then
  echo "    data      $API/demo"
else
  echo "    data      disabled (the demo panel only shows synthetic data)"
fi
cat <<EOF
    rehearse  .venv/bin/python -m demo.rehearse
    studio    https://smith.langchain.com/studio/?baseUrl=$API   (optional; a LangSmith page)

  The script is demo/SCRIPT.md.  Ctrl-C stops both.
EOF
wait
