#!/usr/bin/env python3
"""Rehearse the demo: play demo/script.py against the running server and check the data.

    demo/run.sh                                        # the server (and the UI), in another terminal
    .venv/bin/python -m demo.rehearse                  # every act once, from the starting data
    .venv/bin/python -m demo.rehearse --repeat 3       # three passes: how reliably each act lands
    .venv/bin/python -m demo.rehearse --act ask --act email

Every act starts from the starting data (POST /demo/reset) in a new thread, as the presenter
does with a new chat, so one act's changes (a copied job, say) can't change the next one's
answer; at the end it deletes its threads, so the chat's history stays clean, and resets the
data for the real run.  At an approval card it does what the script says: approve,
reject or edit.  A turn marked if_needed is sent only while the act's check still fails, like
the answer to "as a team member or an administrator?".  The check reads the data panel, so a
pass means the data changed as the script says, not that the reply sounded right.  An act
also fails if a card it names never appeared.

It talks to the live model, like the demo: see docs/agent-frameworks.md §14 for tokens and time.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from demo import script
from terminal import printable

MAX_CARDS = 8            # approval rounds per turn before giving up


class Server:
    """The LangGraph server's API as the chat UI uses it (langgraph_sdk), and the panel's."""

    def __init__(self, url: str):
        import httpx
        from langgraph_sdk import get_sync_client
        self.url = url.rstrip("/")
        self.client = get_sync_client(url=self.url)
        self.http = httpx.Client(base_url=self.url, timeout=30)

    def new_thread(self) -> str:
        return self.client.threads.create()["thread_id"]

    def send(self, thread: str, text: str) -> None:
        self.client.runs.wait(thread, "jeni", input={"messages": [{"role": "user", "content": text}]})

    def resume(self, thread: str, decisions: list[dict]) -> None:
        self.client.runs.wait(thread, "jeni", command={"resume": {"decisions": decisions}})

    def state(self, thread: str) -> dict:
        state = self.client.threads.get_state(thread)
        return {"messages": (state.get("values") or {}).get("messages", []),
                "interrupts": state.get("interrupts") or []}

    def snapshot(self) -> dict:
        return self.http.get("/demo/state").raise_for_status().json()

    def reset(self) -> None:
        self.http.post("/demo/reset").raise_for_status()

    def delete_thread(self, thread: str) -> None:
        self.client.threads.delete(thread)


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def _args(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False)


def show_messages(messages: list, seen: int, out) -> int:
    """Print the thread's messages after the first `seen`; returns how many there are now."""
    for m in messages[seen:]:
        if m.get("type") == "ai":
            for call in m.get("tool_calls") or []:
                out(f"  jeni  → {call['name']}({_args(call['args'])})")
            if _text(m):
                out(f"  jeni  : {_text(m)}")
        elif m.get("type") == "tool":
            text = _text(m)
            out(f"  tool  ← {text[:140]}{'…' if len(text) > 140 else ''}")
    return len(messages)


def _jobs(snapshot: dict) -> dict:
    return {j["jobId"]: j for j in snapshot["jobs"]}


def run_act(server, act: script.Act, out=print) -> dict:
    """One act in a new thread; the check reads the data afterwards."""
    before = len(server.snapshot()["activity"])
    shown: set[str] = set()

    def passes() -> bool:
        snap = server.snapshot()
        return act.check(_jobs(snap), snap["activity"][before:])

    thread, seen, started = server.new_thread(), 0, time.monotonic()
    for turn in act.turns:
        if turn.if_needed and passes():
            break
        out(f"  you   > {turn.say}")
        server.send(thread, turn.say)
        for _ in range(MAX_CARDS):
            state = server.state(thread)
            seen = show_messages(state["messages"], seen, out)
            if not state["interrupts"]:
                break
            request = state["interrupts"][0]["value"]
            decisions = []
            for action in request["action_requests"]:
                shown.add(action["name"])
                d = script.decision(act, action)
                edited = f" {_args(d['edited_action']['args'])}" if d["type"] == "edit" else ""
                out(f"  card  : {action['name']}({_args(action['args'])}) → {d['type']}{edited}")
                decisions.append(d)
            server.resume(thread, decisions)
        seen = show_messages(server.state(thread)["messages"], seen, out)
    messages = server.state(thread)["messages"]
    ai = [m for m in messages if m.get("type") == "ai"]
    usage = [m.get("usage_metadata") or {} for m in ai]
    models = {(m.get("response_metadata") or {}).get("model_name") for m in ai} - {None}
    tokens_in = sum(u.get("input_tokens", 0) for u in usage)
    tokens_out = sum(u.get("output_tokens", 0) for u in usage)
    missing = sorted(set(act.cards) - shown)
    if missing:
        out(f"  no approval card for {', '.join(missing)}")
    return {"act": act.key, "thread": thread, "ok": passes() and not missing,
            "model_calls": len(ai), "tokens": tokens_in + tokens_out,
            "cost": _cost(min(models, default=None), tokens_in, tokens_out),
            "seconds": round(time.monotonic() - started, 1)}


def _cost(model: str | None, tokens_in: int, tokens_out: int) -> float | None:
    """litellm's price-table estimate, as compare_agents.py reports it."""
    if not model:
        return None
    os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")     # litellm loads .env on import
    try:
        import litellm
        prompt, completion = litellm.cost_per_token(model=model, prompt_tokens=tokens_in,
                                                    completion_tokens=tokens_out)
        return prompt + completion
    except Exception:
        return None


def rehearse(server, acts, repeat: int = 1, out=print, keep_threads: bool = False) -> list[dict]:
    results = []
    try:
        for n in range(1, repeat + 1):
            for act in acts:
                server.reset()
                out(printable(f"\n[pass {n}] {act.key}: {act.title}"))
                result = run_act(server, act, out=lambda line: out(printable(line)))
                out(f"  {'PASS' if result['ok'] else 'FAIL'}: {act.expect}"
                    f"   ({result['model_calls']} model calls, {result['tokens']} tokens, "
                    f"{result['seconds']}s)")
                results.append({"pass": n, **result})
    finally:
        if not keep_threads:
            for r in results:
                server.delete_thread(r["thread"])
        server.reset()
    return results


def summary(results: list[dict], acts) -> str:
    lines = ["", "| act | passes | model calls | tokens | seconds |", "|---|---|---|---|---|"]
    for act in acts:
        rows = [r for r in results if r["act"] == act.key]
        med = lambda key: statistics.median(r[key] for r in rows)    # noqa: E731
        lines.append(f"| {act.key} | {sum(r['ok'] for r in rows)}/{len(rows)} | {med('model_calls'):g} "
                     f"| {med('tokens'):g} | {med('seconds'):g} |")
    costs = [r["cost"] for r in results if r.get("cost") is not None]
    lines.append(f"| **total** | **{sum(r['ok'] for r in results)}/{len(results)}** | | "
                 f"{sum(r['tokens'] for r in results)} | {sum(r['seconds'] for r in results):.0f} |")
    if costs:
        passes = max(r["pass"] for r in results)
        lines.append(f"\nabout ${sum(costs) / passes:.3f} a pass (litellm's price table)")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Play the demo script against the running server.")
    p.add_argument("--url", default="http://127.0.0.1:2024", help="the LangGraph server")
    p.add_argument("--repeat", type=int, default=1, help="passes, each from the starting data")
    p.add_argument("--act", action="append", choices=[a.key for a in script.ACTS],
                   help="only these acts (repeatable); default: all, in order")
    p.add_argument("--keep-threads", action="store_true",
                   help="keep the rehearsal's threads (they show in the chat's history)")
    args = p.parse_args(argv)
    acts = [a for a in script.ACTS if not args.act or a.key in args.act]
    server = Server(args.url)
    try:
        server.snapshot()
    except Exception as exc:
        print(f"No demo server at {args.url} ({type(exc).__name__}). Start it with demo/run.sh.")
        return 2
    results = rehearse(server, acts, args.repeat, keep_threads=args.keep_threads)
    print(summary(results, acts))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
