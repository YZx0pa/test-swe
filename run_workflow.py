#!/usr/bin/env python3
"""VIRA as an explicit LangGraph workflow: code drives, the LLM only summarises.

    score applicants -> shortlist top-k (in code) -> human approval (interrupt)
        -> candidate insights for the shortlist -> one LLM call writes a summary

Where run_langgraph.py lets the model choose every step, this graph fixes the
steps and their order; the model can't skip the approval or pass the wrong id
type, because it never picks a tool.  The checkpointer holds the state while
the graph waits at the approval step, so resuming continues from there.

    python run_workflow.py --app-ids 11,12,13 --top 2          # mock VIRA
    python run_workflow.py --app-ids 11,12,13 --top 2 --yes --no-llm
"""
import argparse
import json
import uuid
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import agent_kit
import vira_tools
from terminal import printable


class Review(TypedDict, total=False):
    app_ids: list[int]
    top: int
    scores: list[dict]
    shortlist: list[int]
    approved: bool
    insights: list[dict]
    summary: str
    error: str


SUMMARY_PROMPT = ("You write for a recruiter. In at most 4 short lines, summarise the scoring "
                  "and insights below: who was shortlisted and why, and whether insights ran. "
                  "Use only the data given; do not invent facts.")


def score(state: Review) -> dict:
    res = vira_tools.score_candidates(app_ids=state["app_ids"])
    if res.get("status") != "ok":
        return {"error": f"scoring failed: {json.dumps(res, ensure_ascii=False)}"}
    return {"scores": res["result"].get("scores", [])}


def shortlist(state: Review) -> dict:
    ranked = sorted((s for s in state["scores"] if "app_id" in s),
                    key=lambda s: s.get("composite_score") or 0, reverse=True)   # stable on ties
    return {"shortlist": [s["app_id"] for s in ranked[:state.get("top", 2)]]}


def approve(state: Review) -> Command[Literal["insights", "summarize"]]:
    # A node re-runs from the top when resumed: nothing before interrupt() may have side effects.
    decision = interrupt({"question": f"Run candidate insights for applicants {state['shortlist']}?",
                          "shortlist": state["shortlist"]})
    ok = bool(decision.get("approve")) if isinstance(decision, dict) else bool(decision)
    return Command(update={"approved": ok}, goto="insights" if ok else "summarize")


def insights(state: Review) -> dict:
    res = vira_tools.candidate_insights(app_ids=state["shortlist"])
    if res.get("status") != "ok":
        return {"error": f"insights failed: {json.dumps(res, ensure_ascii=False)}"}
    return {"insights": res["result"].get("insights", [])}


def make_summarize(model):
    def summarize(state: Review) -> dict:
        facts = {k: state.get(k) for k in ("scores", "shortlist", "approved", "insights", "error")}
        if model is None:
            return {"summary": (f"Shortlisted {state.get('shortlist')}; insights "
                                f"{'ran' if state.get('approved') else 'not run (not approved)'}.")}
        reply = model.invoke([SystemMessage(SUMMARY_PROMPT),
                              HumanMessage(json.dumps(facts, ensure_ascii=False))])
        return {"summary": reply.text}
    return summarize


def build_graph(model=None):
    g = StateGraph(Review)
    g.add_node("score", score)
    g.add_node("shortlist", shortlist)
    g.add_node("approve", approve)
    g.add_node("insights", insights)
    g.add_node("summarize", make_summarize(model))
    g.add_edge(START, "score")
    g.add_conditional_edges("score", lambda s: END if s.get("error") else "shortlist",
                            ["shortlist", END])
    g.add_edge("shortlist", "approve")
    g.add_edge("insights", "summarize")
    g.add_edge("summarize", END)
    return g.compile(checkpointer=InMemorySaver())


def run(graph, app_ids: list[int], top: int, decide) -> dict:
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    out = graph.invoke({"app_ids": app_ids, "top": top}, config, version="v2")
    while out.interrupts:
        out = graph.invoke(Command(resume={"approve": decide(out.interrupts[0].value)}),
                           config, version="v2")
    return out.value


def main(argv=None):
    p = argparse.ArgumentParser(description="VIRA review workflow on an explicit LangGraph graph.")
    p.add_argument("--app-ids", required=True, help="csv of application ids to score")
    p.add_argument("--top", type=int, default=2, help="how many applicants to shortlist")
    p.add_argument("--mode", choices=["real", "mock"], default="mock")
    p.add_argument("--yes", action="store_true", help="approve the shortlist without asking")
    p.add_argument("--no-llm", action="store_true", help="template summary instead of an LLM call")
    p.add_argument("--trace", action="store_true", help="allow LangSmith tracing (off by default)")
    args = p.parse_args(argv)
    agent_kit.set_tracing(args.trace)
    vira_tools.configure(args.mode)

    def decide(request: dict) -> bool:
        print(f"\n[approval] {request['question']}")
        return args.yes or input("approve? [y/N] > ").strip().lower().startswith("y")

    graph = build_graph(model=None if args.no_llm else agent_kit.build_chat_model())
    state = run(graph, [int(x) for x in args.app_ids.split(",") if x.strip()], args.top, decide)
    print("\n=== workflow result ===")
    for key in ("scores", "shortlist", "approved", "insights", "error", "summary"):
        if key in state:
            print(printable(f"{key:9}: {json.dumps(state[key], ensure_ascii=False)}"))


if __name__ == "__main__":
    main()
