#!/usr/bin/env python3
"""recruiter-cli — the ONLY tool mini-swe-agent may use.

mini calls this via bash.  Every recruiter capability is one subcommand.
All domain "weight" lives HERE (outside mini's core, where the model can't
reach it): endpoint routing, query-vs-body split, auth headers from env,
PII masking, an audit log, and a confirm-gate for dangerous actions.

Design notes tied to the four real curls you gave:
  * base url + 3 header auth (x-api-key / x-client-name / x-user-id), read
    from env, NEVER placed in model context.
  * some params go in the URL QUERY (?lang=ar, ?version=v3, ?composite_score=..)
    and some in the JSON BODY.  Each subcommand declares which is which.
  * #3 and #4 hit the SAME endpoint (candidate_score_calculation) with
    different query params, so they are two subcommands, one path.

Run modes:
  --mode real  -> calls VIRA at $VIRA_BASE_URL (your localhost).  Default.
  --mode mock  -> calls a local fake (mock_vira.MockVira) so the whole mini
                  loop can run with no backend.  Used by run_demo.py.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv() 
# --- config from environment (secrets never touch model context) ------------
VIRA_BASE_URL = os.environ.get("VIRA_BASE_URL", "http://localhost:8080/v1")
HEADERS = {
    "x-api-key": os.environ.get("VIRA_API_KEY", ""),
    "x-client-name": os.environ.get("VIRA_CLIENT_NAME", ""),
    "x-user-id": os.environ.get("VIRA_USER_ID", ""),
    "Content-Type": "application/json",
}
EVENTS_LOG = os.environ.get("EVENTS_LOG", "events.jsonl")

# Dangerous subcommands require --confirmed (structural gate, not a prompt
# request).  The four AI endpoints here are read/compute only, so this is
# empty; add names here the moment a write/notify/irreversible action is added.
NEEDS_CONFIRM: set[str] = set()

# Fields whose values may ONLY come from explicit user input — the model is
# never allowed to invent them.  (None needed for these four; kept for parity
# with the CRUD tool where app_ids/emails live.)
USER_ONLY_FIELDS: set[str] = set()


# --- helpers ---------------------------------------------------------------
def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _mask_pii(obj: Any) -> Any:
    """Very small PII masker for anything that flows back into model context.

    Real deployment: swap this for your existing vault mask.  Here it redacts
    common candidate fields by key name.
    """
    PII_KEYS = {"email", "emails", "name", "candidate_name", "candidate_email",
                "phone", "resume", "cv"}
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k.lower() in PII_KEYS else _mask_pii(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_pii(x) for x in obj]
    return obj
# def cmd_get_match_id(ns):
#     # "assume this endpoint exists" — synthesize a real-shaped response locally,
#     # even in --mode real, since the server may not have it yet.
#     pids = _csv_int(ns.profile_ids)
#     result = {
#         "status": "ok",
#         "http_status": 200,
#         "result": {
#             "matches": [{"profile_id": p, "match_id": p + 500000} for p in pids]
#         },
#     }
#     _audit("get-match-id", {}, {"profile_ids": pids}, result)   # keep it in the audit log
#     _emit(result)

def _audit(cmd: str, query: Dict, body: Dict, result: Dict) -> None:
    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": _now(), "command": cmd,
            "query": query, "body": _mask_pii(body),
            "status": result.get("status", "ok"),
        }, ensure_ascii=False) + "\n")


def _guard(cmd: str, ns: argparse.Namespace) -> None:
    if cmd in NEEDS_CONFIRM and not getattr(ns, "confirmed", False):
        _emit({
            "status": "needs_confirmation",
            "message": (f"'{cmd}' affects candidates or is irreversible and "
                        f"needs explicit user confirmation. Tell the user exactly "
                        f"what will happen and to whom, get agreement, then retry "
                        f"with --confirmed."),
        })
        sys.exit(2)


def _emit(result: Dict) -> None:
    """Print result for mini to read — PII-masked before it enters context."""
    print(json.dumps(_mask_pii(result), ensure_ascii=False))


def _csv_int(s: str | None) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def _csv_str(s: str | None) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


# --- the actual VIRA call (the ONE line you swap real<->mock) ---------------
def _call(path: str, query: Dict[str, Any], body: Dict[str, Any], mode: str) -> Dict:
    if mode == "mock":
        from mock_vira import MockVira
        return MockVira.call(path, query, body)
    # real mode
    import requests  # imported lazily so mock mode needs no dependency
    # decrypt_pii(body) would go here in your deployment (vault -> real values)
    resp = requests.post(f"{VIRA_BASE_URL}/{path}", headers=HEADERS,
                         params=query, json=body, timeout=30)
    try:
        return {"status": "ok" if resp.ok else "error",
                "http_status": resp.status_code, "result": resp.json()}
    except Exception:
        return {"status": "error", "http_status": resp.status_code,
                "result": {"raw": resp.text}}


def _run(cmd: str, path: str, query: Dict, body: Dict, ns: argparse.Namespace) -> None:
    _guard(cmd, ns)
    result = _call(path, query, body, ns.mode)
    _audit(cmd, query, body, result)
    _emit(result)


# --- subcommands (one per API you gave) ------------------------------------
def cmd_find_talents(ns):
    # #1 fast_retargeting — params all in BODY
    _run("find-talents", "fast_retargeting",
         query={},
         body={"job_ids": _csv_int(ns.job_ids),
               "profile_ids": _csv_int(ns.profile_ids)},
         ns=ns)


def cmd_generate_jd(ns):
    # #2 JD_generation — lang in QUERY, rest in BODY
    _run("generate-jd", "JD_generation/jd_generation",
         query={"lang": ns.lang},
         body={"job_id": ns.job_id or 0,
               "job_title": ns.job_title,
               "skills": _csv_str(ns.skills),
               "job_function": _csv_str(ns.job_function),
               "industry": _csv_str(ns.industry),
               "other_requirements": _csv_str(ns.other_requirements)},
         ns=ns)


def cmd_score_candidates(ns):
    # #3 candidate_score_calculation — scoring flags in QUERY, ids in BODY
    _run("score-candidates", "candidate_score_calculation",
         query={"composite_score": "True", "briq": "True", "recal_briq": "True"},
         body={"app_ids": _csv_int(ns.app_ids),
               "match_ids": _csv_int(ns.match_ids)},
         ns=ns)


def cmd_candidate_insights(ns):
    # #4 candidate_score_calculation?version=v3 — SAME path, different query
    _run("candidate-insights", "candidate_score_calculation",
         query={"version": "v3"},
         body={"app_ids": _csv_int(ns.app_ids),
               "match_ids": _csv_int(ns.match_ids)},
         ns=ns)


# --- argparse wiring --------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recruiter-cli",
                                description="Recruiter AI actions for the agent.")
    p.add_argument("--mode", choices=["real", "mock"], default="real",
                   help="real: call VIRA; mock: call local fake backend")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("find-talents", help="Find potential talents for a job (read).")
    s.add_argument("--job-ids", dest="job_ids", required=True, help="csv of job ids")
    s.add_argument("--profile-ids", dest="profile_ids", default="", help="csv of profile ids")
    s.set_defaults(func=cmd_find_talents)

    s = sub.add_parser("generate-jd", help="LLM-generate a job posting (read/compute).")
    s.add_argument("--job-title", dest="job_title", required=True)
    s.add_argument("--skills", default="", help="csv of skills")
    s.add_argument("--lang", default="en", help="language code, e.g. ar, en")
    s.add_argument("--job-id", dest="job_id", type=int, default=0)
    s.add_argument("--job-function", dest="job_function", default="")
    s.add_argument("--industry", default="")
    s.add_argument("--other-requirements", dest="other_requirements", default="")
    s.set_defaults(func=cmd_generate_jd)

    s = sub.add_parser("score-candidates",
                       help="Trigger AI scoring for applicants/suggested talents (compute).")
    s.add_argument("--app-ids", dest="app_ids", default="", help="csv of application ids")
    s.add_argument("--match-ids", dest="match_ids", default="", help="csv of match ids")
    s.set_defaults(func=cmd_score_candidates)

    s = sub.add_parser("candidate-insights",
                       help="Trigger candidate insights for applicants (compute).")
    s.add_argument("--app-ids", dest="app_ids", default="", help="csv of application ids")
    s.add_argument("--match-ids", dest="match_ids", default="", help="csv of match ids")
    s.set_defaults(func=cmd_candidate_insights)

    # s = sub.add_parser("get-match-id", help="Convert profile_ids to match_ids (assumed endpoint).")
    # s.add_argument("--profile-ids", dest="profile_ids", required=True)
    # s.set_defaults(func=cmd_get_match_id)
    return p


def main(argv=None):
    ns = build_parser().parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
