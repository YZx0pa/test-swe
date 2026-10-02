#!/usr/bin/env python3
"""recruiter-cli — the ONLY tool mini-swe-agent may use.

mini calls this via bash.  Every recruiter capability is one subcommand.
All domain "weight" lives HERE (outside mini's core, where the model can't
reach it): endpoint routing, query-vs-body split, auth headers from env,
PII masking, an audit log, and a confirm-gate for dangerous actions.

The same guarded path is importable: execute() and the typed actions
(find_talents, generate_jd, score_candidates, candidate_insights,
get_match_id_from_profile_id) back vira_tools.py (LangGraph, deepagents, MCP)
and run_workflow.py.

Design notes tied to the four real curls you gave:
  * base url + 3 header auth (x-api-key / x-client-name / x-user-id), read
    from env, NEVER placed in model context.
  * some params go in the URL QUERY (?lang=ar, ?version=v3, ?composite_score=..)
    and some in the JSON BODY.  Each subcommand declares which is which.
  * #3 and #4 hit the SAME endpoint (candidate_score_calculation) with
    different query params, so they are two subcommands, one path.
  * #5 get-match-id-from-profile-id is an ASSUMED endpoint (no curl yet):
    mock mode answers it; real mode sends it to VIRA like the others.

Run modes (--mode is required, so nothing reaches VIRA by default):
  --mode real  -> calls VIRA at $VIRA_BASE_URL (your localhost).
  --mode mock  -> calls a local fake (mock_vira.MockVira) so the whole mini
                  loop can run with no backend.
Under mini-swe-agent the host adds --mode (mini_env.RecruiterEnvironment); the
model can't choose it.
"""
from __future__ import annotations

import argparse
import datetime
import ipaddress
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

from dotenv import load_dotenv

import pii_vault

DOTENV = Path(__file__).with_name(".env")    # this file's .env, not whichever a parent dir holds


def _warn_if_shared(path: Path) -> None:
    """Warn when a secrets file is readable by other users.  Checks mode bits only."""
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if os.name == "posix" and mode & 0o077:
        print(f"recruiter_cli: {path.name} is readable by other users; run: chmod 600 {path}",
              file=sys.stderr)


if load_dotenv(DOTENV):
    _warn_if_shared(DOTENV)
# --- config from environment (secrets never touch model context) ------------
VIRA_BASE_URL = os.environ.get("VIRA_BASE_URL", "http://localhost:8080/v1")
HEADERS = {
    "x-api-key": os.environ.get("VIRA_API_KEY", ""),
    "x-client-name": os.environ.get("VIRA_CLIENT_NAME", ""),
    "x-user-id": os.environ.get("VIRA_USER_ID", ""),
    "Content-Type": "application/json",
}
# Jeni's task groups go to the VIRA engine instead: its own location (the full endpoint URL, or
# a path under VIRA_BASE_URL), authenticated as one user by that user's xrtoken.
TASK_GROUP_PATH = "agent_task_group"
VIRA_ACTUAL_LOCATION = os.environ.get("VIRA_ACTUAL_LOCATION", "")
VIRA_XRTOKEN = os.environ.get("VIRA_XRTOKEN", "")
EVENTS_LOG = os.environ.get("EVENTS_LOG", "events.jsonl")
# A private CA for VIRA, since the environment's REQUESTS_CA_BUNDLE is ignored (trust_env=False).
VIRA_CA_BUNDLE = os.environ.get("VIRA_CA_BUNDLE") or None
RAW_LIMIT = 500          # characters of a non-JSON reply that reach the caller

# Input limits, checked before anything is sent (vira_tools.py puts the same numbers
# in the tool schemas).  Scoring and JD generation run LLM work on VIRA per id / per text.
MAX_IDS = 50             # ids per argument
MAX_TEXT = 200           # characters per text value
MAX_ITEMS = 30           # entries per list argument
LANG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")    # en, ar, zh-CN, zh-Hans

# Dangerous subcommands require --confirmed (structural gate, not a prompt
# request).  The endpoints here are read/compute only, so this is
# empty; add names here the moment a write/notify/irreversible action is added.
NEEDS_CONFIRM: set[str] = set()
# Commands whose results come from the company's user directory: emails in them are colleagues'.
DIRECTORY_COMMANDS = {"search-users"}

# Fields whose values may ONLY come from explicit user input — the model is
# never allowed to invent them.  (None needed for these; kept for parity
# with the CRUD tool where app_ids/emails live.)
USER_ONLY_FIELDS: set[str] = set()


# --- helpers ---------------------------------------------------------------
def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


# --- PII masking --------------------------------------------------------------
# A value is masked when its key (normalised: camelCase and "-" become "_", lower case)
# is in PII_KEYS or matches one of the patterns.  Extend them when VIRA returns a new
# sensitive field.  Whole "_"-separated words only, so job_name_similarity stays.
PII_KEYS = {"email", "emails", "name", "names", "candidate_name", "candidate_email",
            "phone", "phones", "resume", "cv", "address", "dob", "date_of_birth",
            "birth_date", "birthday", "nationality", "gender"}
_PII_WORD = re.compile(r"(^|_)(e?mails?|phones?|mobile|tel|telephone|address(es)?|resume|cv|"
                       r"dob|nric|ssn|passport|linkedin|photo|avatar)(_|$)")
_PII_NAME = re.compile(r"^(first|last|full|middle|given|family|sur|user|candidate|applicant|"
                       r"contact|display|person|legal|creator|owner)_?names?$")
# Inside any string value (free-text summaries, a non-JSON reply): emails, and phone
# numbers written in groups or with a "+" (8+ digits; bare digit runs, like ids, stay).
_EMAIL = pii_vault.EMAIL
_PHONE = re.compile(r"(?<![\w+])(?<!\d\.)(?:\+\d{1,3}[ .-]?)?(?:\(\d{1,4}\)[ .-]?)?"
                    r"\d{2,4}(?:[ .-]\d{2,5}){1,4}(?!\w|\.\d)")      # not inside a decimal
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}$")
# A uuid's digit groups can look like a phone number ("8969-4271"); uuids are ids, not PII.
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def _is_pii_key(key: Any) -> bool:
    k = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower().replace("-", "_").replace(" ", "_")
    return k in PII_KEYS or bool(_PII_WORD.search(k) or _PII_NAME.match(k))


def _redact_phone(m: re.Match) -> str:
    text = m.group(0)
    if sum(c.isdigit() for c in text) < 8 or _DATE.match(text):
        return text
    return "<redacted-phone>"


def _scrub_text(text: str, source: str | None = None) -> str:
    """Emails as tokens the tools can send back (pii_vault) when `source` says where the text came
    from, redacted otherwise; phone numbers always redacted."""
    text = pii_vault.VAULT.tokenize(text, source) if source else _EMAIL.sub("<redacted-email>", text)
    parts, last = [], 0
    for m in _UUID.finditer(text):           # phones are looked for between uuids, never in one
        parts += [_PHONE.sub(_redact_phone, text[last:m.start()]), m.group(0)]
        last = m.end()
    return "".join(parts) + _PHONE.sub(_redact_phone, text[last:])


def _mask_pii(obj: Any, source: str | None = None) -> Any:
    """Small PII masker for anything that flows back into model context.

    Real deployment: swap this for your existing vault mask.  Here it redacts
    candidate fields by key name and pattern, and emails and phone numbers
    inside any string.  Names inside free text are not detected.  A
    {"field_name": ..., "field_value": ...} pair (Jeni's task payloads) counts
    as a key and its value.

    With a `source` (pii_vault.COLLEAGUE for the user directory, RECORD for anything else), each
    email address, under an email key or inside text, becomes a pii_vault token instead: the
    model still never sees it, and a tool can send it back.  The audit log passes no source.
    """
    if isinstance(obj, dict):
        pair_is_pii = "field_value" in obj and _is_pii_key(obj.get("field_name", ""))
        return {k: (_mask_pii(v, source) if source and _is_email_key(k) and _EMAIL.search(str(v))
                    else "<redacted>" if _is_pii_key(k) or (pair_is_pii and k == "field_value"
                                                          and v is not None)
                    else _mask_pii(v, source)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_pii(x, source) for x in obj]
    if isinstance(obj, str):
        return _scrub_text(obj, source)
    return obj


def _is_email_key(key: Any) -> bool:
    return "email" in str(key).lower()


def _audit(cmd: str, query: Dict, body: Dict, result: Dict, error: str | None = None) -> None:
    entry = {"ts": _now(), "command": cmd, "query": _mask_pii(query), "body": _mask_pii(body),
             "status": result.get("status", "ok")}
    if error:
        entry["error"] = error
    # Owner-only: the log holds request bodies.  fchmod also tightens a log created before
    # this change, but only a regular file (tests and compare_agents point it at os.devnull).
    fd = os.open(EVENTS_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    if hasattr(os, "fchmod") and stat.S_ISREG(os.fstat(fd).st_mode):
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass                          # not our file to chmod; still append
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _guard(cmd: str, confirmed: bool) -> Dict | None:
    """Return the needs_confirmation result for a gated, unconfirmed command."""
    if cmd in NEEDS_CONFIRM and not confirmed:
        return {
            "status": "needs_confirmation",
            "message": (f"'{cmd}' affects candidates or is irreversible and needs a "
                        f"person's approval, which the agent cannot give. Tell the user "
                        f"exactly what would happen and to whom, and do not retry."),
        }
    return None


def _emit(result: Dict) -> None:
    """Print result for mini to read — PII-masked before it enters context.  Each CLI call is its own
    process, so an email token couldn't be sent back by the next one: emails are redacted here."""
    print(json.dumps(_mask_pii(result), ensure_ascii=False))


def _csv_int(s: str | None) -> list[int]:
    # A non-number stays a string, so _input_problem() can say what's wrong with it.
    return [int(x) if x.strip().isdigit() else x.strip() for x in s.split(",") if x.strip()] if s else []


def _csv_str(s: str | None) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


# --- the actual VIRA call (the ONE line you swap real<->mock) ---------------
def _is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def _url_problem(url: str, setting: str) -> str | None:
    parts = urlsplit(url)
    if not parts.hostname or not (parts.scheme == "https"
                                  or (parts.scheme == "http" and _is_loopback(parts.hostname))):
        return f"{setting} must be https:// (plain http only to localhost)"
    return None


def _task_group_url() -> str:
    location = VIRA_ACTUAL_LOCATION.strip()
    if urlsplit(location).scheme:
        return location
    return f"{VIRA_BASE_URL.rstrip('/')}/{location.lstrip('/')}"


def _target(path: str) -> tuple[str, Dict[str, str]]:
    """(url, headers) of a real call: the engine for task groups, VIRA_BASE_URL otherwise."""
    if path == TASK_GROUP_PATH:
        return _task_group_url(), {"xrtoken": VIRA_XRTOKEN, "Content-Type": "application/json"}
    return f"{VIRA_BASE_URL}/{path}", HEADERS


def _real_mode_problem(path: str = "") -> str | None:
    """Why a real call to `path` must not be sent, or None.  Never names the host."""
    if path == TASK_GROUP_PATH:
        if not VIRA_ACTUAL_LOCATION.strip():
            return "VIRA engine not configured: VIRA_ACTUAL_LOCATION"
        problem = _url_problem(_task_group_url(), "VIRA_ACTUAL_LOCATION")
        if problem:
            return problem
        return None if VIRA_XRTOKEN else "VIRA credentials not configured: VIRA_XRTOKEN"
    problem = _url_problem(VIRA_BASE_URL, "VIRA_BASE_URL")
    if problem:
        return problem
    missing = [name for name, header in (("VIRA_API_KEY", "x-api-key"),
                                         ("VIRA_CLIENT_NAME", "x-client-name"),
                                         ("VIRA_USER_ID", "x-user-id")) if not HEADERS[header]]
    if missing:
        return f"VIRA credentials not configured: {', '.join(missing)}"
    return None


def _error(message: str, http_status: int | None = None) -> Dict:
    return {"status": "error", "http_status": http_status, "result": {"message": message}}


def _call(path: str, query: Dict[str, Any], body: Dict[str, Any], mode: str) -> Dict:
    try:              # the decrypt step: email tokens back to addresses, just before sending
        query, body = pii_vault.VAULT.resolve(query), pii_vault.VAULT.resolve(body)
    except pii_vault.UnknownToken:
        return _error("an email token this conversation didn't receive; ask the user for the address")
    if mode == "mock":
        from mock_vira import MockVira
        return MockVira.call(path, query, body)
    # real mode
    problem = _real_mode_problem(path)
    if problem:
        return _error(problem)
    import requests  # imported lazily so mock mode needs no dependency
    url, headers = _target(path)
    with requests.Session() as session:
        # No proxies, .netrc or CA bundle from the environment (or from .env via it).
        session.trust_env = False
        session.verify = VIRA_CA_BUNDLE or True
        # No redirects: requests strips only `Authorization` on a cross-host redirect, so
        # x-api-key / x-user-id / xrtoken (and, on 307/308, the body) would follow one anywhere.
        resp = session.post(url, headers=headers, params=query,
                            json=body, timeout=30, allow_redirects=False)
    if 300 <= resp.status_code < 400:
        return _error("VIRA answered with a redirect, which is not followed", resp.status_code)
    try:
        return {"status": "ok" if resp.ok else "error",
                "http_status": resp.status_code, "result": resp.json()}
    except Exception:
        return {"status": "error", "http_status": resp.status_code,
                "result": {"raw": resp.text[:RAW_LIMIT]}}


def _input_problem(*, ids: Dict[str, list] | None = None, texts: Dict[str, str] | None = None,
                   lists: Dict[str, list] | None = None, lang: str | None = None) -> str | None:
    """Why these arguments must not be sent to VIRA, or None."""
    for name, values in (ids or {}).items():
        if len(values) > MAX_IDS:
            return f"{name}: at most {MAX_IDS} ids per call"
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            return f"{name}: ids are positive integers"
    for name, value in (texts or {}).items():
        if len(value) > MAX_TEXT:
            return f"{name}: at most {MAX_TEXT} characters"
    for name, values in (lists or {}).items():
        if len(values) > MAX_ITEMS:
            return f"{name}: at most {MAX_ITEMS} entries"
        if any(len(v) > MAX_TEXT for v in values):
            return f"{name}: at most {MAX_TEXT} characters per entry"
    if lang is not None and not LANG_RE.match(lang):
        return "lang: a language code such as en, ar or zh-CN"
    return None


def execute(cmd: str, path: str, query: Dict, body: Dict, *, mode: str,
            confirmed: bool = False) -> Dict:
    """The one guarded path to VIRA: confirm-gate -> call -> audit -> PII mask.

    Every runtime goes through here: this CLI (mini-swe-agent), vira_tools.py
    (LangGraph, deepagents, MCP) and run_workflow.py.  `mode` is required so no
    caller reaches the real backend by default.  Returns the masked result and
    never prints (vira_mcp.py's stdout is its JSON-RPC channel).
    """
    blocked = _guard(cmd, confirmed)
    if blocked:
        return blocked
    try:
        result = _call(path, query, body, mode)
    except Exception as exc:          # e.g. VIRA unreachable: audited, then the caller decides
        _audit(cmd, query, body, {"status": "exception"}, error=type(exc).__name__)
        raise
    _audit(cmd, query, body, result)
    return _mask_pii(result, pii_vault.COLLEAGUE if cmd in DIRECTORY_COMMANDS else pii_vault.RECORD)


# --- typed actions (one per API you gave) — the importable surface ----------
def find_talents(job_ids: list[int], profile_ids: list[int] | None = None, *,
                 mode: str, confirmed: bool = False) -> Dict:
    # #1 fast_retargeting — params all in BODY
    problem = _input_problem(ids={"job_ids": job_ids, "profile_ids": profile_ids or []})
    if problem:
        return _error(problem)
    return execute("find-talents", "fast_retargeting",
                   query={},
                   body={"job_ids": job_ids,
                         "profile_ids": profile_ids or []},
                   mode=mode, confirmed=confirmed)


def generate_jd(job_title: str, skills: list[str] | None = None, lang: str = "en",
                job_id: int = 0, job_function: list[str] | None = None,
                industry: list[str] | None = None,
                other_requirements: list[str] | None = None, *,
                mode: str, confirmed: bool = False) -> Dict:
    # #2 JD_generation — lang in QUERY, rest in BODY
    problem = _input_problem(ids={"job_id": [job_id] if job_id else []},
                             texts={"job_title": job_title},
                             lists={"skills": skills or [], "job_function": job_function or [],
                                    "industry": industry or [],
                                    "other_requirements": other_requirements or []},
                             lang=lang)
    if problem:
        return _error(problem)
    return execute("generate-jd", "JD_generation/jd_generation",
                   query={"lang": lang},
                   body={"job_id": job_id or 0,
                         "job_title": job_title,
                         "skills": skills or [],
                         "job_function": job_function or [],
                         "industry": industry or [],
                         "other_requirements": other_requirements or []},
                   mode=mode, confirmed=confirmed)


def score_candidates(app_ids: list[int] | None = None, match_ids: list[int] | None = None,
                     *, mode: str, confirmed: bool = False) -> Dict:
    # #3 candidate_score_calculation — scoring flags in QUERY, ids in BODY
    problem = _input_problem(ids={"app_ids": app_ids or [], "match_ids": match_ids or []})
    if problem:
        return _error(problem)
    return execute("score-candidates", "candidate_score_calculation",
                   query={"composite_score": "True", "briq": "True", "recal_briq": "True"},
                   body={"app_ids": app_ids or [],
                         "match_ids": match_ids or []},
                   mode=mode, confirmed=confirmed)


def candidate_insights(app_ids: list[int] | None = None, match_ids: list[int] | None = None,
                       *, mode: str, confirmed: bool = False) -> Dict:
    # #4 candidate_score_calculation?version=v3 — SAME path, different query
    problem = _input_problem(ids={"app_ids": app_ids or [], "match_ids": match_ids or []})
    if problem:
        return _error(problem)
    return execute("candidate-insights", "candidate_score_calculation",
                   query={"version": "v3"},
                   body={"app_ids": app_ids or [],
                         "match_ids": match_ids or []},
                   mode=mode, confirmed=confirmed)


def get_match_id_from_profile_id(job_id: int, profile_ids: list[int], *,
                                 mode: str, confirmed: bool = False) -> Dict:
    # #5 ASSUMED endpoint: VIRA doesn't have it yet, so only --mode mock answers it.
    # Params all in BODY.  A match is one profile for one job, hence the job_id.
    problem = _input_problem(ids={"job_id": [job_id], "profile_ids": profile_ids})
    if problem:
        return _error(problem)
    return execute("get-match-id-from-profile-id", "get_match_id_from_profile_id",
                   query={},
                   body={"job_id": job_id,
                         "profile_ids": profile_ids},
                   mode=mode, confirmed=confirmed)


# --- CLI subcommands: parse the CSV flags, call the typed action -------------
def _finish(result: Dict) -> None:
    _emit(result)
    if result.get("status") == "needs_confirmation":
        sys.exit(2)


def _opts(ns: argparse.Namespace) -> Dict[str, Any]:
    return {"mode": ns.mode, "confirmed": getattr(ns, "confirmed", False)}


def cmd_find_talents(ns):
    _finish(find_talents(_csv_int(ns.job_ids), _csv_int(ns.profile_ids), **_opts(ns)))


def cmd_generate_jd(ns):
    _finish(generate_jd(job_title=ns.job_title, skills=_csv_str(ns.skills), lang=ns.lang,
                        job_id=ns.job_id, job_function=_csv_str(ns.job_function),
                        industry=_csv_str(ns.industry),
                        other_requirements=_csv_str(ns.other_requirements), **_opts(ns)))


def cmd_score_candidates(ns):
    _finish(score_candidates(_csv_int(ns.app_ids), _csv_int(ns.match_ids), **_opts(ns)))


def cmd_candidate_insights(ns):
    _finish(candidate_insights(_csv_int(ns.app_ids), _csv_int(ns.match_ids), **_opts(ns)))


def cmd_get_match_id_from_profile_id(ns):
    _finish(get_match_id_from_profile_id(ns.job_id, _csv_int(ns.profile_ids), **_opts(ns)))


# --- argparse wiring --------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: "--mo real" must not pass for --mode.
    p = argparse.ArgumentParser(prog="recruiter-cli", allow_abbrev=False,
                                description="Recruiter AI actions for the agent.")
    p.add_argument("--mode", choices=["real", "mock"], required=True,
                   help="real: call VIRA; mock: call local fake backend")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("find-talents", allow_abbrev=False,
                       help="Find potential talents for a job (read).")
    s.add_argument("--job-ids", dest="job_ids", required=True, help="csv of job ids")
    s.add_argument("--profile-ids", dest="profile_ids", default="", help="csv of profile ids")
    s.set_defaults(func=cmd_find_talents)

    s = sub.add_parser("generate-jd", allow_abbrev=False,
                       help="LLM-generate a job posting (read/compute).")
    s.add_argument("--job-title", dest="job_title", required=True)
    s.add_argument("--skills", default="", help="csv of skills")
    s.add_argument("--lang", default="en", help="language code, e.g. ar, en")
    s.add_argument("--job-id", dest="job_id", type=int, default=0)
    s.add_argument("--job-function", dest="job_function", default="")
    s.add_argument("--industry", default="")
    s.add_argument("--other-requirements", dest="other_requirements", default="")
    s.set_defaults(func=cmd_generate_jd)

    s = sub.add_parser("score-candidates", allow_abbrev=False,
                       help="Trigger AI scoring for applicants/suggested talents (compute).")
    s.add_argument("--app-ids", dest="app_ids", default="", help="csv of application ids")
    s.add_argument("--match-ids", dest="match_ids", default="", help="csv of match ids")
    s.set_defaults(func=cmd_score_candidates)

    s = sub.add_parser("candidate-insights", allow_abbrev=False,
                       help="Trigger candidate insights for applicants (compute).")
    s.add_argument("--app-ids", dest="app_ids", default="", help="csv of application ids")
    s.add_argument("--match-ids", dest="match_ids", default="", help="csv of match ids")
    s.set_defaults(func=cmd_candidate_insights)

    s = sub.add_parser("get-match-id-from-profile-id", allow_abbrev=False,
                       help="Look up match ids of suggested talents for a job (read; assumed).")
    s.add_argument("--job-id", dest="job_id", type=int, required=True)
    s.add_argument("--profile-ids", dest="profile_ids", required=True, help="csv of profile ids")
    s.set_defaults(func=cmd_get_match_id_from_profile_id)
    return p


def main(argv=None):
    ns = build_parser().parse_args(argv)
    try:
        ns.func(ns)
    except Exception as exc:   # SystemExit isn't one: argparse and the confirm gate exit as before
        # One JSON line and no traceback, which could name hosts or paths.
        _emit({"status": "error", "message": f"recruiter_cli failed ({type(exc).__name__})"})
        sys.exit(1)


if __name__ == "__main__":
    main()
