#!/usr/bin/env python3
"""Jeni's VIRA tasks as typed tools: one task per call, the result read before the next.

Jeni v1 asks the model for a whole plan in one reply (every task and field, with
{{output_from:...}} placeholders for values an earlier task would produce) and hands
the plan to VIRA.  Here each task in v1's catalog (its XAGENT_SUBTASK_API_CONFIG_SCHEMA)
becomes one typed tool.  A call sends VIRA a task group holding just that task, in v1's
payload format, and the model gets back only the sub-task's result, so it can use real
outputs (a new job's id) in the next call.

The catalog is internal and stays out of git.  It is read from config/jeni_tasks.json, or
$JENI_TASKS_FILE (config/README.md), the first time a Jeni tool is needed, so nothing else
depends on it.

The model never sees an endpoint, a URL or a key: a call goes through
recruiter_cli.execute(), the same guarded path as the other VIRA tools (audit log, PII
mask), in the mode the host set with vira_tools.configure().  Every value is checked
against the schema here as well, so a bad call never reaches VIRA whoever makes it.

Changes to v1's catalog (proposed to engineering, docs/agent-frameworks.md §12):
  * add_job_collaborators: role_id has no default.  v1 pre-fills 1 (administrator).
  * make_job_private / make_job_public: is_private is set by the task, not the model.

    python jeni_tools.py                                                  # check the catalog
    python jeni_tools.py --from-js path/to/tasks.js > config/jeni_tasks.json
"""
import argparse
import functools
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, Literal

from pydantic import ConfigDict, Field, ValidationError, create_model

import pii_vault
import recruiter_cli as vira
import vira_results
import vira_tools

HERE = Path(__file__).resolve().parent
STAGE_LOG = logging.getLogger("jeni.stages")


def _stage(event: str, **details) -> None:
    values = " ".join(f"{key}={value}" for key, value in sorted(details.items())
                      if value is not None)
    STAGE_LOG.info("%s%s", event, f" {values}" if values else "")
DEFAULT_CATALOG = HERE / "config" / "jeni_tasks.json"
# The task-group route: real mode sends it to the VIRA engine at VIRA_ACTUAL_LOCATION with the
# user's VIRA_XRTOKEN (recruiter_cli._target); mock mode answers it (mock_jeni.py).
PATH = vira.TASK_GROUP_PATH
SESSION_NS = uuid.UUID("8d1f6a3e-2b47-4c55-9e0a-7a1d6c2f0b91")
_PROCESS_SESSION = str(uuid.uuid4())
PLACEHOLDER = re.compile(r"^\{\{.*\}\}$")

# Field types, as tasks.js declares them for v1's response schema.
NUMBER_ARRAY_FIELDS = {"user_ids", "app_ids", "job_ids"}
STRING_ARRAY_FIELDS = {"skills", "emails"}
NUMBER_FIELDS = {"job_id", "app_id", "role_id", "limit", "skip", "min_exp", "max_exp",
                 "min_salary", "max_salary", "vacancy"}
BOOLEAN_FIELDS = {"is_private"}
# Refinements for v2.
ID_FIELDS = {"job_id", "app_id"}
EMAIL_FIELDS = {"emails", "candidate_email", "new_owner_user_email"}
LONG_TEXT_FIELDS = {"job_description", "job_requirements", "message"}
MAX_LONG_TEXT = 2000
EMAIL_RE = r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$"
# A share recipient or a new owner may also be a colleague's <email:...> token, or "me".
RECIPIENT_RE = r"^(?:[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+|<email:[0-9a-f]{12}>|[Mm][Ee])$"
# Values only the user can give.  Tool results carry them masked, so a value from a
# result would be "<redacted>" anyway; ToolCallGuard refuses one the user didn't write.
USER_ONLY = frozenset({"emails", "candidate_email", "candidate_name", "new_owner_user_email"})

READ_ONLY = frozenset({"get_applications", "get_single_application_details",
                       "get_single_job_details", "search_users",
                       "get_suggested_candidates_for_a_job",
                       "get_self_sourcing_candidates_for_a_job"})
FIXED_VALUES = {"make_job_private": {"is_private": True},
                "make_job_public": {"is_private": False}}
NO_DEFAULT = {("add_job_collaborators", "role_id")}

FIELD_HELP = {
    "job_id": "Numeric job id: from the user or an earlier result.",
    "app_id": "Numeric application id: from the user or an earlier result.",
    "app_ids": "Application ids: from the user or an earlier result (e.g. get_applications).",
    "user_ids": "User ids of people in the company: from the user or search_users.",
    "role_id": "1 = administrator, 5 = team member. If the user didn't say which, ask.",
    "job_title": "The job title the user gave, in title case with acronyms kept (e.g. 'Senior "
                 "Backend Engineer', 'ML Engineer'), even if they typed it in lower case.",
    "skills": "Skills the user named, each spelled the usual way (e.g. 'Python', 'SQL', 'Node.js').",
    "job_description": "Job description text the user gave.",
    "job_requirements": "Job requirements the user gave.",
    "country_name": "Country name, capitalised, e.g. Singapore.",
    "region": "Region the user named.",
    "min_exp": "Minimum years of experience.",
    "max_exp": "Maximum years of experience.",
    "min_salary": "Minimum salary.",
    "max_salary": "Maximum salary.",
    "vacancy": "Number of openings.",
    "new_owner_user_email": ("The new owner's email: exactly as the user wrote it, a colleague's "
                             "<email:...> token from search_users or find_user, or \"me\" for the "
                             "user themself."),
    "emails": ("Recipients' emails: each exactly as the user wrote it, a colleague's <email:...> "
               "token from search_users or find_user, or \"me\" for the user themself."),
    "message": "A short message to send with the shared applications.",
    "candidate_name": "The candidate's name as the user wrote it, capitalised (e.g. 'Maya Lim').",
    "candidate_email": "The candidate's email, exactly as the user wrote it.",
    "reason_for_closure": "Why the job is being closed, if the user said.",
    "search_key": "Text to search for (a name or email).",
    "limit": "Maximum number of results.",
    "skip": "Number of results to skip (paging).",
}

RULES = """
Jeni rules:
- Only transfer job ownership when the user explicitly says "transfer ownership". Assigning a
  job to someone or adding a collaborator is add_job_collaborators.
- Candidate names and emails must come from the user. Results show email addresses as
  <email:...> tokens: a colleague's token (from search_users or find_user) may be a share
  recipient or a new owner, and "me" means the user themself. Never use a candidate's token,
  and never make an address up: ask the user. Don't show a token to the user: name the
  person, or say "you".
- Sharing an application (a CV) is supported. Sharing a job is not.
- If no tool does what the user asks, say it isn't supported instead of approximating it.
- A task result can report status "ok" and still list items in failedArr: read it.
- A task can come back "queued": VIRA accepted it and runs it in the background. Say it was
  submitted, not that it is done, and don't use anything it would have returned in a later step.
- get_applications returns a job's applications with their match scores and stages in one
  call: use it to compare or rank applicants, rather than reading them one by one.
- Ask the user only for what no tool can give you, and don't ask them to confirm values they
  already gave (skills, ids).
- Do each step as soon as you have everything it needs; don't hold it back to ask about a
  later step.
- When a call fails, report the reason. Don't offer to retry it, or to get the same effect
  with a different tool.
"""


# Values typed all in lower case are tidied before they are sent (tidy_case): job titles,
# places and names in title case, each skill in its usual spelling.
TITLE_FIELDS = {"job_title", "country_name", "region", "candidate_name"}
SKILL_FIELDS = {"skills"}
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}
SPELLINGS = {w.lower(): w for w in (
    "AI", "ML", "QA", "UI", "UX", "HR", "IT", "VP", "PM", "CEO", "CTO", "CFO", "COO", "R&D",
    "API", "AWS", "GCP", "SQL", "NoSQL", "CSS", "HTML", "iOS", "macOS", "DevOps", "MLOps", "SRE",
    "ETL", "NLP", "LLM", "CI/CD", "SaaS", "B2B", "B2C", "CRM", "ERP", "SEO", "PHP", "C#", "C++",
    ".NET", "ASP.NET", "Node.js", "React.js", "Vue.js", "Next.js", "JavaScript", "TypeScript",
    "GraphQL", "PostgreSQL", "MySQL", "MongoDB", "TensorFlow", "PyTorch", "GitHub", "GitLab",
    "LinkedIn", "Power BI", "SAP", "UAE", "USA", "UK")}


def _word(word: str, first: bool) -> str:
    if word.lower() in SPELLINGS:
        return SPELLINGS[word.lower()]
    if not first and word in SMALL_WORDS:
        return word
    return "-".join(part[:1].upper() + part[1:] for part in word.split("-"))


def _title(text: str) -> str:
    """'senior backend engineer' -> 'Senior Backend Engineer', 'sql' -> 'SQL'.  Only a value typed
    all in lower case changes: 'iOS Developer' or 'McKinsey' stays as the user wrote it."""
    if not isinstance(text, str) or not text.islower():
        return text
    whole = SPELLINGS.get(text.strip())
    if whole:
        return whole
    return " ".join(_word(w, i == 0) for i, w in enumerate(text.split()))


def tidy_case(args: dict) -> dict:
    """The arguments with lower-case titles, places, names and skills tidied (see _title)."""
    out = dict(args or {})
    for field in TITLE_FIELDS & set(out):
        out[field] = _title(out[field])
    for field in SKILL_FIELDS & set(out):
        if isinstance(out[field], list):
            out[field] = [_title(v) for v in out[field]]
    return out


ROLE_NAMES = {1: "an administrator", 5: "a team member"}
EDIT_WORDS = {"job_title": "the title", "skills": "the skills", "job_description": "the description",
              "job_requirements": "the requirements", "country_name": "the country",
              "region": "the region", "min_exp": "the experience", "max_exp": "the experience",
              "min_salary": "the salary", "max_salary": "the salary", "vacancy": "the openings"}


def _words(values) -> str:
    items = [str(v) for v in (values if isinstance(values, list) else [values]) if v not in (None, "")]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _numbered(word: str, ids) -> str:
    ids = ids if isinstance(ids, list) else [ids]
    return f"{word}{'' if len(ids) == 1 else 's'} {_words(ids)}"


def _short(text, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _new_job(a: dict) -> str:
    parts = [f"Create the job “{_short(a.get('job_title'), 80)}”"]
    if a.get("skills"):
        parts.append(f"needing {_words(a['skills'])}")
    lo, hi = a.get("min_exp"), a.get("max_exp")
    if lo is not None or hi is not None:
        parts.append(f"with {lo}–{hi} years of experience" if lo is not None and hi is not None
                     else f"with at least {lo} years of experience" if lo is not None
                     else f"with up to {hi} years of experience")
    if a.get("country_name"):
        parts.append(f"in {a['country_name']}")
    return " ".join(parts) + "."


def _edit(a: dict) -> str:
    changed = sorted({EDIT_WORDS.get(k, k.replace("_", " ")) for k, v in a.items()
                      if k != "job_id" and v not in (None, "", [])})
    return f"Change {_words(changed) or 'nothing'} of job {a.get('job_id')}."


def _share(a: dict) -> str:
    text = f"Share {_numbered('application', a.get('app_ids') or [])} with {_words(a.get('emails') or [])}"
    note = _short(a.get("message") or "").rstrip(" .!?")
    return text + (f", with the note “{note}”." if note else ".")


# What each task would do, as an approval card says it: plain words, no field names or JSON.
SUMMARIES = {
    "create_job": _new_job,
    "edit_job": _edit,
    "add_job_skills": lambda a: f"Add {_words(a.get('skills'))} to job {a.get('job_id')}.",
    "remove_job_skills": lambda a: f"Remove {_words(a.get('skills'))} from job {a.get('job_id')}.",
    "clone_job": lambda a: f"Make a copy of job {a.get('job_id')}.",
    "transfer_job_ownership": lambda a: (f"Transfer ownership of job {a.get('job_id')} to "
                                         f"{a.get('new_owner_user_email')}."),
    "add_job_collaborators": lambda a: (f"Add {_numbered('user', a.get('user_ids') or [])} to the hiring "
                                        f"team of job {a.get('job_id')} as "
                                        f"{ROLE_NAMES.get(a.get('role_id'), 'an unknown role')}."),
    "publish_job_to_linkedin": lambda a: f"Publish job {a.get('job_id')} to LinkedIn.",
    "make_job_private": lambda a: f"Make job {a.get('job_id')} private.",
    "make_job_public": lambda a: f"Make job {a.get('job_id')} public.",
    "make_job_closed": lambda a: (f"Close job {a.get('job_id')}"
                                  + (f" (“{_short(a['reason_for_closure'])}”)." if a.get("reason_for_closure") else ".")),
    "make_job_open": lambda a: f"Reopen job {a.get('job_id')}.",
    "reject_multiple_application": lambda a: f"Reject {_numbered('application', a.get('app_ids') or [])}.",
    "shortlist_multiple_application": lambda a: f"Shortlist {_numbered('application', a.get('app_ids') or [])}.",
    "share_application": _share,
    "create_application_to_job": lambda a: (f"Add {a.get('candidate_name')} ({a.get('candidate_email')}) "
                                            f"as a candidate for job {a.get('job_id')}."),
}


def summary(name: str, args: dict) -> str:
    """One plain sentence for an approval card: what the call would do.  Any other tool gets its
    name and its values in words, never as JSON."""
    args = {k: ([pii_vault.VAULT.display(x) for x in v] if isinstance(v, list)
                else pii_vault.VAULT.display(v)) if k in pii_vault.RECIPIENT_FIELDS else v
            for k, v in tidy_case(args).items()}
    try:
        if name in SUMMARIES:
            return SUMMARIES[name](args)
    except Exception:          # an odd value must not break the card; fall back to the words
        pass
    values = "; ".join(f"{k.replace('_', ' ')} {_short(_words(v), 60)}"
                       for k, v in (args or {}).items() if v not in (None, "", []))
    title = name.replace("_", " ").capitalize()
    return f"{title}: {values}." if values else f"{title}."


def catalog_from_js(text: str) -> Dict:
    """v1's tasks.js -> {"tasks": [...]}: the object literal, without comments or trailing commas."""
    start = text.index("{", text.index("XAGENT_SUBTASK_API_CONFIG_SCHEMA ="))
    end = text.index("\n}\n", start) + 2
    body = "\n".join(line for line in text[start:end].splitlines()
                     if not line.strip().startswith("//"))
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", body))


class CatalogMissing(FileNotFoundError):
    """The task catalog isn't where JENI_TASKS_FILE (or the default) says."""


def catalog_path() -> Path:
    path = Path(os.environ.get("JENI_TASKS_FILE") or DEFAULT_CATALOG)
    return path if path.is_absolute() else HERE / path


def load_catalog(path: Path | None = None) -> list[dict]:
    path = path or catalog_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))["tasks"]
    except FileNotFoundError:
        raise CatalogMissing(
            f"Jeni's task catalog isn't at {path}. It is shared internally, not in git: put "
            f"the file there or set JENI_TASKS_FILE (config/README.md).") from None


def tool_name(task: dict) -> str:
    return task["task_name"].removeprefix("task_")


def _annotation(field: str):
    """(type, Field constraints) for one catalog field."""
    if field in NUMBER_ARRAY_FIELDS:
        return list[vira_tools.Id], {"min_length": 1, "max_length": vira.MAX_IDS}
    if field in STRING_ARRAY_FIELDS:
        item = (Annotated[str, Field(pattern=_email_pattern(field), max_length=vira.MAX_TEXT)]
                if field in EMAIL_FIELDS else vira_tools.Text)
        return list[item], {"min_length": 1, "max_length": vira.MAX_ITEMS}
    if field == "role_id":
        return Literal[1, 5], {}
    if field in ID_FIELDS:
        return vira_tools.Id, {}
    if field in {"min_exp", "max_exp"}:
        return int, {"ge": 0, "le": 60}
    if field in {"min_salary", "max_salary"}:
        return int, {"ge": 0}
    if field == "vacancy":
        return int, {"ge": 1, "le": 1000}
    if field == "limit":
        return int, {"ge": 1, "le": vira.MAX_IDS}
    if field == "skip":
        return int, {"ge": 0}
    if field in BOOLEAN_FIELDS:
        return bool, {}
    if field in EMAIL_FIELDS:
        return str, {"pattern": _email_pattern(field), "max_length": vira.MAX_TEXT}
    return str, {"min_length": 1,
                 "max_length": MAX_LONG_TEXT if field in LONG_TEXT_FIELDS else vira.MAX_TEXT}


def _email_pattern(field: str) -> str:
    return RECIPIENT_RE if field in pii_vault.RECIPIENT_FIELDS else EMAIL_RE


def _field_description(f: dict) -> str:
    """FIELD_HELP, else v1's own notes (some of which predate the typed schema, e.g. skills
    as a "comma separated string")."""
    if f["field_name"] in FIELD_HELP:
        return FIELD_HELP[f["field_name"]]
    parts = [f.get("description"), f.get("field_value_description")]
    return " ".join(p.strip().rstrip(".") + "." for p in parts if p) or f["field_name"]


def args_model(task: dict):
    """The pydantic model for one task's arguments: its tool schema."""
    name = tool_name(task)
    fields = {}
    for f in task["sub_tasks"][0]["fields"]:
        field = f["field_name"]
        if field in FIXED_VALUES.get(name, {}):
            continue
        typ, constraints = _annotation(field)
        default = f.get("field_value")
        has_default = (default is not None and not PLACEHOLDER.match(str(default))
                       and (name, field) not in NO_DEFAULT)
        kwargs = {"description": _field_description(f), **constraints}
        if f["mandatory"] and not has_default:
            fields[field] = (typ, Field(**kwargs))
        else:
            fields[field] = (typ | None, Field(default if has_default else None, **kwargs))
    return create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)


def description(task: dict) -> str:
    text = task["sub_tasks"][0]["description"].strip()
    if tool_name(task) in READ_ONLY:
        return f"{text} (read-only)"
    return f"{text} (Changes data: in real mode a person approves each call.)"


def session_uuid(thread_id: str | None = None) -> str:
    """The agent_session_uuid VIRA requires on every task group: one per process and thread,
    so a conversation's groups share a session. VIRA scopes group-name uniqueness by session,
    so payload() makes each group's name unique instead."""
    seed = f"{_PROCESS_SESSION}:{thread_id or 'no_thread'}"
    return str(uuid.uuid5(SESSION_NS, seed))


def payload(task: dict, args: dict, session: str | None = None) -> dict:
    """One task group holding just this task, in v1's format (handleGenerateViraPayload)."""
    name, sub = tool_name(task), task["sub_tasks"][0]
    values = {**args, **FIXED_VALUES.get(name, {})}
    fields = []
    for f in sub["fields"]:
        entry = {"field_name": f["field_name"], "mandatory": f["mandatory"]}
        if values.get(f["field_name"]) is not None:
            entry["field_value"] = values[f["field_name"]]
        fields.append(entry)
    return {"agent_session_uuid": session or session_uuid(), "task_group_name": f"Jeni v2-{name}-{uuid.uuid1().hex}",
            "tasks": [{"task_name": task["task_name"], "description": task["description"],
                       "level": task["level"],
                       "sub_tasks": [{"sub_task_name": sub["sub_task_name"],
                                      "description": sub["description"], "fields": fields}]}]}


QUEUED = {
    "request_status": "ok",
    "group_status": "queued",
    "subtasks": [],
    "message": "VIRA accepted the task and runs it in the background; its outcome isn't known yet.",
}


def _queued_uuid(result: Dict) -> str | None:
    """The group's uuid when the engine only acknowledged it ("We are processing your tasks")."""
    reply = result.get("result") if result.get("status") == "ok" else None
    if isinstance(reply, dict) and reply.get("agentTaskGroupUuid") and "tasks" not in reply:
        return reply["agentTaskGroupUuid"]
    return None


def _read_back(command: str, group_uuid: str) -> Dict | None:
    """The group's outcome from vira_results, masked like any VIRA reply and audited, or None."""
    try:
        group = vira_results.wait_for(group_uuid)
    except Exception as exc:           # e.g. the database is unreachable: still just queued
        vira._audit("read-result", {}, {"agentTaskGroupUuid": group_uuid, "for": command},
                    {"status": "exception"}, error=type(exc).__name__)
        return None
    if group is None:
        return None
    vira._audit("read-result", {}, {"agentTaskGroupUuid": group_uuid, "for": command}, group)
    return vira.mask_result(command, group)


def _request_failed(message: str) -> Dict:
    """A failure before a trustworthy VIRA task result was obtained."""
    return {"request_status": "failed", "message": message}


def _result_message(result: Dict, default: str) -> str:
    """Return a safe, concise transport/error message from a VIRA-shaped reply."""
    message = result.get("message")
    if not message and isinstance(result.get("result"), dict):
        message = result["result"].get("message")
    return str(message or default)


def project(result: Dict) -> Dict:
    """Project VIRA's real execution state for the model.

    ``request_status`` says whether Jeni successfully sent/read the request.  The
    engine's ``agentTaskGroupStatus`` and every subtask status remain separate, so
    an HTTP/transport success is never presented as a completed business action.
    """
    if result.get("status") != "ok":
        return _request_failed(_result_message(result, "VIRA request failed"))
    if _queued_uuid(result):
        return QUEUED      # the VIRA engine queued the group and runs it later: nothing is known yet
    try:
        group = result["result"]
        group_status = group["agentTaskGroupStatus"]
        tasks = group["tasks"]
    except (KeyError, TypeError):
        return _request_failed("unexpected VIRA reply (no task-group result)")

    subtasks = []
    for task in tasks:
        for sub in task.get("subTasks", []):
            sub_status = sub.get("agentSubTaskStatus")
            item = {"status": sub_status}
            if str(sub_status).lower() == "failed" and sub.get("failedReason"):
                item["failed_reason"] = sub["failedReason"]
            if sub.get("agentSubTaskResponse") is not None:
                item["result"] = sub["agentSubTaskResponse"]
            subtasks.append(item)
    if not subtasks:
        return _request_failed("unexpected VIRA reply (no sub-task result)")
    if (str(group_status).lower() in vira_results.UNFINISHED or
            any(str(sub["status"]).lower() in vira_results.UNFINISHED for sub in subtasks)):
        return QUEUED      # read back, but still not run when the wait ran out
    return {"request_status": "ok", "group_status": group_status, "subtasks": subtasks}


@functools.lru_cache(maxsize=None)
def _load(path: str) -> tuple[dict, dict]:
    tasks = {tool_name(t): t for t in load_catalog(Path(path))}
    return tasks, {name: args_model(t) for name, t in tasks.items()}


def catalog() -> dict:
    """{tool name: v1 task}, read on first use."""
    return _load(str(catalog_path()))[0]


def models() -> dict:
    """{tool name: pydantic model of its arguments}."""
    return _load(str(catalog_path()))[1]


def names() -> frozenset:
    return frozenset(catalog())


def _problems(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}"
                     for e in exc.errors()[:5])


def run(name: str, args: dict, session: str | None = None) -> Dict:
    """Validate, send the one-task group through the guarded path, and project the reply."""
    if name not in catalog():
        return _request_failed(f"unknown task {name}")
    _stage("PYDANTIC_CHECK", tool=name)
    try:
        clean = tidy_case(models()[name].model_validate(args).model_dump(exclude_none=True))
    except ValidationError as exc:
        _stage("PYDANTIC_RESULT", tool=name, result="failed")
        return _request_failed(f"invalid arguments: {_problems(exc)}")
    _stage("PYDANTIC_RESULT", tool=name, result="passed")
    try:              # "me" as the signed-in user's token; recruiter_cli sends the address
        clean = {k: pii_vault.resolve_me(k, v) for k, v in clean.items()}
    except LookupError:
        return _request_failed("The signed-in user's email isn't known here: ask the user to type it.")
    try:
        command = name.replace("_", "-")
        result = vira.execute(command, PATH, {}, payload(catalog()[name], clean, session),
                              mode=vira_tools.current_mode(), confirmed=False)
    except Exception as exc:        # e.g. VIRA unreachable; the message may name hosts
        return _request_failed(f"VIRA call failed ({type(exc).__name__})")
    group_uuid = _queued_uuid(result)
    if group_uuid:                  # the engine runs it in the background: wait for the outcome
        result = _read_back(command, group_uuid) or result
    return project(result)


def langchain_tools() -> list:
    """The tasks as LangChain StructuredTools (LangGraph, deepagents)."""
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import StructuredTool

    def tool(name: str):
        def call(config: RunnableConfig, **kwargs):     # config is injected, not in the schema
            thread = (config.get("configurable") or {}).get("thread_id")
            return run(name, kwargs, session_uuid(thread))

        return StructuredTool.from_function(
            func=call, name=name,
            description=description(catalog()[name]), args_schema=models()[name])

    return [tool(name) for name in catalog()]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Jeni's task catalog: check it, or build it from tasks.js.")
    p.add_argument("--from-js", metavar="TASKS_JS",
                   help="print v1's tasks.js as config/jeni_tasks.json content")
    args = p.parse_args(argv)
    if args.from_js:
        tasks = catalog_from_js(Path(args.from_js).read_text(encoding="utf-8"))["tasks"]
        json.dump({"source": "Jeni v1 XAGENT_SUBTASK_API_CONFIG_SCHEMA (tasks.js). Internal: "
                             "keep out of git (config/README.md).",
                   "tasks": tasks}, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return
    from mock_jeni import HANDLERS
    print(f"{catalog_path()}: {len(catalog())} tasks")
    for name, task in catalog().items():
        kind = "read" if name in READ_ONLY else "write"
        mock = "" if task["task_name"] in HANDLERS else "   (mock VIRA can't answer it)"
        print(f"{kind:5}  {name}({', '.join(models()[name].model_fields)}){mock}")


if __name__ == "__main__":
    main()
