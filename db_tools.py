#!/usr/bin/env python3
"""Adapter: db_queries.py's read-only QueryTools -> LangChain StructuredTools.

The db layer (db_queries.py, db_lookup.py) was written for agent_loop_v2's
Coordinator: async handlers of the form ``async def h(inputs, context)`` that read
the tenant from ``context["auth_profile"]["company_id"]`` and return the rich
resolved / ambiguous / not_found / validated contracts.

This module lets the SAME query tools run inside run_langgraph.py's LangChain
create_agent loop, next to the jeni tools, so one agent can resolve an id (db) and
then act on it (jeni).  Four things it takes care of:

  * async -> the tools are registered as coroutines (StructuredTool coroutine=...),
    so the graph must be driven with ``ainvoke`` (agent_kit.arun_task).
  * context injection -> ``context`` (the AuthProfile's company_id) is CLOSED OVER at
    build time and never exposed as a tool argument.  The model supplies only search
    terms / ids (title, search_key, job_id, ...), exactly as db_queries.py requires.
    A model-supplied company_id would be a cross-tenant hole; this prevents it.
  * contract translation -> the LLM (not a Coordinator) reads the result here, so a
    "resolved" result is flattened to its resolved_fields and an "ambiguous" result
    keeps its candidates for the model to ask about (see RULES below).
  * masking -> results go through recruiter_cli._mask_pii before the model sees them, so a
    user search's email labels arrive redacted, as people do in the Jeni tools' results.

Only the read-only query tools are exposed; nothing here changes data.  QueryTool is
defined in db_queries.py, so there is no dependency on the other project's agent_loop_v2.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from pydantic import Field, create_model

import pii_vault
import recruiter_cli

# Rules appended to the prompt when these tools are present, so the model resolves and
# checks ids with reads BEFORE calling an action tool.  Lives here, next to the tools it
# describes, mirroring jeni_tools.RULES.
RULES = """
Database lookup rules:
- If the user names a job or person by title/name instead of a numeric id, call
  find_job_by_title or find_user FIRST to get the id, then use it. Do not ask the user
  for an id you can look up.
- Before any action that changes data, validate every user-supplied id or email with the
  matching validate_* tool (validate_job_id(s), validate_app_ids, validate_email(s)).
  Proceed only when the status is "resolved". Ids a tool returned are valid already, and so
  are a colleague's <email:...> token and "me".
- "The applicants" of a job means all of them: get their ids with list_job_applications.
  Ask which ones only if the user said "some" without saying which.
- On status "ambiguous", list every candidate by its label, which has the name and the id
  (e.g. "Senior Backend Engineer (job 7001, opened 2026-08-21)"), so the user can compare
  them, and ask which one; they may answer with the name, a detail or the id. Never choose
  for them, and ask nothing else then: the rest of the request stands as they gave it.
- On status "not_found" or "error", tell the user plainly and do not attempt the action.
- You supply only search terms and ids; the company is set by the system, never as a
  tool argument.
"""

# All db_queries tools are reads; none change data.
READ_ONLY = frozenset({
    "find_job_by_title", "find_user", "list_job_applications",
    "validate_job_id", "validate_job_ids", "validate_app_ids",
    "validate_email", "validate_emails",
})

# Identifiers that must come from the USER, so ToolCallGuard refuses a value the user
# didn't write (consistent with jeni_tools.USER_ONLY: emails are user-supplied).
USER_ONLY = frozenset({"email", "emails"})
# Lookups in the company's user directory: emails in their results are colleagues'.
DIRECTORY_TOOLS = frozenset({"find_user", "validate_email", "validate_emails"})

# db_queries declares inputs as these type strings; map them to Python types for the schema.
_PY_TYPES: dict[str, Any] = {
    "str": str, "int": int, "list[int]": list[int], "list[str]": list[str],
}


def _qt_parts(qt: Any) -> tuple[str, Mapping[str, str], Callable]:
    """(description, inputs, handler) from a db_queries.QueryTool (a dataclass)."""
    return qt.description, qt.inputs, qt.handler


def _args_schema(name: str, inputs: Mapping[str, str]):
    """A pydantic model for one tool's arguments, from db_queries' {field: typestr}."""
    fields = {}
    for field, typestr in inputs.items():
        typ = _PY_TYPES.get(typestr)
        if typ is None:
            raise ValueError(f"{name}: unknown input type {typestr!r} (add it to _PY_TYPES)")
        fields[field] = (typ, Field(..., description=field))
    return create_model(f"{name}_args", **fields)


def _translate(result: Any) -> Any:
    """Shape a QueryTool contract for an LLM reader (no Coordinator here).

    resolved  -> the resolved_fields (e.g. {"job_id": 501}), so the next call is obvious.
    ambiguous -> keep candidates; RULES tells the model to ask the user which one.
    other     -> pass through (not_found / error / validated / selection).
    """
    if not isinstance(result, Mapping):
        return result
    status = result.get("status")
    if status == "resolved" and result.get("resolved_fields"):
        return {"status": "resolved", **result["resolved_fields"]}
    return dict(result)


def langchain_tools(query_tools: Mapping[str, Any], context: Mapping[str, Any]) -> list:
    """The db QueryTools as LangChain StructuredTools, with `context` bound at build time.

    query_tools: db_queries.build_db_queries(pool) or fake_db_queries(fixtures).
    context:     {"auth_profile": {"company_id": ...}} -- injected, never from the model.
    """
    from langchain_core.tools import StructuredTool

    tools = []
    for name, qt in query_tools.items():
        if name not in READ_ONLY:
            continue
        description, inputs, handler = _qt_parts(qt)

        # Bind name/handler per-iteration; context is shared and closed over.  The result is
        # masked like every VIRA result: a user search's labels are emails, which come back as
        # colleagues' tokens; a token or "me" in the input is looked up by its address.
        async def _call(_handler=handler, _name=name, **kwargs):
            try:
                kwargs = {k: pii_vault.VAULT.resolve(pii_vault.resolve_me(k, v))
                          for k, v in kwargs.items()}
            except LookupError:            # an unknown token, or "me" without a signed-in user
                return json.dumps({"status": "error", "message": "unknown email: ask the user "
                                                                 "for the address"})
            result = await _handler(kwargs, context)   # context NOT a model arg
            source = pii_vault.COLLEAGUE if _name in DIRECTORY_TOOLS else pii_vault.RECORD
            return json.dumps(recruiter_cli._mask_pii(_translate(result), source),
                              ensure_ascii=False, default=str)

        tools.append(StructuredTool.from_function(
            coroutine=_call, name=name, description=description,
            args_schema=_args_schema(name, inputs)))
    return tools


def names(query_tools: Mapping[str, Any]) -> frozenset:
    return frozenset(n for n in query_tools if n in READ_ONLY)
