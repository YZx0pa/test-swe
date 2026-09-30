"""Registered read-only queries, defined once for both production and tests.

- ``build_db_queries(pool)``  -> QueryTools backed by the real asyncpg handlers
  in db_lookup.py.  asyncpg is imported lazily so this module loads offline.
- ``fake_db_queries(fixtures)`` -> the SAME QueryTool contracts backed by
  in-memory fixtures, for the scenario harness and offline tests.

Contract mapping (0 / 1 / many rows) is identical in both, so a scenario that
passes against fakes exercises the real query's control flow.

Tenant safety: ``company_id`` is read from ``context["auth_profile"]``. The
profile is injected by the authenticated backend and passed directly from the
Coordinator to trusted query adapters; it is never supplied by the model. A
query refuses (status=error) if tenant scope is missing, rather than silently
querying the wrong tenant. The LLM only supplies SEARCH terms (title /
search_key / job_id).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping


@dataclass
class QueryTool:
    """The container each registered query is built from.

    Constructed positionally as QueryTool(description, inputs, outputs, handler); the
    handler is ``async def handler(inputs: dict, context: dict) -> dict`` returning the
    resolved / ambiguous / not_found / validated contract.  db_tools.py reads
    .description, .inputs and .handler off these.  Defined here (not imported from the
    other project's agent_loop_v2) so this module stays self-contained and LangChain-free.
    """
    description: str
    inputs: Mapping[str, str]        # {field_name: type_str}, e.g. {"title": "str"}
    outputs: Mapping[str, str]       # {field_name: type_str}
    handler: Callable[..., Any]


# --- shared 0/1/many -> contract shaping ------------------------------------
# Candidates shown per search.  A search asks for one more row, to know whether there are more.
SEARCH_LIMIT = 10


def _resolve_one(rows: List[Mapping[str, Any]], id_key: str, out_field: str,
                 label: Callable[[Mapping[str, Any]], str], what: str) -> Dict[str, Any]:
    """0 rows -> not_found; 1 -> resolved; many -> ambiguous + candidates.

    Up to SEARCH_LIMIT + 1 rows come in; the extra one only says the list is cut short.
    """
    if not rows:
        return {"status": "not_found", "message": f"no {what} matched"}
    if len(rows) == 1:
        return {"status": "resolved", "resolved_fields": {out_field: rows[0][id_key]}}
    shown = rows[:SEARCH_LIMIT]
    message = (f"more than {SEARCH_LIMIT} {what}s match; showing {SEARCH_LIMIT}. Ask the user "
               f"for the id or a more specific search" if len(rows) > SEARCH_LIMIT
               else f"{len(rows)} {what}s match")
    return {
        "status": "ambiguous",
        "message": message,
        "candidates": [{out_field: r[id_key], "label": label(r)} for r in shown],
        "requires_semantic_review": True,
    }


def _job_label(row: Mapping[str, Any]) -> str:
    """'Data Scientist (opened 2026-08-21)': jobs often share a title, their open dates differ."""
    opened = row.get("openDate")
    return f"{row['jobName']} (opened {opened:%Y-%m-%d})" if opened else row["jobName"]


def _user_label(row: Mapping[str, Any]) -> str:
    return row.get("email", "")


def _like_literal(text: str) -> str:
    """A search term matched literally by ILIKE: its % and _ are not wildcards."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _company_id(context: Mapping[str, Any]) -> int:
    cid = ((context or {}).get("auth_profile") or {}).get("company_id")
    if cid is None:
        raise _TenantMissing("company_id is not set on the authenticated profile")
    return int(cid)


class _TenantMissing(Exception):
    pass


def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapped(inputs, context):
        try:
            return await fn(inputs, context)
        except _TenantMissing as exc:
            # Surfaced as a normal query error -> becomes evidence, not a crash.
            return {"status": "error", "message": str(exc)}
    return wrapped


def _validated(requested: List[Any], found: List[Any], what: str) -> Dict[str, Any]:
    """Stable validation contract for explicit user identifiers.

    A partial list is still ``resolved`` because the DB query itself succeeded;
    ``invalid_values`` tells the coordinator to reject the whole user value
    rather than silently applying an action to only some of the requested IDs.
    """
    found_set = set(found)
    invalid = [value for value in requested if value not in found_set]
    return {
        "status": "resolved" if not invalid else "not_found",
        "valid_values": [value for value in requested if value in found_set],
        "invalid_values": invalid,
        "message": (f"{what} validated" if not invalid else f"unknown or unavailable {what}"),
    }


def _select_many(*, target_field: str, candidate_field: str, values: List[Any], message: str,
                 empty_policy: str = "skip_step") -> Dict[str, Any]:
    """Common lookup outcome for a user-governed many-value selection.

    The coordinator understands this shape for *any* lookup.  It is not an
    application-specific instruction: adapters merely state the available
    values and the selection/empty policies appropriate to their domain.
    """
    return {
        "status": "resolved",
        "resolved_fields": {},
        "selection": {
            "target_field": target_field,
            "candidate_field": candidate_field,
            "selection_policy": "choose_many",
            "empty_policy": empty_policy,
            "candidates": [{candidate_field: value, "label": f"Application {value}"} for value in values],
            "message": message,
        },
        # Retained temporarily for consumers of the old QueryTool response.
        "applications": values,
        "message": message,
    }


# --- REAL: backed by db_lookup.py + an asyncpg pool -------------------------
def build_db_queries(pool) -> Dict[str, QueryTool]:
    """QueryTools for production. ``pool`` is an asyncpg pool/acquire source."""
    import db_lookup  # lazy: imports asyncpg

    async def _find_job_by_title(inputs, context):
        cid = _company_id(context)
        async with pool.acquire() as conn:
            rows = await db_lookup.handle_search_jobs(
                conn, cid, search_keys=[_like_literal(inputs["title"])], limit=SEARCH_LIMIT + 1)
        return _resolve_one(rows, "jobId", "job_id", _job_label, "job")

    async def _find_user(inputs, context):
        cid = _company_id(context)
        async with pool.acquire() as conn:
            rows = await db_lookup.handle_search_users(
                conn, cid, search_keys=[_like_literal(inputs["search_key"])],
                limit=SEARCH_LIMIT + 1)
        return _resolve_one(rows, "userId", "user_id", _user_label, "user")

    async def _list_job_applications(inputs, context):
        cid = _company_id(context)
        async with pool.acquire() as conn:
            # parameterized by job_id and scoped to the tenant, like the validate_* queries:
            # a job of another company lists nothing
            rows = await conn.fetch(
                "SELECT a.app_id FROM hris.application a JOIN hris.job j ON j.job_id = a.job_id "
                "WHERE j.recuiter_company_id = $1 AND a.job_id = $2 LIMIT 200;",
                cid, int(inputs["job_id"]),
            )
        ids = [r["app_id"] for r in rows]
        return _select_many(target_field="app_ids", candidate_field="app_id", values=ids,
                            message=f"{len(ids)} applications for job {inputs['job_id']}")

    async def _validate_job_ids(inputs, context):
        cid, requested = _company_id(context), [int(v) for v in inputs["job_ids"]]
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT job_id FROM hris.job WHERE recuiter_company_id = $1 "
                "AND from_resume IS FALSE AND job_id = ANY($2::bigint[]);", cid, requested)
        return _validated(requested, [r["job_id"] for r in rows], "job IDs")

    async def _validate_app_ids(inputs, context):
        cid, requested = _company_id(context), [int(v) for v in inputs["app_ids"]]
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT a.app_id FROM hris.application a JOIN hris.job j ON j.job_id = a.job_id "
                "WHERE j.recuiter_company_id = $1 AND a.app_id = ANY($2::bigint[]);", cid, requested)
        return _validated(requested, [r["app_id"] for r in rows], "application IDs")

    async def _validate_emails(inputs, context):
        cid, requested = _company_id(context), [str(v).lower() for v in inputs["emails"]]
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT lower(email) AS email FROM hris.userinfo WHERE company_id = $1 "
                "AND active IS TRUE AND lower(email) = ANY($2::text[]);", cid, requested)
        return _validated(requested, [r["email"] for r in rows], "emails")

    return _registry(_find_job_by_title, _find_user, _list_job_applications,
                     _validate_job_ids, _validate_app_ids, _validate_emails)


# --- FAKE: backed by in-memory fixtures -------------------------------------
def fake_db_queries(fixtures: Mapping[str, Any]) -> Dict[str, QueryTool]:
    """QueryTools for tests. ``fixtures`` = {
        "company_id": 1,
        "jobs":  [{"jobId":501,"jobName":"Data Scientist","company_id":1,
                   "openDate": datetime(...) (optional)}, ...],
        "users": [{"userId":9,"firstname":"A","lastname":"B","email":"a@x.com","company_id":1}, ...],
        "applications": [{"app_id":11,"job_id":501}, ...],
    }  All rows are filtered by the fixture company_id, mirroring the real WHERE."""
    jobs = fixtures.get("jobs", [])
    users = fixtures.get("users", [])
    apps = fixtures.get("applications", [])

    def _tenant(rows, cid):
        return [r for r in rows if r.get("company_id", cid) == cid]

    async def _find_job_by_title(inputs, context):
        cid = _company_id(context)
        kw = str(inputs["title"]).lower()
        rows = sorted((r for r in _tenant(jobs, cid) if kw in r["jobName"].lower()),
                      key=lambda r: r["jobId"], reverse=True)[:SEARCH_LIMIT + 1]
        return _resolve_one(rows, "jobId", "job_id", _job_label, "job")

    async def _find_user(inputs, context):
        cid = _company_id(context)
        kw = str(inputs["search_key"]).lower()
        rows = [r for r in _tenant(users, cid)
                if kw in r["email"].lower()
                or kw in f"{r['firstname']} {r['lastname']}".lower()][:SEARCH_LIMIT + 1]
        return _resolve_one(rows, "userId", "user_id", _user_label, "user")

    async def _list_job_applications(inputs, context):
        job_ids = {r["jobId"] for r in _tenant(jobs, _company_id(context))}
        ids = [a["app_id"] for a in apps
               if a["job_id"] == int(inputs["job_id"]) and a["job_id"] in job_ids]
        return _select_many(target_field="app_ids", candidate_field="app_id", values=ids,
                            message=f"{len(ids)} applications for job {inputs['job_id']}")

    async def _validate_job_ids(inputs, context):
        cid, requested = _company_id(context), [int(v) for v in inputs["job_ids"]]
        found = [r["jobId"] for r in _tenant(jobs, cid) if r["jobId"] in set(requested)]
        return _validated(requested, found, "job IDs")

    async def _validate_app_ids(inputs, context):
        cid, requested = _company_id(context), [int(v) for v in inputs["app_ids"]]
        job_ids = {r["jobId"] for r in _tenant(jobs, cid)}
        found = [a["app_id"] for a in apps if a["job_id"] in job_ids and a["app_id"] in set(requested)]
        return _validated(requested, found, "application IDs")

    async def _validate_emails(inputs, context):
        cid, requested = _company_id(context), [str(v).lower() for v in inputs["emails"]]
        found = [str(r["email"]).lower() for r in _tenant(users, cid)
                 if str(r["email"]).lower() in set(requested)]
        return _validated(requested, found, "emails")

    return _registry(_find_job_by_title, _find_user, _list_job_applications,
                     _validate_job_ids, _validate_app_ids, _validate_emails)


def _registry(find_job, find_user, list_apps, validate_jobs, validate_apps, validate_emails) -> Dict[str, QueryTool]:
    return {
        "find_job_by_title": QueryTool(
            "Find jobs by (partial) title.",
            {"title": "str"}, {"job_id": "int"}, _guard(find_job)),
        "find_user": QueryTool(
            "Find an active recruiter user by name or email.",
            {"search_key": "str"}, {"user_id": "int"}, _guard(find_user)),
        "list_job_applications": QueryTool(
            "List all applicants for a given job.",
            {"job_id": "int"}, {"app_ids": "list[int]"}, _guard(list_apps)),
        "validate_job_id": QueryTool(
            "Validate one user-supplied job ID in the authenticated company.",
            {"job_id": "int"}, {}, _guard(lambda inputs, context: validate_jobs({"job_ids": [inputs["job_id"]]}, context))),
        "validate_job_ids": QueryTool(
            "Validate user-supplied job IDs in the authenticated company.",
            {"job_ids": "list[int]"}, {}, _guard(validate_jobs)),
        "validate_app_ids": QueryTool(
            "Validate user-supplied application IDs in the authenticated company.",
            {"app_ids": "list[int]"}, {}, _guard(validate_apps)),
        "validate_email": QueryTool(
            "Validate one user-supplied active email in the authenticated company.",
            {"email": "str"}, {}, _guard(lambda inputs, context: validate_emails({"emails": [inputs["email"]]}, context))),
        "validate_emails": QueryTool(
            "Validate user-supplied active emails in the authenticated company.",
            {"emails": "list[str]"}, {}, _guard(validate_emails)),
    }
