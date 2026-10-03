# db_lookup.py
from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional

if TYPE_CHECKING:          # only for the annotations: the caller's pool brings the driver
    import asyncpg


async def handle_search_users(
    conn: asyncpg.Connection,
    company_id: int,
    search_keys: Optional[List[str]] = None,
    xuser_ids: Optional[List[int]] = None,
    xemails: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Search eligible active role-4 users in a company.

    ``search_keys`` is the LLM-facing partial name/email search. ``xuser_ids``
    and ``xemails`` are exact validation modes used by entity validation; they
    reuse this same eligibility population rather than duplicating user SQL.
    returns list: [{"userId":int,"firstname":str,"lastname":str,"email":str}]
    """
    search_keys = search_keys or []
    xuser_ids = xuser_ids or []
    xemails = [str(email).lower() for email in (xemails or [])]
    base_sql = """
        SELECT ui.user_id, ui.firstname, ui.lastname, ui.email
        FROM hris.userinfo ui
        WHERE ui.company_id = $1
        AND ui.role_id = 4
        AND ui.active IS TRUE
    """
    params = [company_id]
    or_parts = []
    for kw in search_keys:
        params.append(f"%{kw}%")
        pos = len(params)
        frag = f"""
        (
            ui.email ILIKE ${pos}
            OR trim(concat(ui.firstname, ' ', ui.lastname)) ILIKE ${pos}
            OR trim(concat(ui.raw_firstname, ' ', ui.raw_lastname)) ILIKE ${pos}
        )
        """
        or_parts.append(frag.strip())
    if or_parts:
        base_sql += f" AND ({' OR '.join(or_parts)})"
    if xuser_ids:
        params.append([int(user_id) for user_id in xuser_ids])
        base_sql += f" AND ui.user_id = ANY(${len(params)}::bigint[])"
    if xemails:
        params.append(xemails)
        base_sql += f" AND lower(ui.email) = ANY(${len(params)}::text[])"
    params.append(limit)
    base_sql += f" ORDER BY ui.email ASC LIMIT ${len(params)};"

    rows = await conn.fetch(base_sql, *params)
    return [
        {
            "userId": r["user_id"],
            "firstname": r["firstname"],
            "lastname": r["lastname"],
            "email": r["email"],
        }
        for r in rows
    ]


async def handle_find_candidate_by_email(
    conn: asyncpg.Connection,
    company_id: int,
    email: str,
) -> List[Dict]:
    """Candidate profiles with an application in the authenticated company.

    A profile can have several applications, so the result is distinct by
    profile.  Candidate-name comparison is intentionally not included until
    the canonical profile name columns are confirmed.
    """
    rows = await conn.fetch(
        "SELECT DISTINCT p.profile_id, p.email FROM hris.profile p "
        "JOIN hris.application a ON a.profile_id = p.profile_id "
        "JOIN hris.job j ON j.job_id = a.job_id "
        "WHERE j.recuiter_company_id = $1 AND lower(p.email) = lower($2) "
        "LIMIT 2;",
        company_id, str(email),
    )
    return [{"profileId": r["profile_id"], "email": r["email"]} for r in rows]


async def handle_search_jobs(
    conn: asyncpg.Connection,
    company_id: int,
    search_keys: Optional[List[str]] = None,
    xjob_ids: Optional[List[int]] = None,
    limit: int = 10,
) -> List[Dict]:
    """
    Equivalent Node handleSearchJobs
    Search jobs: filter by company_id, from_resume=false.
    Can filter by partial job-name keywords OR list of job_ids.
    returns list: [{"jobId":int, "jobName":str, "openDate":datetime|None,
                   "isPrivate":bool|None, "closeDate":datetime|None}], newest first
    """
    search_keys = search_keys or []
    xjob_ids = xjob_ids or []
    base_sql = """
        SELECT j.job_id, jn.name_name AS job_name, j.open_date, j.is_private, j.close_date
        FROM hris.job j
        INNER JOIN hris.jobname jn ON jn.name_id = j.name_id
        WHERE j.recuiter_company_id = $1
        AND j.from_resume IS FALSE
    """
    params = [company_id]
    cond_parts = []

    # keyword search on job name
    if search_keys:
        or_frags = []
        for kw in search_keys:
            params.append(f"%{kw}%")
            pos = len(params)
            or_frags.append(f"jn.name_name ILIKE ${pos}")
        cond_parts.append(f"({' OR '.join(or_frags)})")

    # filter by job id list
    if xjob_ids:
        params.append(xjob_ids)
        pos = len(params)
        cond_parts.append(f"j.job_id = ANY(${pos}::bigint[])")

    if cond_parts:
        base_sql += f" AND ({' OR '.join(cond_parts)})"
    params.append(limit)
    base_sql += f" ORDER BY j.job_id DESC LIMIT ${len(params)};"

    rows = await conn.fetch(base_sql, *params)
    return [
        {
            "jobId": r["job_id"],
            "jobName": r["job_name"],
            "openDate": r["open_date"],
            "isPrivate": r["is_private"],
            "closeDate": r["close_date"],
        }
        for r in rows
    ]


async def handle_additional_context_by_search(
    conn: asyncpg.Connection,
    company_id: int,
    jobnames: Optional[List[str]] = None,
    usernames: Optional[List[str]] = None,
    xjob_ids: Optional[List[int]] = None,
):
    """
    Equivalent Node handleAdditionalContextBySearch
    Run both user + job search, build text additional-context string for LLM prompt.
    return {"additionalContext": str}
    """
    jobnames = jobnames or []
    usernames = usernames or []
    xjob_ids = xjob_ids or []

    jobs = await handle_search_jobs(conn, company_id, search_keys=jobnames, xjob_ids=xjob_ids)
    users = await handle_search_users(conn, company_id, search_keys=usernames)

    contexts = []
    if jobs:
        contexts.append(f"- Fetched Jobs: {jobs!r}")
    if users:
        contexts.append(f"- Fetched Users: {users!r}")

    additional_context = "\n".join(contexts)
    return {"additionalContext": additional_context}
