# db_lookup.py
from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional

if TYPE_CHECKING:          # only for the annotations: the caller's pool brings the driver
    import asyncpg


async def handle_search_users(
    conn: asyncpg.Connection,
    company_id: int,
    search_keys: List[str],
    limit: int = 10,
) -> List[Dict]:
    """
    Equivalent to Node handleSearchUsers
    Search active role-4 users in given company by partial name / email.
    returns list: [{"userId":int,"firstname":str,"lastname":str,"email":str}]
    """
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
    returns list: [{"jobId":int, "jobName":str, "openDate":datetime|None}], newest first
    """
    search_keys = search_keys or []
    xjob_ids = xjob_ids or []
    base_sql = """
        SELECT j.job_id, jn.name_name AS job_name, j.open_date
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
