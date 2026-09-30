"""Typed VIRA tools — the model-facing contract for every non-bash runtime.

mini-swe-agent sees VIRA as a CLI (recruiter_cli.py + commands.md).  Every
other runtime sees these typed functions instead: LangGraph and
deepagents via langchain_tools(), any MCP client via vira_mcp.py.  The
signatures (Annotated + pydantic Field) and docstrings ARE the tool schema
and description the model gets, so edit them like you'd edit commands.md.

Each tool calls the matching recruiter_cli action, i.e. the same guarded
path as the CLI: confirm gate -> VIRA -> audit -> PII mask.  Rules that hold
for every tool:
  * the mode (real | mock) is set once by the host via configure(); the model
    never chooses it.
  * confirmed=False always: a confirm-gated command can't run from here.
  * failures come back as {"status": "error", ...} dicts, never exceptions,
    and never carry hosts or URLs into model context.

Keep `from __future__ import annotations` out of this file: LangChain and
FastMCP both build the tool schema from these annotations at runtime.
"""
import inspect
from typing import Annotated, Any, Callable, Dict

from pydantic import Field

import recruiter_cli as vira

_MODE = "mock"

# The model sees these limits in the schema; recruiter_cli enforces the same numbers.
Id = Annotated[int, Field(ge=1)]
Text = Annotated[str, Field(max_length=vira.MAX_TEXT)]


def configure(mode: str) -> None:
    """Set real|mock for every tool call in this process."""
    if mode not in ("real", "mock"):
        raise ValueError(f"mode must be 'real' or 'mock', not {mode!r}")
    global _MODE
    _MODE = mode


def current_mode() -> str:
    return _MODE


def _call(action: Callable[..., Dict], *args, **kwargs) -> Dict:
    try:
        return action(*args, mode=_MODE, confirmed=False, **kwargs)
    except Exception as exc:  # e.g. VIRA unreachable; the message may name hosts, so drop it
        return {"status": "error", "message": f"VIRA call failed ({type(exc).__name__})"}


# --- the tools ---------------------------------------------------------------
def find_talents(
    job_ids: Annotated[list[Id], Field(
        min_length=1, max_length=vira.MAX_IDS, description="Job ids to source talents for.")],
    profile_ids: Annotated[list[Id] | None, Field(
        max_length=vira.MAX_IDS, description="Optional: only consider these profile ids.")] = None,
) -> Dict[str, Any]:
    """Find potential/suggested talents for one or more jobs (read-only).

    Returns profile_id values and their scores (overall_score, skill_score,
    job_name_similarity). These are profile ids: they are neither match ids
    nor application ids. To score them, get their match ids from
    get_match_id_from_profile_id first.
    """
    return _call(vira.find_talents, job_ids, profile_ids or [])


def get_match_id_from_profile_id(
    job_id: Annotated[Id, Field(description="The job the talents were found for.")],
    profile_ids: Annotated[list[Id], Field(
        min_length=1, max_length=vira.MAX_IDS,
        description="profile_id values that find_talents returned for that job.")],
) -> Dict[str, Any]:
    """Look up the match ids of suggested talents for one job (read-only).

    Returns matches[] with a profile_id and its match_id. Pass those match_id
    values to score_candidates or candidate_insights.
    """
    return _call(vira.get_match_id_from_profile_id, job_id, profile_ids)


def generate_jd(
    job_title: Annotated[str, Field(
        min_length=1, max_length=vira.MAX_TEXT,
        description="Just the job title the user gave; skills go in `skills`.")],
    skills: Annotated[list[Text] | None, Field(
        max_length=vira.MAX_ITEMS,
        description="Skills the user named. Leave empty if they named none.")] = None,
    lang: Annotated[str, Field(
        pattern=vira.LANG_RE.pattern,
        description="Language code of the posting, e.g. en, ar.")] = "en",
    job_id: Annotated[Id | None, Field(
        description="Existing job id, only if the user gave one.")] = None,
    job_function: Annotated[list[Text] | None, Field(
        max_length=vira.MAX_ITEMS,
        description="Job functions the user named. Leave empty if they named none.")] = None,
    industry: Annotated[list[Text] | None, Field(
        max_length=vira.MAX_ITEMS,
        description="Industries the user named. Leave empty if they named none.")] = None,
    other_requirements: Annotated[list[Text] | None, Field(
        max_length=vira.MAX_ITEMS,
        description="Other requirements the user stated. Leave empty if they stated none.")] = None,
) -> Dict[str, Any]:
    """LLM-generate a job posting / job description.

    VIRA writes the description: pass only what the user gave, never
    requirements of your own. Returns the job_description text.
    """
    return _call(vira.generate_jd, job_title=job_title, skills=skills or [],
                 lang=lang or "en", job_id=job_id or 0, job_function=job_function or [],
                 industry=industry or [], other_requirements=other_requirements or [])


def score_candidates(
    app_ids: Annotated[list[Id] | None, Field(
        max_length=vira.MAX_IDS, description="Application ids (applicants who applied).")] = None,
    match_ids: Annotated[list[Id] | None, Field(
        max_length=vira.MAX_IDS,
        description="Match ids of suggested talents, from get_match_id_from_profile_id. "
                    "Not profile ids.")] = None,
) -> Dict[str, Any]:
    """Trigger CV scoring (composite score + briq) for applicants and/or suggested talents.

    Pass app_ids, match_ids, or both. Returns scores[] with composite_score
    and briq per id.
    """
    if not app_ids and not match_ids:
        return {"status": "error", "message": "pass app_ids and/or match_ids"}
    return _call(vira.score_candidates, app_ids or [], match_ids or [])


def candidate_insights(
    app_ids: Annotated[list[Id], Field(
        min_length=1, max_length=vira.MAX_IDS, description="Application ids to get insights for.")],
    match_ids: Annotated[list[Id] | None, Field(
        max_length=vira.MAX_IDS, description="Optional match ids. Not profile ids.")] = None,
) -> Dict[str, Any]:
    """Trigger candidate insights (v3) for applicants.

    Returns insights per id.
    """
    return _call(vira.candidate_insights, app_ids, match_ids or [])


TOOLS = [find_talents, get_match_id_from_profile_id, generate_jd, score_candidates,
         candidate_insights]
NAMES = {fn.__name__ for fn in TOOLS}
# Pure reads.  score/insights *trigger* calculations on VIRA (e.g. recal_briq),
# so they are advertised as idempotent, not read-only.
READ_ONLY = {"find_talents", "get_match_id_from_profile_id", "generate_jd"}


def description(fn: Callable) -> str:
    return inspect.getdoc(fn) or ""


def langchain_tools() -> list:
    """The tools as LangChain StructuredTools (LangGraph, deepagents)."""
    from langchain_core.tools import StructuredTool
    return [StructuredTool.from_function(fn, description=description(fn)) for fn in TOOLS]
