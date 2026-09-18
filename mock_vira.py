"""MockVira — a local stand-in for the real VIRA backend.

Lets the entire mini-swe-agent loop run with no server, so you can watch the
trajectory and the guardrails before pointing recruiter_cli at your localhost.
Responses are deliberately synthetic and clearly marked.
"""
from __future__ import annotations

from typing import Any, Dict


class MockVira:
    @staticmethod
    def call(path: str, query: Dict[str, Any], body: Dict[str, Any]) -> Dict:
        if path == "fast_retargeting":
            job_ids = body.get("job_ids") or []
            if not job_ids:
                return {"status": "error", "http_status": 400,
                        "result": {"message": "job_ids is required"}}
            # synthetic suggested talents (profile ids only; no PII)
            return {"status": "ok", "http_status": 200, "result": {
                "job_id": job_ids[0],
                "suggested_profiles": [
                    {"profile_id": 900001, "match_score": 0.91},
                    {"profile_id": 900002, "match_score": 0.87},
                    {"profile_id": 900003, "match_score": 0.83},
                ],
                "_note": "SYNTHETIC mock response"}}

        if path == "JD_generation/jd_generation":
            title = body.get("job_title") or "(untitled)"
            lang = query.get("lang", "en")
            return {"status": "ok", "http_status": 200, "result": {
                "lang": lang,
                "job_description": f"[{lang}] Draft JD for '{title}'. "
                                   f"Responsibilities... Requirements... (mock)",
                "skills_used": body.get("skills", []),
                "_note": "SYNTHETIC mock response"}}

        if path == "candidate_score_calculation":
            app_ids = body.get("app_ids") or []
            if "version" in query:  # #4 insights
                return {"status": "ok", "http_status": 200, "result": {
                    "version": query["version"],
                    "insights": [{"app_id": a, "summary": "strong backend fit (mock)"}
                                 for a in app_ids],
                    "_note": "SYNTHETIC mock response"}}
            # #3 scoring — scores applicants (app_ids) and/or suggested
            # talents (match_ids)
            match_ids = body.get("match_ids") or []
            scored = ([{"app_id": a, "composite_score": 0.78, "briq": 0.81} for a in app_ids]
                      + [{"match_id": m, "composite_score": 0.74, "briq": 0.69} for m in match_ids])
            return {"status": "ok", "http_status": 200, "result": {
                "scores": scored, "_note": "SYNTHETIC mock response"}}

        return {"status": "error", "http_status": 404,
                "result": {"message": f"unknown path {path}"}}
