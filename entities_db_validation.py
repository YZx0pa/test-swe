"""Deterministic entity validation for Jeni write-tool calls.

This module composes reusable DB query contracts by action arguments.  It does
not create a SQL helper per task, and it does not expose validation as an LLM
tool.  The LLM gathers a complete payload; this middleware verifies referenced
entities before the VIRA tool can execute it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

import pii_vault

LOG = logging.getLogger(__name__)
STAGE_LOG = logging.getLogger("jeni.stages")


def _stage(event: str, **details) -> None:
    values = " ".join(f"{key}={value}" for key, value in sorted(details.items())
                      if value is not None)
    STAGE_LOG.info("%s%s", event, f" {values}" if values else "")

ROLE_IDS = frozenset({1, 5})


def _error(request, error_type: str, message: str, **details) -> ToolMessage:
    payload = {"status": "validation_error", "error_type": error_type,
               "message": message, "user_action_required": True, **details}
    return ToolMessage(content=json.dumps(payload, default=str),
                       tool_call_id=request.tool_call["id"],
                       name=request.tool_call["name"], status="error")


def _unavailable(request, check: str, result: Any) -> ToolMessage:
    payload = {"status": "validation_unavailable", "check": check,
               "message": "The entity validation query could not be completed; the action was not run.",
               "user_action_required": True}
    LOG.warning("ENTITY_DB_VALIDATION_UNAVAILABLE tool=%s check=%s result=%r",
                request.tool_call["name"], check, result)
    return ToolMessage(content=json.dumps(payload, default=str),
                       tool_call_id=request.tool_call["id"],
                       name=request.tool_call["name"], status="error")


def _email_values(args: Mapping[str, Any], field: str) -> list[str]:
    values = args[field] if isinstance(args.get(field), list) else [args[field]]
    return [pii_vault.VAULT.resolve(pii_vault.resolve_me(field, value)).lower()
            for value in values]


@dataclass(frozen=True)
class ValidationConfig:
    query_tools: Mapping[str, Any]
    context: Mapping[str, Any]
    action_names: frozenset[str]
    pre_approval_names: frozenset[str]


class EntitiesDBValidationMiddleware(AgentMiddleware):
    """Validate Jeni action entities before a tool handler runs.

    ``phase='pre_approval'`` validates calls that will interrupt before their
    card is shown.  ``phase='execute'`` validates ungated writes and reviewer
    edits after the card.  Thus an ordinary write is checked once, an approved
    unchanged write is checked once before its card, and an edited write is
    checked again with the edited payload.
    """

    def __init__(self, config: ValidationConfig, phase: str):
        super().__init__()
        self.config, self.phase = config, phase

    def _applies(self, request) -> bool:
        name = request.tool_call["name"]
        if name not in self.config.action_names:
            return False
        gated = name in self.config.pre_approval_names
        if self.phase == "pre_approval":
            return gated
        if not gated:
            return True
        # HumanInTheLoopMiddleware records only changed calls here. An unchanged
        # approval has already passed the outer pre-approval validation.
        edits = request.state.get("hitl_edited_tool_calls") or {}
        return request.tool_call.get("id") in edits

    async def _query(self, name: str, inputs: dict) -> dict:
        query = self.config.query_tools[name]
        return await query.handler(inputs, self.config.context)

    async def _validate(self, request) -> ToolMessage | None:
        args = request.tool_call.get("args") or {}
        _stage("DB_VALIDATION", tool=request.tool_call["name"], phase=self.phase,
               fields=",".join(sorted(args)))

        # One job-detail result covers all job-targeting actions.  It is also
        # the canonical title/state source for future title-reference/state rules.
        if "job_id" in args:
            result = await self._query("get_job_detail", {"job_id": args["job_id"]})
            if result.get("status") == "error":
                return _unavailable(request, "job_target", result)
            if result.get("status") != "resolved":
                return _error(request, "job_not_found", "The job does not exist in this company.",
                              field="job_id", value=args["job_id"])

        if "app_ids" in args:
            result = await self._query("validate_app_ids", {"app_ids": args["app_ids"]})
            if result.get("status") == "error":
                return _unavailable(request, "applications", result)
            if result.get("status") != "resolved":
                return _error(request, "application_not_found",
                              "One or more applications do not exist in this company.",
                              field="app_ids", invalid_values=result.get("invalid_values", []))

        if "user_ids" in args:
            result = await self._query("validate_user_ids", {"user_ids": args["user_ids"]})
            if result.get("status") == "error":
                return _unavailable(request, "eligible_users", result)
            if result.get("status") != "resolved":
                return _error(request, "user_not_eligible",
                              "One or more users are not active role-4 users in this company.",
                              field="user_ids", invalid_values=result.get("invalid_values", []))

        if request.tool_call["name"] == "add_job_collaborators":
            role_id = args.get("role_id")
            if role_id not in ROLE_IDS:
                return _error(request, "invalid_collaborator_role",
                              "Collaborator role must be 1 (administrator) or 5 (team member).",
                              field="role_id", value=role_id)

        for field in ("new_owner_user_email",):
            if field in args:
                try:
                    emails = _email_values(args, field)
                except LookupError:
                    return _error(request, "unknown_email", "The supplied email token cannot be resolved.",
                                  field=field)
                result = await self._query("validate_emails", {"emails": emails})
                if result.get("status") == "error":
                    return _unavailable(request, "eligible_internal_emails", result)
                if result.get("status") != "resolved":
                    return _error(request, "email_not_eligible",
                                  "The email is not an active role-4 user in this company.",
                                  field=field, invalid_values=result.get("invalid_values", []))

        if request.tool_call["name"] == "share_application" and "emails" in args:
            try:
                emails = _email_values(args, "emails")
            except LookupError:
                return _error(request, "unknown_email", "A supplied email token cannot be resolved.",
                              field="emails")
            result = await self._query("validate_emails", {"emails": emails})
            if result.get("status") == "error":
                return _unavailable(request, "eligible_internal_emails", result)
            if result.get("status") != "resolved":
                return _error(request, "email_not_eligible",
                              "One or more emails are not active role-4 users in this company.",
                              field="emails", invalid_values=result.get("invalid_values", []))

        if request.tool_call["name"] == "create_application_to_job" and "candidate_email" in args:
            result = await self._query("find_candidate_by_email", {"email": args["candidate_email"]})
            if result.get("status") == "error":
                return _unavailable(request, "candidate_email", result)
            if result.get("status") != "resolved":
                return _error(request, "candidate_not_found",
                              "No candidate with this email has an application in this company.",
                              field="candidate_email")

        if not any(key in args for key in ("job_id", "app_ids", "user_ids", "new_owner_user_email",
                                            "emails", "candidate_email")):
            LOG.info("ENTITY_DB_VALIDATION_SKIPPED tool=%s reason=no_configured_entity_fields",
                     request.tool_call["name"])
        _stage("DB_VALIDATION_RESULT", tool=request.tool_call["name"], phase=self.phase,
               result="passed")
        return None

    async def awrap_tool_call(self, request, handler):
        if not self._applies(request):
            return await handler(request)
        try:
            refused = await self._validate(request)
        except Exception as exc:
            return _unavailable(request, "unexpected_exception", {"type": type(exc).__name__})
        if refused is not None:
            _stage("DB_VALIDATION_RESULT", tool=request.tool_call["name"], phase=self.phase,
                   result="refused")
            return refused
        return await handler(request)


class PreApprovalEntitiesDBValidationMiddleware(EntitiesDBValidationMiddleware):
    """Distinct LangChain middleware type for validation before an approval card."""

    def __init__(self, config: ValidationConfig):
        super().__init__(config, "pre_approval")


class ExecutionEntitiesDBValidationMiddleware(EntitiesDBValidationMiddleware):
    """Distinct LangChain middleware type for normal writes and reviewer edits."""

    def __init__(self, config: ValidationConfig):
        super().__init__(config, "execute")


def middleware(query_tools: Mapping[str, Any], context: Mapping[str, Any], *,
               action_names: set[str] | frozenset[str],
               pre_approval_names: set[str] | frozenset[str]) -> list[AgentMiddleware]:
    """Outer pre-card + inner execution-time validation middleware pair."""
    config = ValidationConfig(query_tools=query_tools, context=context,
                              action_names=frozenset(action_names),
                              pre_approval_names=frozenset(pre_approval_names))
    # LangChain forbids duplicate middleware *types*. The stages intentionally
    # coexist on opposite sides of HumanInTheLoopMiddleware, so they use two
    # distinct subclasses while sharing all validation behaviour above.
    return [PreApprovalEntitiesDBValidationMiddleware(config),
            ExecutionEntitiesDBValidationMiddleware(config)]


def stages(query_tools, context, *, action_names, read_only, pre_approval_names):
    """Validation middleware positioned on either side of an optional approval layer.

    The caller owns HumanInTheLoopMiddleware and therefore performs the final
    ordering.  Empty tuples let non-DB toolsets use the same build path without
    importing or configuring entity validation.
    """
    if query_tools is None or context is None:
        return (), ()
    pre, execute = middleware(
        query_tools, context,
        action_names=set(action_names) - set(read_only),
        pre_approval_names=set(pre_approval_names),
    )
    return (pre,), (execute,)
