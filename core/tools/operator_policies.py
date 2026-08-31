"""Operator-only controls for inspecting and revoking standing policy."""

from __future__ import annotations

from typing import Any

from .base import (
    ToolCategory,
    ToolContext,
    ToolErrorType,
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
)


class ManageOperatorPoliciesHandler(ToolHandler):
    """Expose policy control only inside an identity-verified private turn."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="manage_operator_policies",
            description=(
                "List or revoke identity-verified standing instructions. Use list "
                "when the operator asks what permanent preferences or instructions "
                "are active. Revoke only when the operator explicitly asks to forget, "
                "remove, stop following, or replace a policy. To replace one, revoke "
                "it and ask the operator to state the new standing instruction."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "revoke"],
                    },
                    "policy_key": {
                        "type": "string",
                        "description": "Exact operator.standing.* key; required for revoke.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the operator asked to revoke the policy.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 50,
                    },
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            category=ToolCategory.MEMORY,
            energy_cost=0,
            is_read_only=False,
            supports_parallel=False,
            allowed_contexts={ToolContext.CHAT},
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if context.registry is None:
            return ToolResult.error_result(
                "manage_operator_policies requires a database-backed registry",
                ToolErrorType.EXECUTION_FAILED,
            )
        if context.is_group or not context.is_operator:
            return ToolResult.error_result(
                "Standing policies can only be inspected or changed in a private, identity-verified operator turn.",
                ToolErrorType.PERMISSION_DENIED,
            )

        action = str(arguments.get("action") or "").strip().lower()
        from services.operator_policy_corrections import (
            list_operator_policies,
            revoke_operator_policy,
        )

        if action == "list":
            result = await list_operator_policies(
                context.registry.pool,
                limit=int(arguments.get("limit") or 50),
            )
            if not result.get("ok"):
                return ToolResult.error_result(
                    str(result.get("next_step") or "Could not list operator policies."),
                    ToolErrorType.EXECUTION_FAILED,
                )
            return ToolResult.success_result(result)

        if action == "revoke":
            policy_key = str(arguments.get("policy_key") or "").strip()
            if not policy_key:
                return ToolResult.error_result(
                    "policy_key is required for revoke; list active policies first.",
                    ToolErrorType.INVALID_PARAMS,
                )
            result = await revoke_operator_policy(
                context.registry.pool,
                policy_key=policy_key,
                actor=f"operator:{context.surface}",
                reason=str(arguments.get("reason") or "").strip() or None,
                event_id=context.call_id,
            )
            if not result.get("revoked"):
                return ToolResult.error_result(
                    str(result.get("next_step") or "The policy was not revoked."),
                    ToolErrorType.EXECUTION_FAILED,
                )
            return ToolResult.success_result(result)

        return ToolResult.error_result(
            "action must be list or revoke",
            ToolErrorType.INVALID_PARAMS,
        )


def create_operator_policy_tools() -> list[ToolHandler]:
    return [ManageOperatorPoliciesHandler()]


__all__ = ["ManageOperatorPoliciesHandler", "create_operator_policy_tools"]
