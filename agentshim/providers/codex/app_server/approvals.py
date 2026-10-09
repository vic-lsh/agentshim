"""Answering what the server asks of the client.

Codex blocks the turn on every server request until the client replies, so each
one is answered at once, whatever it is. Which refusal depends on the request
and on the ``ApprovalPolicy``: ``DENY`` declines and lets the turn continue,
``FAIL_TURN`` cancels so the turn can be ended with a failure. A request the
client does not understand gets a JSON-RPC error rather than silence.

Refusals by request:

- command and file-change approvals: ``decline`` (``DENY``) or ``cancel``;
- permission grants: an empty grant;
- MCP elicitation: ``decline`` or ``cancel``;
- user-input questions: no answers;
- dynamic tool calls: an unsuccessful, empty result;
- legacy patch and exec approvals: ``denied`` or ``abort``;
- token refresh, attestation, and any method not in the protocol: error
  ``-32601``.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentshim.core.permissions import ApprovalPolicy

from .protocol import (
    ApplyPatchApprovalParams,
    ApplyPatchApprovalResponse,
    CommandExecutionApprovalDecisionKind,
    CommandExecutionRequestApprovalParams,
    CommandExecutionRequestApprovalResponse,
    DynamicToolCallParams,
    DynamicToolCallResponse,
    ErrorResponse,
    ExecCommandApprovalParams,
    ExecCommandApprovalResponse,
    FileChangeApprovalDecision,
    FileChangeRequestApprovalParams,
    FileChangeRequestApprovalResponse,
    GrantedPermissionProfile,
    McpServerElicitationAction,
    McpServerElicitationRequestParams,
    McpServerElicitationRequestResponse,
    PermissionsRequestApprovalParams,
    PermissionsRequestApprovalResponse,
    Response,
    ReviewDecisionDenied,
    ReviewDecisionKind,
    RpcError,
    ServerRequest,
    ServerRequestResult,
    ToolRequestUserInputParams,
    ToolRequestUserInputResponse,
    reply,
)

#: JSON-RPC "method not found".
METHOD_NOT_FOUND = -32601


@dataclass(frozen=True)
class Refusal:
    """An approval or input request the policy refused."""

    #: What was asked, in Codex's words: ``command``, ``file_change``, ``permissions``, ...
    kind: str
    #: The command, reason or question, whichever the request carried.
    detail: str


@dataclass(frozen=True)
class Answer:
    """The reply to one server request, and what it meant.

    ``refusal`` is ``None`` for a request that was not an approval (an
    unsupported method, say), which is answered but not counted as a denial.
    """

    response: Response | ErrorResponse
    refusal: Refusal | None = None


def answer_request(request: ServerRequest, policy: ApprovalPolicy) -> Answer:  # noqa: PLR0911 - one branch per request type
    """Build the reply to *request* under *policy*."""
    cancel = policy is ApprovalPolicy.FAIL_TURN
    params = request.params
    if isinstance(params, CommandExecutionRequestApprovalParams):
        kind = CommandExecutionApprovalDecisionKind
        decision = kind.CANCEL if cancel else kind.DECLINE
        detail = params.command or params.reason or params.item_id
        return _refused(
            request, CommandExecutionRequestApprovalResponse(decision=decision), "command", detail
        )
    if isinstance(params, FileChangeRequestApprovalParams):
        file_decision = (
            FileChangeApprovalDecision.CANCEL if cancel else FileChangeApprovalDecision.DECLINE
        )
        detail = params.reason or params.grant_root or params.item_id
        return _refused(
            request,
            FileChangeRequestApprovalResponse(decision=file_decision),
            "file_change",
            detail,
        )
    if isinstance(params, PermissionsRequestApprovalParams):
        detail = params.reason or params.item_id
        granted = PermissionsRequestApprovalResponse(permissions=GrantedPermissionProfile())
        return _refused(request, granted, "permissions", detail)
    if isinstance(params, McpServerElicitationRequestParams):
        action = McpServerElicitationAction.CANCEL if cancel else McpServerElicitationAction.DECLINE
        return _refused(
            request,
            McpServerElicitationRequestResponse(action=action),
            "mcp_elicitation",
            params.server_name,
        )
    if isinstance(params, ToolRequestUserInputParams):
        detail = "; ".join(question.question for question in params.questions) or params.item_id
        return _refused(request, ToolRequestUserInputResponse(answers={}), "user_input", detail)
    if isinstance(params, DynamicToolCallParams):
        return _refused(
            request,
            DynamicToolCallResponse(content_items=(), success=False),
            "dynamic_tool",
            params.tool,
        )
    if isinstance(params, ApplyPatchApprovalParams):
        decision = ReviewDecisionKind.ABORT if cancel else ReviewDecisionDenied(rejection=_REASON)
        detail = params.reason or params.call_id
        return _refused(
            request, ApplyPatchApprovalResponse(decision=decision), "file_change", detail
        )
    if isinstance(params, ExecCommandApprovalParams):
        decision = ReviewDecisionKind.ABORT if cancel else ReviewDecisionDenied(rejection=_REASON)
        detail = " ".join(params.command) or params.reason or params.call_id
        return _refused(request, ExecCommandApprovalResponse(decision=decision), "command", detail)
    return Answer(_unsupported(request))


_REASON = "refused by the agentshim approval policy"


def _refused(request: ServerRequest, result: ServerRequestResult, kind: str, detail: str) -> Answer:
    return Answer(reply(request.id, result), Refusal(kind, detail))


def _unsupported(request: ServerRequest) -> ErrorResponse:
    error = RpcError(
        code=METHOD_NOT_FOUND, message=f"agentshim does not support {request.method!r}"
    )
    return ErrorResponse(id=request.id, error=error)
