"""HTTP auth layer, audit log, and per-identity policy for multi-identity mode.

Two cooperating pieces, both active only in :class:`~mcp_server_snipeit.config.AuthMode.MULTI_IDENTITY`:

* :class:`MultiIdentityAuthMiddleware` — a pure ASGI middleware in front of the
  FastMCP app. Serves the unauthenticated ``GET /healthz`` endpoint, validates
  the ``Authorization: Bearer <token>`` header on every other request, answers
  ``401`` with a ``WWW-Authenticate: Bearer`` challenge when the token is
  missing or unknown, and stores the resolved identity in the
  ``current_identity`` contextvar for the duration of the request. Registered
  via ``FastMCP.run(transport="http", middleware=[...])``.

* :class:`IdentityToolMiddleware` — a FastMCP protocol middleware that runs per
  tool call. Enforces the per-identity tool allowlist and the read-only flag
  (a denied call is answered as a tool error, before the tool function runs),
  filters ``tools/list`` per identity, and emits one JSON audit line per call
  on the ``mcp_server_snipeit.audit`` logger (stderr). Registered via the ``FastMCP``
  constructor's ``middleware`` argument.

The audit line never contains full arguments — only a short SHA-256 digest —
and never contains any token. It is the second, Snipe-IT-independent trail:
identity, tool, action, outcome, and duration per call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from fastmcp.server.middleware import Middleware, MiddlewareContext
from mcp.types import CallToolRequestParams, ListToolsRequest

from .identity import Identity, IdentityRegistry, current_identity

AUDIT_LOGGER_NAME = "mcp_server_snipeit.audit"
HEALTHZ_PATH = "/healthz"
ARGS_DIGEST_LENGTH = 16

# Tool ``action`` values that modify Snipe-IT state. READ_ONLY identities are
# blocked from these; everything else (list/get/download/...) stays allowed.
# Keep in sync with the action literals in :mod:`mcp_server_snipeit.tools`.
WRITE_ACTIONS: frozenset[str] = frozenset(
    {
        "create",
        "update",
        "delete",
        "checkout",
        "checkin",
        "audit",
        "restore",
        "upload",
        "complete",
        "request",
        "cancel",
        "edit",
        "associate",
        "disassociate",
        "reorder",
        "add_item",
        "update_item",
        "remove_item",
        "reset",
        "sync",
        "process",
    }
)


# Actions allowed in server-wide read-only mode on tools that can also write.
# An allowlist: an action added later is refused until it is classified here.
READ_ACTIONS: frozenset[str] = frozenset(
    {
        "assets",
        "fields",
        "get",
        "list",
        "list_assets",
        "list_checkouts",
        "list_items",
        "me",
        "requestable",
        "test",
        "users",
    }
)

# Tool parameters naming files on the server's own disk.
PATH_PARAMETERS = ("save_path", "file_path", "file_paths")


def file_root() -> Path | None:
    """Directory local file access is confined to (``SNIPEIT_FILE_ROOT``)."""
    value = os.getenv("SNIPEIT_FILE_ROOT", "").strip()
    return Path(value).resolve() if value else None


def _check_path(root: Path, value: Any) -> None:
    if not isinstance(value, str):
        raise PermissionError("File paths must be strings.")
    # resolve() follows symlinks, so a link inside the root cannot lead out
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(root):
        raise PermissionError("File access is limited to absolute paths inside SNIPEIT_FILE_ROOT.")


async def _tool_of(context: MiddlewareContext) -> Any:
    if context.fastmcp_context is None:
        return None
    return await context.fastmcp_context.fastmcp.get_tool(context.message.name)


class FileRootMiddleware(Middleware):
    """Confine tool parameters naming local files to ``SNIPEIT_FILE_ROOT``."""

    async def on_call_tool(self, context: MiddlewareContext[CallToolRequestParams], call_next: Any) -> Any:
        root = file_root()
        if root is not None:
            arguments = context.message.arguments or {}
            tool = await _tool_of(context)
            properties = (tool.parameters.get("properties") or {}) if tool else {}
            for name in PATH_PARAMETERS:
                value = arguments.get(name)
                if value is None:
                    # the tool would fall back to its own default path
                    value = (properties.get(name) or {}).get("default")
                for item in value if isinstance(value, list) else [value]:
                    if item is not None:
                        _check_path(root, item)
        return await call_next(context)


def _extract_bearer_token(headers: Sequence[tuple[bytes, bytes]]) -> str | None:
    """Return the bearer token from an ASGI header list, or ``None``."""
    for name, value in headers:
        if name.lower() != b"authorization":
            continue
        parts = value.decode("latin-1").split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() or None
        return None
    return None


async def _send_json(
    send: Any,
    status_code: int,
    payload: dict[str, Any],
    extra_headers: Sequence[tuple[bytes, bytes]] = (),
) -> None:
    """Send a complete JSON ASGI response."""
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class MultiIdentityAuthMiddleware:
    """ASGI middleware: unauthenticated ``/healthz`` + per-request identity resolution.

    Every HTTP request except ``GET /healthz`` must carry
    ``Authorization: Bearer <mcp_token>`` for a known identity; anything else
    gets a ``401`` with a ``WWW-Authenticate: Bearer`` challenge before the
    FastMCP app (and thus any tool) is touched.
    """

    def __init__(self, app: Any, registry: IdentityRegistry):
        self.app = app
        self.registry = registry
        self._logger = logging.getLogger(__name__)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("method") == "GET" and scope.get("path") == HEALTHZ_PATH:
            await _send_json(send, 200, {"status": "ok", "identities": len(self.registry)})
            return

        token = _extract_bearer_token(scope.get("headers") or [])
        identity = self.registry.lookup(token)
        if identity is None:
            self._logger.warning("Rejected request to %s: missing or unknown bearer token",
                                 scope.get("path"))
            await _send_json(
                send,
                401,
                {"error": "Unauthorized: missing or invalid bearer token"},
                extra_headers=[(b"www-authenticate", b"Bearer")],
            )
            return

        # Set in this request's task; anyio copies the context into the worker
        # threads that run the (sync) tool functions for this request/session.
        context_token = current_identity.set(identity)
        try:
            await self.app(scope, receive, send)
        finally:
            current_identity.reset(context_token)


def _args_digest(arguments: Any) -> str:
    """Short SHA-256 digest of the tool arguments — never the arguments themselves."""
    if not arguments:
        return ""
    canonical = json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:ARGS_DIGEST_LENGTH]


class ReadOnlyToolMiddleware(Middleware):
    """Refuse everything but reads (``SNIPEIT_READ_ONLY=true``)."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: Any,
    ) -> Any:
        arguments = context.message.arguments or {}
        action = arguments.get("action") if isinstance(arguments, dict) else None
        tool = await _tool_of(context)
        annotations = getattr(tool, "annotations", None)
        read_tool = annotations is not None and annotations.read_only_hint is True
        # Downloads write to the server's disk: only into a configured directory.
        download = action == "download" and file_root() is not None
        if not (action in READ_ACTIONS or download or (read_tool and action not in WRITE_ACTIONS)):
            raise PermissionError(
                f"Action '{action or context.message.name}' is blocked: "
                "the server runs in read-only mode (SNIPEIT_READ_ONLY=true)."
            )
        return await call_next(context)


class IdentityToolMiddleware(Middleware):
    """Per-tool-call policy enforcement and audit logging (multi-identity mode)."""

    def __init__(self, registry: IdentityRegistry):
        self.registry = registry
        self._audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)

    def _audit(
        self,
        identity: Identity | None,
        tool_name: str,
        action: Any,
        *,
        ok: bool,
        duration_ms: int,
        arguments: Any,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "identity": identity.key if identity is not None else None,
            "tool": tool_name,
            "action": action if isinstance(action, str) else None,
            "ok": ok,
            "duration_ms": duration_ms,
            "args_digest": _args_digest(arguments),
        }
        self._audit_logger.info(json.dumps(record, separators=(",", ":")))

    def _policy_denial(self, identity: Identity, tool_name: str, action: Any) -> str | None:
        """Return an error message if the identity may not run this call, else ``None``."""
        if identity.allowed_tools is not None and tool_name not in identity.allowed_tools:
            return (
                f"Tool '{tool_name}' is not allowed for identity '{identity.log_name}'. "
                f"Allowed tools: {sorted(identity.allowed_tools)}"
            )
        if identity.read_only and isinstance(action, str) and action in WRITE_ACTIONS:
            return (
                f"Action '{action}' modifies Snipe-IT state and is blocked for "
                f"read-only identity '{identity.log_name}'."
            )
        return None

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: Any,
    ) -> Any:
        identity = current_identity.get()
        params = context.message
        tool_name = params.name
        arguments = params.arguments or {}
        action = arguments.get("action") if isinstance(arguments, dict) else None

        if identity is not None:
            denial = self._policy_denial(identity, tool_name, action)
            if denial is not None:
                self._audit(identity, tool_name, action, ok=False, duration_ms=0,
                            arguments=arguments)
                # Raising surfaces to the client as a tool error (isError=true)
                # with the message — no tool function is executed.
                raise PermissionError(denial)

        start = time.monotonic()
        try:
            result = await call_next(context)
        except Exception:
            self._audit(identity, tool_name, action, ok=False,
                        duration_ms=int((time.monotonic() - start) * 1000), arguments=arguments)
            raise
        self._audit(identity, tool_name, action, ok=True,
                    duration_ms=int((time.monotonic() - start) * 1000), arguments=arguments)
        return result

    async def on_list_tools(
        self,
        context: MiddlewareContext[ListToolsRequest],
        call_next: Any,
    ) -> Sequence[Any]:
        tools = await call_next(context)
        identity = current_identity.get()
        if identity is not None and identity.allowed_tools is not None:
            return [tool for tool in tools if tool.name in identity.allowed_tools]
        return tools