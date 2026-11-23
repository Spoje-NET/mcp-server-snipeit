"""MCP-SERVER-STANDARDS section 7: annotation, naming and read-only checks.

Run in a subprocess so each case gets a fresh import of the server module
(it builds the FastMCP instance from the environment at import time).
"""

import json
import os
import subprocess
import sys

from mcp_server_snipeit.http_auth import WRITE_ACTIONS

_LIST = """
import asyncio, json
from mcp_server_snipeit.mcp_server import mcp
out = []
for t in asyncio.run(mcp.list_tools()):
    a = t.annotations
    out.append([t.name, t.title, a and [a.read_only_hint, a.destructive_hint,
                                         a.idempotent_hint, a.open_world_hint]])
print(json.dumps(out))
"""

_CALL = """
import asyncio
from mcp_server_snipeit.mcp_server import mcp
try:
    asyncio.run(mcp.call_tool("manage_assets", {"action": "delete", "asset_id": 1}))
except Exception as exc:
    print("ERR:", exc)
"""


def _run(code, read_only):
    env = dict(os.environ, SNIPEIT_URL="http://snipe.invalid", SNIPEIT_TOKEN="x")
    env.pop("SNIPEIT_READ_ONLY", None)
    if read_only:
        env["SNIPEIT_READ_ONLY"] = "true"
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                          capture_output=True, text=True, check=True).stdout


def test_every_tool_has_title_and_all_hints():
    tools = json.loads(_run(_LIST, False).splitlines()[-1])
    assert tools
    for name, title, hints in tools:
        assert title and title != name, name
        assert hints is not None and all(h is not None for h in hints), name
        read_only, destructive = hints[0], hints[1]
        assert not (read_only and destructive), name
        if name.startswith(("get_", "list_", "search_")):
            assert read_only, name


def test_read_only_mode_blocks_write_actions():
    assert "read-only" in _run(_CALL, True)


def test_write_actions_include_core_mutations():
    assert {"create", "update", "delete", "checkout", "checkin"} <= WRITE_ACTIONS


_CALL_OAUTH = """
import asyncio
from mcp_server_snipeit.mcp_server import mcp, _auth_provider
assert _auth_provider is not None, "OAuth provider must be active in this test"
try:
    asyncio.run(mcp.call_tool("manage_assets", {"action": "delete", "asset_id": 1}))
except Exception as exc:
    print("ERR:", exc)
"""


def test_read_only_mode_is_enforced_with_oauth():
    """Read-only must not be silently dropped when the OAuth branch is active."""
    env = dict(
        os.environ,
        SNIPEIT_URL="http://snipe.invalid",
        SNIPEIT_OAUTH_CLIENT_ID="cid",
        SNIPEIT_OAUTH_CLIENT_SECRET="secret",
        SNIPEIT_MCP_BASE_URL="https://mcp.invalid",
        MCP_TRANSPORT="http",
        SNIPEIT_READ_ONLY="true",
    )
    env.pop("SNIPEIT_TOKEN", None)
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", _CALL_OAUTH], env=env,
                         capture_output=True, text=True)
    assert "read-only" in out.stdout, (out.stdout, out.stderr[-500:])


def _call(tool, arguments, **extra):
    code = (
        "import asyncio\nfrom mcp_server_snipeit.mcp_server import mcp\n"
        f"try:\n    print('RES', asyncio.run(mcp.call_tool({tool!r}, {arguments!r})))\n"
        "except Exception as exc:\n    print('ERR:', exc)\n"
    )
    env = dict(os.environ, SNIPEIT_URL="http://127.0.0.1:9", SNIPEIT_TOKEN="x", **extra)
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                          capture_output=True, text=True).stdout


def test_every_action_of_a_writing_tool_is_classified():
    from mcp_server_snipeit.http_auth import READ_ACTIONS

    code = """
import asyncio, json
from mcp_server_snipeit.mcp_server import mcp
actions = set()
for t in asyncio.run(mcp.list_tools()):
    if not t.annotations.read_only_hint:
        a = t.parameters["properties"].get("action") or {}
        actions |= set(a.get("enum") or ([a["const"]] if "const" in a else []))
print(json.dumps(sorted(actions)))
"""
    actions = set(json.loads(_run(code, False).splitlines()[-1]))
    assert not READ_ACTIONS & WRITE_ACTIONS
    assert actions <= READ_ACTIONS | WRITE_ACTIONS | {"download"}


def test_read_only_mode_blocks_tools_without_action(tmp_path):
    out = _call("asset_labels", {"asset_ids": [1], "save_path": str(tmp_path / "x")}, SNIPEIT_READ_ONLY="true")
    assert "read-only" in out


def test_read_only_mode_allows_reads():
    out = _call("manage_assets", {"action": "list"}, SNIPEIT_READ_ONLY="true")
    assert "read-only" not in out and "RES" in out


def test_read_only_download_needs_a_file_root(tmp_path):
    arguments = {"action": "download", "asset_id": 1, "file_id": 1, "save_path": str(tmp_path / "f")}
    assert "read-only" in _call("asset_files", arguments, SNIPEIT_READ_ONLY="true")
    out = _call("asset_files", arguments, SNIPEIT_READ_ONLY="true", SNIPEIT_FILE_ROOT=str(tmp_path))
    assert "read-only" not in out and "RES" in out


def test_file_root_confines_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(tmp_path)
    for path in ("/etc/passwd", f"{root}/../x", f"{root}/link/x", "relative"):
        arguments = {"action": "upload", "asset_id": 1, "file_paths": [path]}
        assert "SNIPEIT_FILE_ROOT" in _call("asset_files", arguments, SNIPEIT_FILE_ROOT=str(root)), path
    # the tool's own default path is outside the root as well
    assert "SNIPEIT_FILE_ROOT" in _call("asset_labels", {"asset_ids": [1]}, SNIPEIT_FILE_ROOT=str(root))


def test_mistyped_read_only_refuses_to_start():
    assert _call("manage_assets", {"action": "list"}, SNIPEIT_READ_ONLY="ture") == ""


def test_direct_api_requests_have_a_timeout(monkeypatch):
    from mcp_server_snipeit import client

    monkeypatch.setenv("SNIPEIT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SNIPEIT_TOKEN", "x")
    seen = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def request(method, url, **kwargs):
        seen.update(kwargs)
        return Response()

    monkeypatch.setattr(client.requests, "request", request)
    client.SnipeITDirectAPI().get("hardware", 1)
    assert seen["timeout"] == 30
