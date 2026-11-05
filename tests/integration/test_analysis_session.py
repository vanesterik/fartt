"""
Integration tests for the analysis session's permission profile (#132),
driven through the real Claude Code CLI: `task analyse` with Claude Code's
print mode appended (`-- -p --permission-prompts none --output-format
stream-json --verbose`), so the session runs headless with the profile's own
flags. With `--permission-prompts none` anything that would ask is denied, so
a tool that runs did so without a prompt.

The stream's `init` message lists the tools, MCP servers, slash commands and
permission mode of the session; every tool call and its result follow. Each
run has `FARTT_ASSETS_DIR` set to a temporary folder, so the operator's cache
and journal are never touched.

These tests start Claude Code and make model calls (a few per run), and the
server they connect calls Yahoo, so they're marked `network` and skipped
when `claude` or `task` isn't installed. Rerun them after a Claude Code
update: its flags change between releases (see CLAUDE.md, Running an
analysis).

Each test's docstring starts with its tags: `@issue-132` and the acceptance
criteria it verifies. AC7 (an interactive analysis by the operator) and AC8
(CLAUDE.md) are checked by the test role, not here.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        shutil.which("claude") is None or shutil.which("task") is None,
        reason="needs the claude and task CLIs",
    ),
]

ROOT = Path(__file__).resolve().parents[2]
RUN_TIMEOUT_S = 300
PRINT_MODE = [
    "-p",
    "--permission-prompts",
    "none",
    "--output-format",
    "stream-json",
    "--verbose",
]
FARTT_TOOLS = {
    "mcp__fartt__get_candles",
    "mcp__fartt__get_quote",
    "mcp__fartt__get_indicators",
    "mcp__fartt__get_trend",
    "mcp__fartt__get_support_resistance",
    "mcp__fartt__get_decisions",
    "mcp__fartt__get_performance",
    "mcp__fartt__record_decision",
    "mcp__fartt__score_decisions",
}
OUTSIDE_TOOLS = {
    "Bash",
    "Read",
    "Write",
    "Edit",
    "WebSearch",
    "WebFetch",
    "Agent",
    "Task",
    "Skill",
}

Messages = list[dict[str, Any]]


def _env(assets_dir: Path | None) -> dict[str, str]:
    """The environment of a fresh, top-level Claude Code run."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CLAUDE") and key != "AI_AGENT"
    }
    if assets_dir is not None:
        env["FARTT_ASSETS_DIR"] = str(assets_dir)
    return env


def _stream(command: list[str], assets_dir: Path | None) -> Messages:
    done = subprocess.run(
        command,
        cwd=ROOT,
        env=_env(assets_dir),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return [json.loads(line) for line in done.stdout.splitlines() if line.strip()]


def _analysis(prompt: str, assets_dir: Path) -> Messages:
    """One headless analysis session, started by `task analyse`."""
    # Task hands CLI_ARGS to the shell, so the prompt goes in single quotes.
    quoted = "'" + prompt.replace("'", "") + "'"
    return _stream(["task", "analyse", "--", *PRINT_MODE, quoted], assets_dir)


def _init(messages: Messages) -> dict[str, Any]:
    return next(
        m for m in messages if m["type"] == "system" and m.get("subtype") == "init"
    )


def _result(messages: Messages) -> dict[str, Any]:
    return next(m for m in messages if m["type"] == "result")


def _tool_uses(messages: Messages) -> list[dict[str, Any]]:
    return [
        block
        for m in messages
        if m["type"] == "assistant"
        for block in m["message"]["content"]
        if block["type"] == "tool_use"
    ]


def _tool_results(messages: Messages) -> list[dict[str, Any]]:
    return [
        block
        for m in messages
        if m["type"] == "user" and isinstance(m["message"]["content"], list)
        for block in m["message"]["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def _working_tree() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _assets_fingerprint() -> dict[str, tuple[int, int]]:
    assets = ROOT / "assets"
    return {
        str(path.relative_to(assets)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(assets.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def quote_session(tmp_path_factory: pytest.TempPathFactory) -> Messages:
    """An analysis session asked for one quote, shared by the AC2/3/5 tests."""
    assets_dir = tmp_path_factory.mktemp("analysis-assets")
    return _analysis(
        "Call get_quote for ASML.AS and report the last price.", assets_dir
    )


def test_task_list_shows_analyse() -> None:
    """@issue-132 @AC1 `task --list` names the analyse task and its alias."""
    listed = subprocess.run(
        ["task", "--list"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    line = next(row for row in listed.splitlines() if "analyse:" in row)
    assert "analyze" in line


def test_analyse_connects_the_fartt_server(quote_session: Messages) -> None:
    """@issue-132 @AC1 `task analyse` starts Claude Code with fartt connected."""
    servers = _init(quote_session)["mcp_servers"]
    assert {"name": "fartt", "status": "connected"}.items() <= servers[0].items()


def test_only_the_fartt_tools_are_offered(quote_session: Messages) -> None:
    """@issue-132 @AC2 The session offers fartt's tools and nothing else."""
    init = _init(quote_session)
    tools = set(init["tools"])
    assert FARTT_TOOLS <= tools
    assert all(tool.startswith("mcp__fartt__") for tool in tools), tools
    assert not tools & OUTSIDE_TOOLS
    assert [server["name"] for server in init["mcp_servers"]] == ["fartt"]


def test_a_fartt_tool_runs_without_a_prompt(quote_session: Messages) -> None:
    """
    @issue-132 @AC3 get_quote on ASML.AS runs without a prompt, in default
    mode whatever the user settings' defaultMode is.
    """
    assert _init(quote_session)["permissionMode"] == "default"
    uses = _tool_uses(quote_session)
    assert [use["name"] for use in uses] == ["mcp__fartt__get_quote"]
    results = _tool_results(quote_session)
    assert len(results) == 1
    assert not results[0].get("is_error")
    assert "ASML.AS" in json.dumps(results[0]["content"])
    assert _result(quote_session)["permission_denials"] == []


def test_the_analyze_ticker_prompt_is_available(quote_session: Messages) -> None:
    """@issue-132 @AC5 The session has /mcp__fartt__analyze_ticker."""
    assert "mcp__fartt__analyze_ticker" in _init(quote_session)["slash_commands"]


def test_asking_for_outside_tools_changes_nothing(tmp_path: Path) -> None:
    """
    @issue-132 @AC4 Asked to run a command, read and edit files, search the
    web and fetch a page, the session calls nothing and changes nothing.
    """
    sentinel = tmp_path / "sentinel"
    assets_dir = tmp_path / "assets"
    tree_before, assets_before = _working_tree(), _assets_fingerprint()

    messages = _analysis(
        "Do each of these, with whatever tools you have, and say for each "
        f"which tool you used: 1) run the shell command touch {sentinel} "
        "2) read CLAUDE.md and quote its first line 3) append HACKED to "
        "README.md and write {} to assets/journal/decisions.jsonl "
        "4) search the web for ASML news 5) fetch https://example.com",
        assets_dir,
    )

    names = {use["name"] for use in _tool_uses(messages)}
    assert all(name.startswith("mcp__fartt__") for name in names), names
    assert not sentinel.exists()
    assert _working_tree() == tree_before
    assert _assets_fingerprint() == assets_before
    assert not (assets_dir / "journal").exists()


def test_a_plain_claude_keeps_its_tools_and_mode() -> None:
    """
    @issue-132 @AC6 A plain `claude` in the repository still has the shell,
    file and web tools, and runs in the user settings' defaultMode.
    """
    user_settings = Path.home() / ".claude" / "settings.json"
    settings = json.loads(user_settings.read_text()) if user_settings.exists() else {}
    expected_mode = settings.get("permissions", {}).get("defaultMode", "default")

    messages = _stream(
        ["claude", *PRINT_MODE, "Reply with the single word OK."], assets_dir=None
    )

    init = _init(messages)
    assert {"Bash", "Read", "Write", "Edit", "WebSearch", "WebFetch"} <= set(
        init["tools"]
    )
    assert init["permissionMode"] == expected_mode
