import json
import shutil
from pathlib import Path

import pytest
from mcp import Client

from fartt import kill_switch
from fartt.journal import Journal
from fartt.risk_config import RISK_FILE
from fartt.server import build_server
from fartt.server import server as server_module
from fartt.stock_cache import StockCache
from tests.fakes import RISK_CONFIG, FakeProvider
from tests.test_plan_trade import TRIPPED_VALUE, provider, trip
from tests.test_server import AFTER_CLOSE, ROOT, cache_for, journal_of

# The clock is AFTER_CLOSE, after 2026-10-29's close.
RESET_AT = AFTER_CLOSE.replace(microsecond=0)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def filled(tmp_path: Path) -> tuple[FakeProvider, StockCache]:
    """A cache holding ASML.AS, with the provider unreachable from now on."""
    fake = provider()
    cache = cache_for(fake, tmp_path)
    cache.update("ASML.AS")
    fake.reachable = False
    return fake, cache


def reset_lines(journal: Journal) -> list[dict[str, object]]:
    if not journal.path.exists():
        return []
    entries = [json.loads(line) for line in journal.path.read_text().splitlines()]
    return [entry for entry in entries if entry["kind"] == "reset"]


def test_a_tripped_switch_is_reset(tmp_path: Path) -> None:
    # @AC5
    fake, cache = filled(tmp_path)
    trip(cache)
    calls = len(fake.history_calls)

    code, text = kill_switch.reset(cache, journal_of(cache), RISK_CONFIG, "gap on ASML")

    assert code == 0
    (line,) = reset_lines(journal_of(cache))
    assert line == {
        "kind": "reset",
        "reset_at": RESET_AT.isoformat().replace("+00:00", "Z"),
        "session": "2026-10-28",
        "total_value": TRIPPED_VALUE,
        "peak": 10000.0,
        "drawdown": 0.2357,
        "tripped_on": "2026-10-28",
        "note": "gap on ASML",
    }
    assert "2026-10-28" in text
    assert "0.2357" in text
    assert len(fake.history_calls) == calls  # Cached data only


def test_a_reset_without_a_note_stores_null(tmp_path: Path) -> None:
    _, cache = filled(tmp_path)
    trip(cache)

    code, _ = kill_switch.reset(cache, journal_of(cache), RISK_CONFIG, None)

    assert code == 0
    (line,) = reset_lines(journal_of(cache))
    assert line["note"] is None


@pytest.mark.anyio
async def test_after_the_reset_get_portfolio_shows_it(tmp_path: Path) -> None:
    # @AC5
    fake, cache = filled(tmp_path)
    trip(cache)
    kill_switch.reset(cache, journal_of(cache), RISK_CONFIG, "gap on ASML")
    fake.reachable = True

    async with Client(build_server(cache, journal_of(cache), RISK_CONFIG)) as client:
        result = await client.call_tool("get_portfolio", {})

    content = result.structured_content
    assert content is not None
    switch = content["kill_switch"]
    assert switch["tripped"] is False
    assert switch["last_reset"]["note"] == "gap on ASML"
    assert switch["last_reset"]["drawdown"] == 0.2357
    assert content["totals"]["peak"] == TRIPPED_VALUE


def test_a_switch_that_isnt_tripped_is_not_reset(tmp_path: Path) -> None:
    # @AC6
    _, cache = filled(tmp_path)

    code, text = kill_switch.reset(cache, journal_of(cache), RISK_CONFIG, None)

    assert code == 1
    assert "isn't tripped" in text
    assert "drawdown of 0.0" in text
    assert reset_lines(journal_of(cache)) == []


def test_a_damaged_journal_exits_2_and_writes_nothing(tmp_path: Path) -> None:
    _, cache = filled(tmp_path)
    trip(cache)
    journal = journal_of(cache)
    with journal.path.open("ab") as file:
        file.write(b"not json\n")
    before = journal.path.read_bytes()

    code, text = kill_switch.reset(cache, journal, RISK_CONFIG, None)

    assert code == 2
    assert "line 4" in text
    assert journal.path.read_bytes() == before


def assets(tmp_path: Path) -> Path:
    shutil.copy(ROOT / "assets" / RISK_FILE, tmp_path / RISK_FILE)
    return tmp_path


def run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *argv: str) -> int:
    monkeypatch.setenv("FARTT_ASSETS_DIR", str(assets(tmp_path)))
    monkeypatch.setenv("FARTT_NOW", "2026-10-29T21:00:00+00:00")
    monkeypatch.setattr(server_module, "ENV_FILE", ROOT / "no-such-dir" / ".env")
    return kill_switch.main(list(argv))


def test_main_reads_the_assets_dir_and_exits_1_when_not_tripped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # @AC6
    code = run_main(monkeypatch, tmp_path, "reset")

    assert code == 1
    assert "isn't tripped" in capsys.readouterr().out
    assert not (tmp_path / "journal" / "decisions.jsonl").exists()


def test_main_resets_a_tripped_switch_with_the_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # @AC5
    _, cache = filled(tmp_path)
    trip(cache)

    code = run_main(monkeypatch, tmp_path, "reset", "--note", "gap on ASML")

    assert code == 0
    (line,) = reset_lines(Journal(tmp_path / "journal"))
    assert (line["note"], line["tripped_on"]) == ("gap on ASML", "2026-10-28")
    assert line["reset_at"] == "2026-10-29T21:00:00Z"


def test_a_note_over_200_characters_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, cache = filled(tmp_path)
    trip(cache)

    code = run_main(monkeypatch, tmp_path, "reset", "--note", "x" * 201)

    assert code == 2
    assert reset_lines(Journal(tmp_path / "journal")) == []


def test_main_without_a_risk_config_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FARTT_ASSETS_DIR", str(tmp_path))
    monkeypatch.setattr(server_module, "ENV_FILE", ROOT / "no-such-dir" / ".env")

    assert kill_switch.main(["reset"]) == 2
    assert RISK_FILE in capsys.readouterr().err


@pytest.mark.anyio
async def test_no_tool_resets_the_kill_switch_or_changes_the_limits(
    tmp_path: Path,
) -> None:
    # @AC7
    cache = cache_for(provider(), tmp_path)
    async with Client(build_server(cache, journal_of(cache), RISK_CONFIG)) as client:
        tools = (await client.list_tools()).tools

    names = {tool.name for tool in tools}
    assert not {n for n in names if "reset" in n or "kill" in n or "limit" in n}
    risk = next(t for t in tools if t.name == "get_risk_config")
    assert risk.annotations is not None
    assert risk.annotations.read_only_hint is True
