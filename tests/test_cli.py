from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from fartt import cli
from fartt.data import History, Listing
from fartt.forecast import RepeatLastReturn, Settings
from fartt.stock_cache import StockCache
from tests.fakes import HOUR_MS, FakeExchange, FakeProvider, candle, stock_candle

AFTER_CLOSE = datetime(2026, 10, 29, 21, 0, tzinfo=UTC)


@pytest.fixture
def fake_exchange(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeExchange:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)

    def make_exchange(exchange_id: str) -> FakeExchange:
        if exchange_id != "bitvavo":
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt")
        return exchange

    monkeypatch.setattr(cli, "CcxtExchange", make_exchange)
    monkeypatch.setattr(cli, "DEFAULT_HISTORY_START_MS", 0)
    monkeypatch.chdir(tmp_path)  # main() writes logs/cli.log relative to the cwd
    return exchange


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeProvider:
    days = [date(2026, 10, 27), date(2026, 10, 28), date(2026, 10, 29)]
    provider = FakeProvider(
        {"ASML.AS": History([stock_candle(d) for d in days], [], [])},
        {"ASML.AS": Listing("ASML.AS", "ASML Holding N.V.", "EUR", "XAMS")},
        now=AFTER_CLOSE,
    )
    monkeypatch.setattr(cli, "YFinanceProvider", lambda: provider)
    monkeypatch.setattr(
        cli,
        "StockCache",
        lambda provider, assets_dir: StockCache(
            provider, assets_dir, clock=lambda: provider.now
        ),
    )
    monkeypatch.chdir(tmp_path)  # main() writes logs/cli.log relative to the cwd
    return provider


class FakeServer:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    built: dict[str, Any] = {}

    def fake_build_server(
        cache: Any, market: str, interval: str, forecaster: Any, settings: Any
    ) -> FakeServer:
        built.update(
            cache=cache,
            market=market,
            interval=interval,
            forecaster=forecaster,
            settings=settings,
            server=FakeServer(),
        )
        return built["server"]

    monkeypatch.setattr(cli, "build_server", fake_build_server)
    return built


def test_no_command_prints_usage_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == 2
    assert "download" in capsys.readouterr().err


def test_serve_runs_the_server_for_the_configured_market(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(
        [
            "serve",
            "--assets-dir",
            str(tmp_path),
            "--market",
            "BTC/EUR",
            "--interval",
            "1h",
        ]
    )

    assert served["market"] == "BTC/EUR"
    assert served["interval"] == "1h"
    assert served["cache"].filepath == tmp_path / "BTC-EUR-1h.csv"
    assert served["forecaster"] == RepeatLastReturn()
    assert served["settings"] == Settings()
    assert "fee 0.25% per leg" in capsys.readouterr().err
    assert served["server"].ran


def test_serve_starts_when_the_exchange_is_unreachable(
    fake_exchange: FakeExchange, served: dict[str, Any], tmp_path: Path
) -> None:
    fake_exchange.reachable = False

    cli.main(["serve", "--assets-dir", str(tmp_path), "--interval", "1h"])

    assert served["server"].ran


def test_serve_rejects_an_unknown_market(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "serve",
                "--assets-dir",
                str(tmp_path),
                "--market",
                "BTC-EUR",
                "--interval",
                "1h",
            ]
        )

    assert exit_info.value.code == 2
    assert "BTC/EUR" in capsys.readouterr().err
    assert not served


def test_serve_passes_the_analysis_options(
    fake_exchange: FakeExchange, served: dict[str, Any], tmp_path: Path
) -> None:
    cli.main(
        [
            "serve",
            "--assets-dir",
            str(tmp_path),
            "--interval",
            "1h",
            "--fee",
            "0.001",
            "--slippage",
            "0",
            "--threshold",
            "0.01",
            "--hit-rate-window",
            "50",
        ]
    )

    assert served["settings"] == Settings(
        fee=0.001, slippage=0.0, threshold=0.01, hit_rate_window=50
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--fee", "-0.001"),
        ("--fee", "nan"),
        ("--slippage", "inf"),
        ("--threshold", "-0.01"),
        ("--hit-rate-window", "0"),
    ],
)
def test_serve_rejects_nonsense_analysis_options(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "--assets-dir", str(tmp_path), option, value])

    assert exit_info.value.code == 2
    assert option in capsys.readouterr().err
    assert not served


def test_serve_warns_about_a_threshold_below_the_round_trip_cost(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["serve", "--assets-dir", str(tmp_path), "--threshold", "0.001"])

    assert "below the round-trip cost" in capsys.readouterr().err
    assert served["server"].ran


def test_serve_help_gives_an_example_fraction_for_each_rate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An operator who reads "0.25" as 0.25% would set a 25% fee.
    with pytest.raises(SystemExit):
        cli.main(["serve", "--help"])

    help_text = " ".join(capsys.readouterr().out.split())
    assert "0.0025 is 0.25%" in help_text
    assert "0.001 is 0.1%" in help_text
    assert "0.007 is 0.7%" in help_text


def test_download_fills_the_ticker_cache(
    fake_provider: FakeProvider, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["download", "--assets-dir", str(tmp_path), "--ticker", "asml.as"])

    lines = (tmp_path / "stocks" / "ASML.AS" / "candles.csv").read_text().splitlines()
    assert lines[0] == "Date,Open,High,Low,Close,Volume"
    assert len(lines) == 4
    assert "3 new candles" in capsys.readouterr().err


def test_download_again_reports_up_to_date(
    fake_provider: FakeProvider, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["download", "--assets-dir", str(tmp_path), "--ticker", "ASML.AS"]
    cli.main(argv)
    cli.main(argv)

    assert "already up to date" in capsys.readouterr().err


def test_download_reports_a_rebuild(
    fake_provider: FakeProvider, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["download", "--assets-dir", str(tmp_path), "--ticker", "ASML.AS"]
    cli.main(argv)
    fake_provider.histories["ASML.AS"].candles[-1] = stock_candle(
        date(2026, 10, 29), 101.0
    )
    fake_provider.now = datetime(2026, 10, 30, 21, 0, tzinfo=UTC)

    cli.main(argv)

    assert "last candle amended" in capsys.readouterr().err


def test_download_rejects_an_unknown_ticker(
    fake_provider: FakeProvider, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--ticker", "NOSUCH"])

    assert exit_info.value.code == 2
    assert "NOSUCH" in capsys.readouterr().err


def test_download_rejects_an_unsupported_exchange(
    fake_provider: FakeProvider, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--ticker", "VOD.L"])

    assert exit_info.value.code == 2
    assert ".AS" in capsys.readouterr().err


def test_download_exits_1_when_the_provider_is_unreachable(
    fake_provider: FakeProvider, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_provider.reachable = False

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--ticker", "ASML.AS"])

    assert exit_info.value.code == 1
    error = capsys.readouterr().err
    assert "unreachable" in error
    assert "Traceback" not in error


def test_download_requires_a_ticker(
    fake_provider: FakeProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download"])

    assert exit_info.value.code == 2
    assert "--ticker" in capsys.readouterr().err
