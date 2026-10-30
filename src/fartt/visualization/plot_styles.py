import matplotlib.pyplot as plt


def apply_plot_styles() -> None:
    """
    Apply matplotlib rcParams that reproduce mplfinance's built-in
    "tradingview" style (grey axes edges, dashed grid, red titles), so
    that every matplotlib plot looks consistent with
    `visualization/candlestick_chart.py`'s charts, not just mplfinance's
    own. Call once, right after imports, before any plotting
    -- rcParams are global, so this affects every plot drawn afterwards
    in the same session.

    """
    plt.style.use(  # pyright: ignore[reportUnknownMemberType] -- pyplot.style.use is untyped upstream
        ["default", "fast"]
    )
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.edgecolor": "grey",
            "axes.titlecolor": "red",
            "figure.titlesize": "x-large",
            "figure.titleweight": "semibold",
            "figure.facecolor": "white",
            "grid.linestyle": "--",
            "axes.grid.axis": "both",
        }
    )
