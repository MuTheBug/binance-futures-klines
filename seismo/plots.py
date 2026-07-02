"""Charts for the SEISMO report (light-mode PNGs).

Follows the dataviz method: single-hue series, recessive hairline grid,
one axis per panel, direct labels, ink for text (never series color).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"       # categorical slot 1
BLUE_200 = "#9ec5f4"   # sequential step for the drawdown fill
AQUA = "#1baf7a"       # categorical slot 2

IS_END = pd.Timestamp("2024-07-01", tz="UTC")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def equity_chart(port: pd.Series, out_png: str, title: str):
    eq = (1 + port).cumprod()
    dd = eq / eq.cummax() - 1.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.4), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)

    ax1.plot(eq.index, eq.values, color=BLUE, linewidth=2)
    ax1.set_yscale("log")
    _style(ax1)
    ax1.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax1.set_ylabel("equity (log, start = 1.0)", color=INK2, fontsize=10)
    yt = [0.5, 1, 2, 4, 8, 16, 32, 64]
    yt = [v for v in yt if eq.min() * 0.8 <= v <= eq.max() * 1.3]
    ax1.set_yticks(yt)
    ax1.set_yticklabels([f"{v:g}x" for v in yt])
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.axvline(IS_END, color=BASE, linewidth=1, linestyle=(0, (4, 3)))
    ymax = ax1.get_ylim()[1]
    ax1.text(IS_END, ymax, "  out-of-sample →", color=INK2, fontsize=9,
             va="top", ha="left")
    ax1.text(IS_END, ymax, "← in-sample (tuned)  ", color=INK2,
             fontsize=9, va="top", ha="right")
    ax1.text(eq.index[-1], eq.iloc[-1], f"  {eq.iloc[-1]:.1f}x",
             color=INK, fontsize=10, va="center")

    ax2.fill_between(dd.index, dd.values * 100, 0, color=BLUE_200)
    ax2.plot(dd.index, dd.values * 100, color=BLUE, linewidth=1)
    _style(ax2)
    ax2.set_ylabel("drawdown %", color=INK2, fontsize=10)
    ax2.axvline(IS_END, color=BASE, linewidth=1, linestyle=(0, (4, 3)))
    tmin = dd.idxmin()
    ax2.text(tmin, dd.min() * 100, f" {dd.min() * 100:.0f}%", color=INK2,
             fontsize=9, va="bottom")

    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def event_hist(events: pd.DataFrame, out_png: str):
    ev = events.copy()
    ev["event"] = pd.to_datetime(ev["event"], utc=True)
    is_ev = ev.loc[ev["event"] < IS_END, "levered_ret"] * 100
    oos_ev = ev.loc[ev["event"] >= IS_END, "levered_ret"] * 100

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    lo = min(is_ev.min(), oos_ev.min())
    hi = max(is_ev.max(), oos_ev.max())
    bins = np.linspace(lo, hi, 34)
    ax.hist(is_ev, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1,
            label=f"in-sample (n={len(is_ev)})")
    ax.hist(oos_ev, bins=bins, color=AQUA, edgecolor=SURFACE, linewidth=1,
            label=f"out-of-sample (n={len(oos_ev)})",
            histtype="bar", rwidth=0.55)
    _style(ax)
    ax.set_title("Per-event net return, levered (%)", color=INK,
                 fontsize=12, loc="left", pad=10)
    ax.axvline(0, color=BASE, linewidth=1)
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_xlabel("event return %", color=INK2, fontsize=10)
    ax.set_ylabel("events", color=INK2, fontsize=10)
    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def gated_vs_ungated_oos(port_gated_oos, port_raw_oos, out_png: str):
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for s, col, lab in [(port_gated_oos, BLUE, "with heartbeat gate"),
                        (port_raw_oos, AQUA, "ungated L1")]:
        eq = (1 + s).cumprod()
        ax.plot(eq.index, eq.values, color=col, linewidth=2, label=lab)
        ax.text(eq.index[-1], eq.iloc[-1], f"  {lab}: {eq.iloc[-1]:.2f}x",
                color=INK2, fontsize=9, va="center")
    _style(ax)
    ax.set_title("Out-of-sample equity (2024-07 →): the anomaly decayed; "
                 "the gate limits the damage", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_ylabel("equity (start = 1.0)", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    xmax = max(port_gated_oos.index[-1], port_raw_oos.index[-1])
    ax.set_xlim(right=xmax + pd.Timedelta(days=200))
    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def heartbeat_chart(events: pd.DataFrame, out_png: str, window: int = 20):
    ev = events.copy()
    ev["event"] = pd.to_datetime(ev["event"], utc=True)
    trail = ev["levered_ret"].rolling(window).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    # shade the periods where the gate stands aside
    off = ~ev["live"].values
    start = None
    for k in range(len(ev)):
        if off[k] and start is None:
            start = ev["event"].iloc[k]
        if (not off[k] or k == len(ev) - 1) and start is not None:
            ax.axvspan(start, ev["event"].iloc[k], color=GRID, alpha=0.5,
                       linewidth=0)
            start = None
    ax.plot(ev["event"], trail.values, color=BLUE, linewidth=2)
    ax.axhline(0, color=BASE, linewidth=1)
    _style(ax)
    ax.set_title(f"Anomaly heartbeat: trailing {window}-event mean paper return "
                 "(gray = capital withdrawn)", color=INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_ylabel("trailing mean, % per event", color=INK2, fontsize=10)
    ax.axvline(IS_END, color=BASE, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(IS_END, ax.get_ylim()[1], "  out-of-sample →", color=INK2,
            fontsize=9, va="top")
    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    port = pd.read_csv("seismo/out/port_full.csv", index_col=0)
    port.index = pd.to_datetime(port.index, utc=True)
    port = port.iloc[:, 0]
    equity_chart(port, "seismo/out/equity_full.png",
                 "SEISMO (gated, vol-targeted leverage ≤3x) — equity, 10 bps/side costs")
    events = pd.read_csv("seismo/out/events_full.csv")
    event_hist(events, "seismo/out/event_hist.png")
    heartbeat_chart(events, "seismo/out/heartbeat.png")
    raw_oos = pd.read_csv("seismo/out/port_raw_oos.csv", index_col=0)
    raw_oos.index = pd.to_datetime(raw_oos.index, utc=True)
    raw_oos = raw_oos.iloc[:, 0]
    gated_vs_ungated_oos(port.loc[IS_END:], raw_oos, "seismo/out/gated_vs_ungated_oos.png")
    print("charts written")
