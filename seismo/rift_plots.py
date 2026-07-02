"""Charts for RIFT."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seismo.plots import AQUA, BLUE, INK, INK2, SURFACE, _style, equity_chart
from seismo.xsmom import IS_END, build_daily, run_xs, signals

KW = dict(lookback=30, skip=0, n_side=10, rebal=7, use_resid=False)

panel = build_daily()
sig = signals(panel)

port, _ = run_xs(sig, w_cap=0.2, gate_window=20, **KW)
equity_chart(port, "seismo/out/rift_equity.png",
             "RIFT momentum L/S (2x gross, capped, heartbeat-gated) — "
             "equity, 10 bps/side costs")

_, cyc_is = run_xs(sig, end=IS_END, w_cap=0.2, **KW)
_, cyc_oos = run_xs(sig, start=IS_END, w_cap=0.2, **KW)

fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
lo = min(cyc_is.min(), cyc_oos.min()) * 100
hi = max(cyc_is.max(), cyc_oos.max()) * 100
bins = np.linspace(lo, hi, 36)
ax.hist(cyc_is * 100, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1,
        label=f"in-sample (n={len(cyc_is)})")
ax.hist(cyc_oos * 100, bins=bins, color=AQUA, edgecolor=SURFACE, linewidth=1,
        label=f"out-of-sample (n={len(cyc_oos)})", rwidth=0.55)
_style(ax)
ax.set_title("RIFT weekly cycle returns (%) — the OOS distribution holds",
             color=INK, fontsize=12, loc="left", pad=10)
ax.axvline(0, color="#c3c2b7", linewidth=1)
ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
ax.set_xlabel("cycle return %", color=INK2, fontsize=10)
ax.set_ylabel("cycles", color=INK2, fontsize=10)
fig.savefig("seismo/out/rift_cycles_hist.png", bbox_inches="tight",
            facecolor=SURFACE)
print("rift charts written")
