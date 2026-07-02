"""Final evaluation of RIFT (cross-sectional momentum L/S, daily).

Frozen config (IS-tuned): lookback=30d, skip=0, 10 per side, weekly
rebalance, raw momentum, inverse-vol weights with 20% per-name leg cap,
gross 2x, 10bps/side, heartbeat gate on trailing 20 paper cycles.
"""
import itertools

import numpy as np
import pandas as pd

from seismo.xsmom import IS_END, build_daily, fmt, run_xs, signals, stats

KW = dict(lookback=30, skip=0, n_side=10, rebal=7, use_resid=False)
CAP = 0.2
GATE = 20

GRID = {"lookback": [14, 30, 60], "skip": [0, 2], "n_side": [5, 10],
        "rebal": [3, 7], "use_resid": [False, True]}


def main():
    panel = build_daily()
    sig = signals(panel)

    rows = []
    for label, gate in [("RIFT capped", None), ("RIFT capped+gate", GATE)]:
        for w, s, e in [("IS", None, IS_END), ("OOS", IS_END, None),
                        ("full", None, None)]:
            port, cyc = run_xs(sig, start=s, end=e, w_cap=CAP,
                               gate_window=gate, **KW)
            m = stats(port, cyc, f"{label} | {w}")
            traded = cyc[cyc != 0]
            if len(traded):
                m["traded_winrate"] = float((traded > 0).mean())
            rows.append(m)
            if label == "RIFT capped+gate" and w == "full":
                port.to_csv("seismo/out/rift_port_full.csv")
                pd.Series(cyc).to_csv("seismo/out/rift_cycles.csv",
                                      index=False)
    main_tbl = pd.DataFrame(rows)

    # ---- walk-forward: re-tune the 48-combo grid per OOS fold --------------
    def tune(end):
        best, bs = None, -np.inf
        for vals in itertools.product(*GRID.values()):
            kw = dict(zip(GRID, vals))
            port, cyc = run_xs(sig, end=end, w_cap=CAP, **kw)
            m = stats(port, cyc, "")
            if m["cycles"] < 50 or not np.isfinite(m["sharpe"]):
                continue
            if m["sharpe"] > bs:
                best, bs = kw, m["sharpe"]
        return best

    folds = [("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-07-01"),
             ("2025-07-01", "2026-01-01"), ("2026-01-01", "2026-05-30")]
    ports, wf_params = [], []
    for fs, fe in folds:
        kw = tune(fs)
        port, _ = run_xs(sig, start=fs, end=fe, w_cap=CAP, **kw)
        ports.append(port)
        wf_params.append({"fold": f"{fs}..{fe}", **kw})
    wf = pd.concat(ports)
    wf = wf[~wf.index.duplicated()]
    wf_m = stats(wf, np.array([]), "RIFT walk-forward OOS (param-blind)")

    # ---- sensitivity --------------------------------------------------------
    cost_rows = []
    for c in (0.0005, 0.0010, 0.0015, 0.0020):
        for w, s, e in [("IS", None, IS_END), ("OOS", IS_END, None)]:
            port, cyc = run_xs(sig, start=s, end=e, w_cap=CAP,
                               cost_side=c, **KW)
            cost_rows.append(stats(port, cyc, f"{w} | {c * 1e4:.0f}bps/side"))
    lev_rows = []
    for g in (1.0, 2.0, 3.0, 4.0):
        port, cyc = run_xs(sig, w_cap=CAP, gross=g, gate_window=GATE, **KW)
        lev_rows.append(stats(port, cyc, f"full | gross {g:.0f}x (gated)"))

    main_tbl.to_csv("seismo/out/rift_main.csv", index=False)
    pd.DataFrame([wf_m]).to_csv("seismo/out/rift_walkforward.csv", index=False)
    pd.DataFrame(wf_params).to_csv("seismo/out/rift_wf_params.csv", index=False)
    pd.DataFrame(cost_rows).to_csv("seismo/out/rift_costs.csv", index=False)
    pd.DataFrame(lev_rows).to_csv("seismo/out/rift_leverage.csv", index=False)

    for name, tbl in [("MAIN", main_tbl), ("WALK-FORWARD", pd.DataFrame([wf_m])),
                      ("WF PARAMS", pd.DataFrame(wf_params)),
                      ("COSTS", pd.DataFrame(cost_rows)),
                      ("LEVERAGE (gross)", pd.DataFrame(lev_rows))]:
        print(f"\n===== {name} =====")
        print(tbl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
