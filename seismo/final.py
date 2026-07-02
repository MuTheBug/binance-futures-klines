"""Final evaluation of the SEISMO system.

Layers:
  L1  aftershock event engine (raw, directional)     -- tuned on IS only
  L2  vol-targeted leverage (risk budget per event)  -- design constant
  L3  anomaly heartbeat gate (trailing paper track)  -- added AFTER observing
      the OOS decay of L1; window sensitivity is reported, not optimized.

The gated system always runs over full history (the paper track needs the
past); metrics are then computed on time slices.
"""
import itertools
import json

import numpy as np
import pandas as pd

from seismo.engine import Params, compute_signals, metrics, run
from seismo.panel import build_panel

IS_END = "2024-07-01"
GATE_W = 20

CHOSEN = dict(z_thr=2.5, aftershock_gap=24.0, aftershock_win=120.0,
              n_picks=5, hold_hours=36, target_risk=0.10)
CHOSEN_HEDGED = dict(z_thr=2.5, aftershock_gap=24.0, aftershock_win=96.0,
                     n_picks=3, hold_hours=36, target_risk=0.10)

GRID = {
    "z_thr": [2.0, 2.5, 3.0],
    "aftershock_gap": [24.0, 48.0],
    "aftershock_win": [72.0, 96.0, 120.0],
    "n_picks": [3, 5, 8],
    "hold_hours": [12, 24, 36],
    "target_risk": [0.06, 0.10],
}


def slice_metrics(port, tr, ev, label, start=None, end=None):
    pr = port
    if start:
        pr = pr.loc[pd.Timestamp(start, tz="UTC"):]
    if end:
        pr = pr.loc[:pd.Timestamp(end, tz="UTC")]
    ev2, tr2 = ev, tr
    if len(ev):
        m = pd.Series(True, index=ev.index)
        if start:
            m &= ev["event"] >= pd.Timestamp(start, tz="UTC")
        if end:
            m &= ev["event"] < pd.Timestamp(end, tz="UTC")
        ev2 = ev[m]
    if len(tr):
        m = pd.Series(True, index=tr.index)
        if start:
            m &= tr["event"] >= pd.Timestamp(start, tz="UTC")
        if end:
            m &= tr["event"] < pd.Timestamp(end, tz="UTC")
        tr2 = tr[m]
    return metrics(pr, tr2, ev2, label)


def main():
    panel = build_panel()
    sig = compute_signals(panel)

    p_gated = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0,
                     gate_window=GATE_W, **CHOSEN)
    p_raw = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0, **CHOSEN)
    p_hedged = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0,
                      side="hedged_overshoot", **CHOSEN_HEDGED)

    # ---- headline: gated system ---------------------------------------------
    g_port, g_tr, g_ev = run(sig, p_gated)
    main_rows = [
        slice_metrics(g_port, g_tr, g_ev, "SEISMO gated | in-sample", None, IS_END),
        slice_metrics(g_port, g_tr, g_ev, "SEISMO gated | out-of-sample", IS_END, None),
        slice_metrics(g_port, g_tr, g_ev, "SEISMO gated | full history"),
    ]

    # ---- disclosure: ungated L1 raw + hedged --------------------------------
    for label, pp in [("L1 raw ungated", p_raw), ("L1 hedged ungated", p_hedged)]:
        port, tr, ev = run(sig, pp)
        main_rows.append(slice_metrics(port, tr, ev, f"{label} | in-sample", None, IS_END))
        main_rows.append(slice_metrics(port, tr, ev, f"{label} | out-of-sample", IS_END, None))
    main_tbl = pd.DataFrame(main_rows)

    # ---- gate window sensitivity --------------------------------------------
    gate_rows = []
    for w in (10, 20, 30):
        pp = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0,
                    gate_window=w, **CHOSEN)
        port, tr, ev = run(sig, pp)
        gate_rows.append(slice_metrics(port, tr, ev, f"gate w={w} | full"))
        gate_rows.append(slice_metrics(port, tr, ev, f"gate w={w} | OOS", IS_END, None))
    gate_tbl = pd.DataFrame(gate_rows)

    # ---- ungated walk-forward over OOS (decay evidence) ---------------------
    def tune_window(end, min_events=40):
        best, bm = None, None
        for vals in itertools.product(*GRID.values()):
            kw = dict(zip(GRID, vals))
            pp = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0, **kw)
            port, tr, ev = run(sig, pp, end=end)
            m = metrics(port.loc[:pd.Timestamp(end, tz="UTC")], tr, ev, "")
            if m["events"] < min_events or not np.isfinite(m["sharpe"]):
                continue
            if bm is None or m["sharpe"] > bm["sharpe"]:
                best, bm = kw, m
        return best

    folds = [("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-07-01"),
             ("2025-07-01", "2026-01-01"), ("2026-01-01", "2026-06-08")]
    wf_ports, wf_evs, wf_params = [], [], []
    for fs, fe in folds:
        kw = tune_window(fs)
        pp = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0, **kw)
        port, tr, ev = run(sig, pp, start=fs, end=fe)
        wf_ports.append(port.loc[pd.Timestamp(fs, tz="UTC"):pd.Timestamp(fe, tz="UTC")])
        wf_evs.append(ev)
        wf_params.append({"fold": f"{fs}..{fe}", **kw})
    wf_port = pd.concat(wf_ports)
    wf_port = wf_port[~wf_port.index.duplicated()]
    wf_ev = pd.concat([e for e in wf_evs if len(e)], ignore_index=True)
    wf_m = metrics(wf_port, pd.DataFrame({"ret_net": []}), wf_ev,
                   "ungated walk-forward OOS")

    # ---- ablations on IS (ungated) ------------------------------------------
    def is_run(pp, label):
        port, tr, ev = run(sig, pp, end=IS_END)
        return metrics(port.loc[:pd.Timestamp(IS_END, tz="UTC")], tr, ev, label)

    ab_rows = [is_run(p_raw, "base L1 (IS)")]
    ab_rows.append(is_run(Params(breadth_thr=0.85, stop_frac=None, leverage=3.0,
                                 **{**CHOSEN, "aftershock_gap": None,
                                    "aftershock_win": None}),
                          "no aftershock gate"))
    ab_rows.append(is_run(Params(breadth_thr=0.85, stop_frac=None, leverage=3.0,
                                 side="short_resilient", **CHOSEN),
                          "short resilient (mirror)"))
    rnd = [is_run(Params(breadth_thr=0.85, stop_frac=None, leverage=3.0,
                         random_seed=s, **CHOSEN), f"rnd{s}")["total_return"]
           for s in range(40)]
    ab_rows.append({"label": "random picks (40 seeds)",
                    "total_return": float(np.mean(rnd)),
                    "rnd_p95": float(np.percentile(rnd, 95)),
                    "pct_seeds_below_base": float((np.array(rnd) <
                                                   ab_rows[0]["total_return"]).mean())})
    ab_tbl = pd.DataFrame(ab_rows)

    # ---- cost & leverage sensitivity on the gated system --------------------
    cost_rows, lev_rows = [], []
    for cst in (0.0005, 0.0010, 0.0015, 0.0020):
        pp = Params(breadth_thr=0.85, stop_frac=None, leverage=3.0,
                    gate_window=GATE_W, cost_side=cst, **CHOSEN)
        port, tr, ev = run(sig, pp)
        cost_rows.append(slice_metrics(port, tr, ev,
                                       f"gated full | {cst * 1e4:.0f}bps/side"))
    for cap in (1.0, 2.0, 3.0, 5.0):
        pp = Params(breadth_thr=0.85, stop_frac=None, leverage=cap,
                    gate_window=GATE_W, **CHOSEN)
        port, tr, ev = run(sig, pp)
        lev_rows.append(slice_metrics(port, tr, ev, f"gated full | cap {cap:.0f}x"))
    cost_tbl = pd.DataFrame(cost_rows)
    lev_tbl = pd.DataFrame(lev_rows)

    # ---- persist -------------------------------------------------------------
    main_tbl.to_csv("seismo/out/main_results.csv", index=False)
    gate_tbl.to_csv("seismo/out/gate_sensitivity.csv", index=False)
    pd.DataFrame([wf_m]).to_csv("seismo/out/walkforward.csv", index=False)
    pd.DataFrame(wf_params).to_csv("seismo/out/wf_params.csv", index=False)
    ab_tbl.to_csv("seismo/out/ablations.csv", index=False)
    cost_tbl.to_csv("seismo/out/cost_sensitivity.csv", index=False)
    lev_tbl.to_csv("seismo/out/leverage_table.csv", index=False)
    with open("seismo/out/chosen_params.json", "w") as fh:
        json.dump({"raw": CHOSEN, "hedged": CHOSEN_HEDGED,
                   "gate_window": GATE_W}, fh, indent=2)
    g_port.to_csv("seismo/out/port_full.csv")
    g_ev.to_csv("seismo/out/events_full.csv", index=False)
    g_tr.to_csv("seismo/out/trades_full.csv", index=False)
    # ungated OOS port for the comparison chart
    r_port, _, _ = run(sig, p_raw)
    r_port.loc[pd.Timestamp(IS_END, tz="UTC"):].to_csv("seismo/out/port_raw_oos.csv")

    for name, tbl in [("MAIN", main_tbl), ("GATE SENSITIVITY", gate_tbl),
                      ("UNGATED WALK-FORWARD", pd.DataFrame([wf_m])),
                      ("WF PARAMS", pd.DataFrame(wf_params)),
                      ("ABLATIONS (IS)", ab_tbl), ("COSTS", cost_tbl),
                      ("LEVERAGE", lev_tbl)]:
        print(f"\n===== {name} =====")
        print(tbl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
