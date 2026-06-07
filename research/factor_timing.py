"""Last legitimate shot at adding carry: deploy it only when it's recently working
(factor momentum). If the filter removes carry's OOS drag while keeping its IS edge,
the combined Sharpe rises without hurting OOS. Otherwise, carry stays rejected."""
from __future__ import annotations
import numpy as np, pandas as pd
import engine, strategies as st, lab, funding
import final_v3 as fv
import carry as cy

I = "1d"; TV = 0.40


def main():
    c, v, r, e = lab.load_data(I)
    e = e & ~e.columns.isin(fv.EXCLUDE)
    F = funding.load_daily_funding(c.index).reindex(columns=c.columns)
    e = e & F.notna()
    cy.FMAT = F

    trend = cy.sim_net(fv.build(c, v, e), r)
    carry = cy.sim_net(cy.carry_weights(F, r, e, N=3, K=12, smooth=10), r)

    def rep(tag, net):
        f, mi, mo = engine.metrics(net, I), engine.metrics(lab.split(net)[0], I), engine.metrics(lab.split(net)[1], I)
        print(f"  {tag:<30} FULL={f['Sharpe']:>5.2f} IS={mi['Sharpe']:>5.2f} OOS={mo['Sharpe']:>5.2f} "
              f"CAGR={f['CAGR']*100:>5.0f}% DD={f['maxDD']*100:>4.0f}%")

    rep("trend-alone", trend)
    rep("carry-alone", carry)
    print("\n static blends:")
    for wt in [0.2, 0.3]:
        rep(f"  trend + {wt} carry (static)", trend + wt * carry)

    print("\n factor-momentum timing on carry (deploy only when recently working):")
    for win in [60, 90, 120]:
        # trailing Sharpe of carry, lagged 1 day (no look-ahead)
        tsharpe = (carry.rolling(win).mean() / carry.rolling(win).std()).shift(1)
        # binary gate
        gate = (tsharpe > 0).astype(float)
        rep(f"  trend + 0.4*carry*gate(>0,{win}d)", trend + 0.4 * gate * carry)
        # continuous gate in [0,1]
        cont = tsharpe.clip(0, 1.5).fillna(0) / 1.5
        rep(f"  trend + 0.5*carry*cont({win}d)", trend + 0.5 * cont * carry)

    # also: gate by carry trailing cumulative return sign
    print("\n gate by trailing 90d carry return sign:")
    g = (carry.rolling(90).sum().shift(1) > 0).astype(float)
    rep("  trend + 0.4*carry*ret_gate", trend + 0.4 * g * carry)
    print(f"\n  (carry gate ON fraction: IS={g[g.index<=lab.IS_END].mean():.0%} "
          f"OOS={g[g.index>lab.IS_END].mean():.0%})")


if __name__ == "__main__":
    main()
