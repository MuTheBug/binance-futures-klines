"""
Loader for the downloaded alt-data (open interest, long/short positioning, taker flow,
funding) in ../altdata/. Returns aligned matrices (index = UTC day, columns = base) so the
sleeves can be built with the same machinery as price sleeves.
"""
from __future__ import annotations
import os, glob
import pandas as pd

ALT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "altdata")

METRIC_FIELDS = ["oi", "oi_value", "toptrader_acc_ls", "toptrader_pos_ls",
                 "global_acc_ls", "taker_ls"]


def _read(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date")


def have_metrics():
    return len(glob.glob(os.path.join(ALT, "*_USDT_metrics_1d.csv"))) > 0


def have_funding():
    return len(glob.glob(os.path.join(ALT, "*_USDT_funding_1d.csv"))) > 0


def load_metric(field, reindex_like=None):
    """field in METRIC_FIELDS -> DataFrame(date x base)."""
    cols = {}
    for f in glob.glob(os.path.join(ALT, "*_USDT_metrics_1d.csv")):
        base = os.path.basename(f)[:-len("_USDT_metrics_1d.csv")]
        df = _read(f)
        if field in df.columns:
            cols[base] = df[field]
    if not cols:
        return None
    out = pd.DataFrame(cols).sort_index()
    if reindex_like is not None:
        out = out.reindex(reindex_like.index).reindex(columns=reindex_like.columns)
    return out


def load_funding(field="funding_sum", reindex_like=None):
    cols = {}
    for f in glob.glob(os.path.join(ALT, "*_USDT_funding_1d.csv")):
        base = os.path.basename(f)[:-len("_USDT_funding_1d.csv")]
        df = _read(f)
        if field in df.columns:
            cols[base] = df[field]
    if not cols:
        return None
    out = pd.DataFrame(cols).sort_index()
    if reindex_like is not None:
        out = out.reindex(reindex_like.index).reindex(columns=reindex_like.columns)
    return out
