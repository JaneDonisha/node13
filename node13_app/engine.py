"""
node13_app/engine.py - the detection engine (the method validated by the walk-forward benchmark).

Two label-free detectors, both PAST-ONLY (so they behave identically live and in replay):
  level   a bin's min/max compared with a baseline from the previous 14 days (updated daily)
  change  a bin's mean/std compared with the previous 24 hours
Alarm threshold per detector: re-picked every `chunk_days` from the previous `window_days` of data so
that, over that window, the detector would have raised at most `budget` alert episodes per week.
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from node13_v2.core import FEATURES, MIN_NS, _robust, choose_threshold, episodes, rolling_robust_z, to_ns
from node13_v2.walkforward import daily_baseline_z

DAY_NS = 24 * 60 * MIN_NS
I_MEAN, I_STD, I_MIN, I_MAX = (FEATURES.index(c) for c in ("mean", "std", "min", "max"))
NAMES = {1: "level", 2: "change", 3: "level+change"}


def make_floors(bins: pd.DataFrame) -> Dict[str, float]:
    """Noise floors that stop flat channels producing absurd scores (from the first ~180 days)."""
    t = to_ns(bins.index)
    n0 = min(max(int(np.searchsorted(t, t[0] + 180 * DAY_NS)), 500), len(bins))
    X = bins[FEATURES].to_numpy()[:n0]
    return {"mean": 0.1 * _robust(X[:, I_MEAN])[1], "std": 0.1 * _robust(X[:, I_STD])[1]}


def compute_scores(bins: pd.DataFrame, floors: Dict[str, float], bin_min: int = 5, baseline_days: int = 14):
    t = to_ns(bins.index)
    X = bins[FEATURES].to_numpy()
    level = daily_baseline_z(t, X, baseline_days, floors["mean"])
    win = int(24 * 60 / bin_min)
    change = np.maximum(rolling_robust_z(bins["mean"], win, floors["mean"]),
                        rolling_robust_z(bins["std"], win, floors["std"]))
    return level, change


def make_context(bins: pd.DataFrame, start, end, max_points: int = 700) -> str:
    a, b = pd.Timestamp(start) - pd.Timedelta(hours=8), min(pd.Timestamp(end) + pd.Timedelta(hours=4),
                                                           pd.Timestamp(start) + pd.Timedelta(hours=72))
    w = bins.loc[a:b]
    if len(w) > max_points:
        w = w.iloc[:: int(np.ceil(len(w) / max_points))]
    return json.dumps({"t": [x.strftime("%Y-%m-%d %H:%M:%S") for x in w.index],
                       "mean": np.round(w["mean"].to_numpy(), 6).tolist(),
                       "min": np.round(w["min"].to_numpy(), 6).tolist(),
                       "max": np.round(w["max"].to_numpy(), 6).tolist()})


def _reason(bins, t_ns, pk: int, which: int, ratio: float, baseline_days: int) -> str:
    row = bins.iloc[pk]
    lo = int(np.searchsorted(t_ns, t_ns[pk] - baseline_days * DAY_NS))
    usual = float(np.median(bins["mean"].to_numpy()[lo:pk])) if pk - lo > 10 else float("nan")
    hi_dev, lo_dev = abs(row["max"] - usual), abs(row["min"] - usual)
    peak, side = (row["max"], "high") if hi_dev >= lo_dev else (row["min"], "low")
    lvl = f"Reached {peak:.4g} ({side}); usually about {usual:.4g} over the last {baseline_days} days"
    chg = f"Sudden change vs the previous 24 h (5-min mean {row['mean']:.4g}, spread {row['std']:.4g})"
    body = {1: lvl, 2: chg, 3: lvl + ". Also: " + chg[0].lower() + chg[1:]}[which]
    return body + "."


def detect_range(bins: pd.DataFrame, level: np.ndarray, change: np.ndarray, start, end, budget: float,
                 window_days: int = 90, chunk_days: int = 7, bin_min: int = 5, baseline_days: int = 14,
                 seed: int = 0) -> Dict:
    """Alert episodes for bins in [start, end). Thresholds use only data BEFORE each chunk."""
    t = to_ns(bins.index)
    n = len(bins)
    bin_ns = bin_min * MIN_NS
    rng = np.random.default_rng(seed)
    alarm = np.zeros(n, bool)
    ratio = np.zeros(n)
    which = np.zeros(n, dtype=np.int8)
    s_ns, e_ns = int(pd.Timestamp(start).value), int(pd.Timestamp(end).value)
    W, step = window_days * DAY_NS, chunk_days * DAY_NS
    ms, calibrated, skipped = s_ns, 0, 0
    while ms < e_ns:
        me = min(ms + step, e_ns)
        lo, mid, hi = (int(np.searchsorted(t, v)) for v in (ms - W, ms, me))
        if mid - lo >= 2000 and hi > mid:
            tw, allm = t[lo:mid], np.ones(mid - lo, bool)
            ztr = level[lo:mid] + 1e-7 * (level[lo:mid].std() + 1e-12) * rng.standard_normal(mid - lo)
            ctr = change[lo:mid] + 1e-7 * (change[lo:mid].std() + 1e-12) * rng.standard_normal(mid - lo)
            thr_l = choose_threshold(ztr, tw, allm, bin_ns, budget, 120)
            thr_c = choose_threshold(ctr, tw, allm, bin_ns, budget, 120)
            rl, rc = level[mid:hi] / thr_l, change[mid:hi] / thr_c
            alarm[mid:hi] = (rl >= 1) | (rc >= 1)
            ratio[mid:hi] = np.maximum(rl, rc)
            which[mid:hi] = (rl >= 1).astype(np.int8) + 2 * (rc >= 1).astype(np.int8)
            calibrated += 1
        else:
            skipped += 1
        ms = me
    out: List[dict] = []
    ai = np.where(alarm)[0]
    s_i, e_i = episodes(t[ai])
    for x, y in zip(s_i, e_i):
        idx = ai[x:y + 1]
        pk = int(idx[np.argmax(ratio[idx])])
        st, en = bins.index[idx[0]], bins.index[idx[-1]] + pd.Timedelta(minutes=bin_min)
        out.append({"start": st, "end": en, "bins": len(idx), "ratio": float(ratio[pk]),
                    "detector": NAMES[int(which[pk])], "reason": _reason(bins, t, pk, int(which[pk]), ratio[pk], baseline_days),
                    "context": make_context(bins, st, en)})
    return {"episodes": out, "calibrated_chunks": calibrated, "skipped_chunks": skipped}
