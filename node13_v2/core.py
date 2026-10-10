"""
node13_v2/core.py - Node13 detector v2: calibrated, explainable telemetry triage.

Design (driven by the v1 benchmark on ESA-ADB):
  * Each channel is aggregated into fixed time bins (default 5 min), 6 features per bin.
  * Three cheap detectors, each scored per bin:
        z_level   static robust z-score of the bin's min/max vs the channel's normal level
        iforest   Isolation Forest on the 6 bin features (same config as Node13 v1)
        roll_z    robust z of the bin's mean/std vs the trailing 24 h (catches drift/level shifts)
  * CALIBRATION: every detector's raw score is converted to a "surprise" =
        -log10(share of NORMAL validation bins at least this extreme).
    Surprise 3 means: bins this extreme occur ~1 in 1000 in normal data.
  * FUSION: fused surprise = max over detectors (an alert fires if ANY detector is surprised).
  * ALERT THRESHOLD: chosen on validation data so that false-alarm EPISODES stay under a
    budget per week.  Nothing is tuned on the test period.
  * EXPLANATION: each alert says which detector fired and why, plus how often a bin that
    extreme occurs in normal data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ["mean", "std", "min", "max", "slope", "mad_diff"]
DETECTORS = ["z_level", "iforest", "roll_z"]
MIN_NS = 60 * 10**9
HOUR_NS = 60 * MIN_NS
WEEK_NS = 7 * 24 * HOUR_NS


# --------------------------------------------------------------------------- #
# Binning and interval helpers
# --------------------------------------------------------------------------- #
def to_ns(ts) -> np.ndarray:
    return pd.DatetimeIndex(ts).values.astype("datetime64[ns]").astype("int64")


def build_bins(series: pd.Series, bin_min: int) -> pd.DataFrame:
    rule = f"{bin_min}min"
    g = series.resample(rule).agg(["mean", "std", "min", "max", "first", "last", "count"])
    d = series.diff().abs().resample(rule).mean().rename("mad_diff")
    g = g.join(d)
    g = g[g["count"] >= 1].copy()
    g["slope"] = g["last"] - g["first"]
    g["std"] = g["std"].fillna(0.0)
    g["mad_diff"] = g["mad_diff"].fillna(0.0)
    return g[FEATURES].astype("float64")


def interval_mask(times_ns, bin_ns, starts_ns, ends_ns, margin_ns=0) -> np.ndarray:
    """True for bins [t, t+bin) overlapping any [start-margin, end+margin]."""
    m = np.zeros(len(times_ns), dtype=bool)
    for s, e in zip(starts_ns, ends_ns):
        i0 = np.searchsorted(times_ns, s - margin_ns - bin_ns, side="right")
        i1 = np.searchsorted(times_ns, e + margin_ns, side="right")
        if i1 > i0:
            m[i0:i1] = True
    return m


def episodes(times_ns: np.ndarray, gap_ns: int = HOUR_NS) -> Tuple[np.ndarray, np.ndarray]:
    """Group sorted alarm times into episodes; returns (start_idx, end_idx) arrays."""
    if len(times_ns) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    br = np.where(np.diff(times_ns) > gap_ns)[0]
    return np.r_[0, br + 1], np.r_[br, len(times_ns) - 1]


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def _robust(x: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(x))
    return med, max(1.4826 * float(np.median(np.abs(x - med))), 1e-12)


def rolling_robust_z(s: pd.Series, window: int, floor: float) -> np.ndarray:
    """|x - trailing median| / trailing robust SD, using only PAST bins."""
    mp = max(window // 3, 10)
    med = s.rolling(window, min_periods=mp).median().shift(1)
    dev = s - med
    mad = 1.4826 * dev.abs().rolling(window, min_periods=mp).median().shift(1)
    z = dev.abs() / mad.clip(lower=floor)
    return z.fillna(0.0).to_numpy()


@dataclass
class ChannelModel:
    scaler: StandardScaler
    iforest: IsolationForest
    z_med: float
    z_scale: float
    floor_mean: float
    floor_std: float
    window_bins: int

    @classmethod
    def fit(cls, X_train: np.ndarray, window_bins: int, seed: int = 0,
            n_estimators: int = 128) -> "ChannelModel":
        rng = np.random.default_rng(seed)
        sub = X_train if len(X_train) <= 100_000 else X_train[rng.choice(len(X_train), 100_000, replace=False)]
        scaler = StandardScaler().fit(sub)
        iforest = IsolationForest(n_estimators=n_estimators, random_state=42, n_jobs=-1).fit(scaler.transform(sub))
        z_med, z_scale = _robust(X_train[:, FEATURES.index("mean")])
        floor_mean = 0.1 * z_scale
        floor_std = 0.1 * _robust(X_train[:, FEATURES.index("std")])[1]
        return cls(scaler, iforest, z_med, z_scale, floor_mean, floor_std, window_bins)

    def score(self, bins: pd.DataFrame) -> Dict[str, np.ndarray]:
        X = bins[FEATURES].to_numpy()
        i_min, i_max = FEATURES.index("min"), FEATURES.index("max")
        z_level = np.maximum(np.abs(X[:, i_min] - self.z_med), np.abs(X[:, i_max] - self.z_med)) / self.z_scale
        iforest = -self.iforest.decision_function(self.scaler.transform(X))
        rz_mean = rolling_robust_z(bins["mean"], self.window_bins, self.floor_mean)
        rz_std = rolling_robust_z(bins["std"], self.window_bins, self.floor_std)
        # Telemetry is quantised, so many bins have IDENTICAL scores.  Ties make the calibration
        # coarse (the threshold can jump from "thousands of alarms" to "none"), so break ties with
        # a negligible, seeded jitter.
        rng = np.random.default_rng(12345)
        def tb(x):
            return x + 1e-7 * (np.std(x) + 1e-12) * rng.standard_normal(len(x))
        return {"z_level": tb(z_level), "iforest": tb(iforest), "roll_z": tb(np.maximum(rz_mean, rz_std)),
                "rz_mean": rz_mean, "rz_std": rz_std}


# --------------------------------------------------------------------------- #
# Calibration, fusion, thresholding
# --------------------------------------------------------------------------- #
class Calibrator:
    """raw score -> surprise = -log10(share of NORMAL bins at least this extreme)."""

    def __init__(self, nominal_scores: np.ndarray):
        self.s = np.sort(np.asarray(nominal_scores, dtype=float))
        self.n = len(self.s)
        self.cap = float(np.log10(self.n + 1))
        smax = self.s[-1]
        q99 = self.s[int(0.99 * (self.n - 1))]
        self.smax, self.spread = smax, max(smax - q99, 1e-12)

    def surprise(self, x: np.ndarray) -> np.ndarray:
        ge = self.n - np.searchsorted(self.s, x, side="left")
        base = -np.log10((1.0 + ge) / (self.n + 1.0))
        extra = np.log10(1.0 + np.maximum(x - self.smax, 0.0) / self.spread)   # keeps ranking beyond range
        return base + extra


def fuse(surprises: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.vstack([surprises[d] for d in DETECTORS])
    return stack.max(axis=0), stack.argmax(axis=0)


def false_alarm_rate(surprise, tau, times_ns, nominal_mask, bin_ns) -> float:
    """False-alarm EPISODES per week on nominal bins at threshold tau."""
    t = times_ns[nominal_mask & (surprise >= tau)]
    n_ep = len(episodes(t)[0])
    weeks = nominal_mask.sum() * bin_ns / WEEK_NS
    return n_ep / weeks if weeks > 0 else float("nan")


def choose_threshold(surprise, times_ns, nominal_mask, bin_ns, fa_per_week: float, n_cand: int = 601) -> float:
    """Smallest threshold such that it AND every higher threshold keep nominal false-alarm
    episodes/week <= budget.  (Episode counts are not strictly monotone in the threshold because
    lower thresholds can merge episodes, so we scan the grid instead of bisecting.)"""
    cand = np.unique(np.quantile(surprise[nominal_mask], np.linspace(0.90, 1.0, n_cand)))
    fa = np.array([false_alarm_rate(surprise, c, times_ns, nominal_mask, bin_ns) for c in cand])
    ok = fa <= fa_per_week
    suffix_ok = np.logical_and.accumulate(ok[::-1])[::-1]
    if not suffix_ok.any():
        return float(cand[-1]) + 1e-9
    return float(cand[int(np.argmax(suffix_ok))])


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
def explain(i: int, det: str, bins: pd.DataFrame, model: ChannelModel, raw: Dict[str, np.ndarray]) -> str:
    row = bins.iloc[i]
    if det == "z_level":
        far = max(abs(row["min"] - model.z_med), abs(row["max"] - model.z_med))
        side = "high" if abs(row["max"] - model.z_med) >= abs(row["min"] - model.z_med) else "low"
        return (f"level out of normal range: bin {side} value {row['max'] if side == 'high' else row['min']:.4g} is "
                f"{raw['z_level'][i]:.1f} robust SDs from the normal level ({model.z_med:.4g})")
    if det == "roll_z":
        feat = "mean" if raw["rz_mean"][i] >= raw["rz_std"][i] else "std"
        return (f"sudden change vs the previous 24 h: bin {feat} = {row[feat]:.4g}, "
                f"{max(raw['rz_mean'][i], raw['rz_std'][i]):.1f} robust SDs from its trailing 24 h behaviour")
    zs = model.scaler.transform(row[FEATURES].to_numpy().reshape(1, -1))[0]
    k = int(np.argmax(np.abs(zs)))
    return (f"unusual combination of features (isolation forest); most unusual: {FEATURES[k]} = "
            f"{row[FEATURES[k]]:.4g} ({zs[k]:+.1f} SD vs normal)")


def build_alerts(channel: str, bins: pd.DataFrame, model: ChannelModel, raw: Dict[str, np.ndarray],
                 sur: Dict[str, np.ndarray], fused: np.ndarray, which: np.ndarray, tau: float,
                 valid: np.ndarray, times_ns: np.ndarray, bins_per_day: float, cap: float) -> pd.DataFrame:
    alarm_idx = np.where(valid & (fused >= tau))[0]
    s_i, e_i = episodes(times_ns[alarm_idx])
    rows: List[dict] = []
    for a, b in zip(s_i, e_i):
        idx = alarm_idx[a:b + 1]
        pk = int(idx[np.argmax(fused[idx])])
        det = DETECTORS[int(which[pk])]
        beyond = fused[pk] >= cap
        rate = 10 ** (-min(fused[pk], cap)) * bins_per_day
        rows.append({
            "channel": channel,
            "start": bins.index[idx[0]], "end": bins.index[idx[-1]] + pd.Timedelta(minutes=0),
            "peak_time": bins.index[pk], "bins": len(idx),
            "peak_surprise": round(float(fused[pk]), 2), "detector": det,
            "nominal_freq": ("< " if beyond else "~ ") + f"{rate:.3g} bins/day in normal data",
            "reason": explain(pk, det, bins, model, raw),
        })
    return pd.DataFrame(rows)
