#!/usr/bin/env python3
"""
evaluate.py - benchmark Node13's detection method on real ESA-ADB telemetry.

What it does (one channel at a time)
------------------------------------
1. Loads the channel (a pickled pandas DataFrame) and aggregates it into
   fixed time bins (default 5 min).  Each bin becomes a 6-number "tick":
   mean, std, min, max, slope (last-first), mean |diff|.
2. Splits by DATE:  train < val_start <= validation < test_start <= test.
   Training uses only bins that overlap NO labelled event on this channel.
3. Fits, using the SAME settings and code as Node13 (anomaly_detector.py):
   StandardScaler + IsolationForest + LSTMAutoencoder.
4. Scores validation and test bins with five detectors:
     - z-score baseline        (trivial baseline: how far min/max stray from normal)
     - IsolationForest alone
     - LSTM reconstruction error (level)
     - LSTM integral term      (exactly as Node13 computes it)
     - Node13 ensemble P(F)    (alpha*IF + beta*integral, exactly as Node13)
5. Picks each detector's alarm threshold on the VALIDATION period only
   (99.5th percentile of its score on nominal bins), then reports on TEST:
   ROC-AUC, PR-AUC, bin-level P/R/F1, event recall, detection delay and
   false-alarm episodes per day.  Also reports Node13's stock rules
   (P >= 0.50 anomaly flag and P >= 0.75 FR-3).

Labels
------
  Category == "Anomaly"  -> positive.
  Any other category (rare nominal events, communication gaps, ...) -> those bins
  are IGNORED in test metrics (neither positive nor negative) and are kept out
  of training.

Usage (Windows)
---------------
  python evaluate.py --channel 44 --detector path\\to\\anomaly_detector.py ^
      --test-end 2011-01-01
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["mean", "std", "min", "max", "slope", "mad_diff"]
MIN_NS = 60 * 10**9


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_detector_module(path: str):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("node13_anomaly_detector", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses need this
    spec.loader.exec_module(mod)
    return mod


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
    """True for bins [t, t+bin) that overlap any [start-margin, end+margin]."""
    m = np.zeros(len(times_ns), dtype=bool)
    for s, e in zip(starts_ns, ends_ns):
        i0 = np.searchsorted(times_ns, s - margin_ns - bin_ns, side="right")
        i1 = np.searchsorted(times_ns, e + margin_ns, side="right")
        if i1 > i0:
            m[i0:i1] = True
    return m


# --------------------------------------------------------------------------- #
# Models (Node13 code and settings)
# --------------------------------------------------------------------------- #
def spread_chunks(X: np.ndarray, n_chunks: int = 20, chunk: int = 100) -> np.ndarray:
    if len(X) <= n_chunks * chunk:
        return X
    starts = np.linspace(0, len(X) - chunk, n_chunks).astype(int)
    return np.concatenate([X[s:s + chunk] for s in starts])


def fit_models(mod, Xtr: np.ndarray, epochs: int, seed: int):
    rng = np.random.default_rng(seed)
    sub = Xtr if len(Xtr) <= 100_000 else Xtr[rng.choice(len(Xtr), 100_000, replace=False)]
    scaler = StandardScaler().fit(sub)
    if_model = IsolationForest(
        n_estimators=mod.IF_N_ESTIMATORS,
        contamination=mod.IF_CONTAMINATION,
        max_samples=mod.IF_MAX_SAMPLES,
        random_state=mod.IF_RANDOM_STATE,
        n_jobs=-1,
    ).fit(scaler.transform(sub))

    lstm = mod.LSTMAutoencoder(input_dim=Xtr.shape[1], seed=seed)
    np.random.seed(seed)                  # Node13's fit() uses unseeded np.random
    t0 = time.time()
    lstm.fit(spread_chunks(scaler.transform(Xtr)), epochs=epochs)
    print(f"  fitted models in {time.time() - t0:.0f}s (LSTM baseline MSE p90 = {lstm._baseline_mse:.4g})")
    return scaler, if_model, lstm


def score_stream(mod, scaler, if_model, lstm, X: np.ndarray, label: str) -> dict:
    """Replicates AnomalyDetector.score_tick() over a stream of ticks."""
    Xs = scaler.transform(X)
    n, k = len(Xs), mod.LSTM_WINDOW_K
    raw_if = if_model.decision_function(Xs)
    if_score = np.clip(0.5 - raw_if, 0.0, 1.0)

    lstm_err = np.zeros(n)
    lstm_int = np.zeros(n)
    hist: collections.deque = collections.deque(maxlen=k)
    t0 = time.time()
    for t in range(n):
        if t >= k - 1:
            d = lstm.normalised_error(Xs[t - k + 1:t + 1])
            lstm_err[t] = d
            hist.append(d)
            lstm_int[t] = mod._lstm_integral(hist)
        else:
            lstm_int[t] = if_score[t]    # Node13 warm-up behaviour
        if t and t % 50_000 == 0:
            print(f"    {label}: scored {t:,}/{n:,} ticks ({time.time() - t0:.0f}s)")
    p = np.clip(mod.ALPHA * if_score + mod.BETA * lstm_int, 0.0, 1.0)
    return {"if": -raw_if, "if_score": if_score, "lstm_err": lstm_err,
            "lstm_int": lstm_int, "p": p}


def zscore_scorer(Xtr: np.ndarray):
    med = np.median(Xtr[:, 0])
    mad = max(1.4826 * np.median(np.abs(Xtr[:, 0] - med)), 1e-9)
    return lambda X: np.maximum(np.abs(X[:, 2] - med), np.abs(X[:, 3] - med)) / mad


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def operating_point(alarm, valid, pos, pos_margin, times_ns, bin_ns, ev_s, ev_e, test_days):
    a = alarm & valid
    tp = int((a & pos).sum())
    fp = int((a & ~pos).sum())
    fn = int((~a & pos & valid).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float("nan")

    alarm_t = times_ns[a]
    detected, delays = 0, []
    for s, e in zip(ev_s, ev_e):
        i0 = np.searchsorted(alarm_t, s - bin_ns, side="right")
        i1 = np.searchsorted(alarm_t, e, side="right")
        if i1 > i0:
            detected += 1
            delays.append(max(0, alarm_t[i0] - s) / MIN_NS)

    fa_t = times_ns[a & ~pos_margin]
    episodes = 0 if len(fa_t) == 0 else 1 + int((np.diff(fa_t) > 60 * MIN_NS).sum())
    return {
        "alarm_bins": int(a.sum()),
        "precision": prec, "recall": rec, "f1": f1,
        "events": len(ev_s), "events_caught": detected,
        "event_recall": detected / len(ev_s) if len(ev_s) else float("nan"),
        "median_delay_min": float(np.median(delays)) if delays else float("nan"),
        "false_alarms_per_day": episodes / test_days if test_days > 0 else float("nan"),
    }


def fmt(v, nd=3):
    return "  n/a" if v != v else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--data", default=os.path.join("data", "ESA-Mission1"))
    ap.add_argument("--pickle", default=None, help=r"default: data\work\channel_N")
    ap.add_argument("--detector", required=True, help="path to Node13 anomaly_detector.py")
    ap.add_argument("--bin-min", type=int, default=5)
    ap.add_argument("--val-start", default="2009-01-01")
    ap.add_argument("--test-start", default="2010-01-01")
    ap.add_argument("--test-end", default=None, help="e.g. 2011-01-01 for a quick first run")
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--margin-min", type=int, default=60)
    ap.add_argument("--lstm-epochs", type=int, default=None, help="default: Node13's LSTM_EPOCHS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    mod = load_detector_module(args.detector)
    epochs = args.lstm_epochs or mod.LSTM_EPOCHS
    ch = f"channel_{args.channel}"
    pkl = args.pickle or os.path.join("data", "work", ch)
    bin_ns = args.bin_min * MIN_NS
    margin_ns = args.margin_min * MIN_NS

    print(f"Loading {pkl} ...")
    df = pd.read_pickle(pkl)
    series = df.iloc[:, 0]
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    val_start, test_start = pd.Timestamp(args.val_start), pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end) if args.test_end else series.index[-1] + pd.Timedelta(days=1)
    series = series[series.index < test_end]
    print(f"  {len(series):,} raw samples, {series.index[0]} -> {series.index[-1]}")

    bins = build_bins(series, args.bin_min)
    del df, series
    times_ns = to_ns(bins.index)
    X_all = bins.values
    print(f"  {len(bins):,} bins of {args.bin_min} min")

    # ---- labels ----------------------------------------------------------- #
    lab = pd.read_csv(os.path.join(args.data, "labels.csv"))
    typ = pd.read_csv(os.path.join(args.data, "anomaly_types.csv"))[["ID", "Category"]]
    lab = lab[lab.Channel == ch].merge(typ, on="ID", how="left")
    lab["Start"] = pd.to_datetime(lab.StartTime, utc=True).dt.tz_localize(None)
    lab["End"] = pd.to_datetime(lab.EndTime, utc=True).dt.tz_localize(None)
    print(f"  labelled rows on {ch}: {len(lab)}  by category: {lab.Category.value_counts().to_dict()}")
    is_anom = lab.Category == "Anomaly"
    S, E = to_ns(lab.Start), to_ns(lab.End)

    any_ev = interval_mask(times_ns, bin_ns, S, E, margin_ns)
    anom_ev = interval_mask(times_ns, bin_ns, S[is_anom], E[is_anom], 0)
    anom_ev_m = interval_mask(times_ns, bin_ns, S[is_anom], E[is_anom], margin_ns)
    other_ev = interval_mask(times_ns, bin_ns, S[~is_anom], E[~is_anom], 0)
    ignore = other_ev & ~anom_ev

    t_idx = bins.index
    tr = (t_idx < val_start)
    va = (t_idx >= val_start) & (t_idx < test_start)
    te = (t_idx >= test_start)
    train_mask = tr & ~any_ev
    val_nom = va & ~any_ev
    print(f"  train bins (nominal): {train_mask.sum():,} | validation: {va.sum():,} "
          f"({val_nom.sum():,} nominal) | test: {te.sum():,} "
          f"({(te & anom_ev & ~ignore).sum():,} anomalous, {(te & ignore).sum():,} ignored)")
    if train_mask.sum() < 2000 or val_nom.sum() < 500:
        sys.exit("Not enough nominal training/validation bins - adjust --val-start / --test-start.")

    # ---- fit + score ------------------------------------------------------ #
    print("Fitting Node13 models on nominal training bins ...")
    scaler, if_model, lstm = fit_models(mod, X_all[train_mask], epochs, args.seed)
    z_fn = zscore_scorer(X_all[train_mask])

    print("Scoring validation period ...")
    sv = score_stream(mod, scaler, if_model, lstm, X_all[va], "val")
    sv["z"] = z_fn(X_all[va])
    print("Scoring test period ...")
    st = score_stream(mod, scaler, if_model, lstm, X_all[te], "test")
    st["z"] = z_fn(X_all[te])

    # ---- evaluate --------------------------------------------------------- #
    t_te = times_ns[te]
    valid = ~ignore[te]
    pos = anom_ev[te]
    pos_m = anom_ev_m[te]
    test_days = valid.sum() * bin_ns / (86400 * 10**9)
    in_test = is_anom.values & (lab.Start >= test_start).values & (lab.Start < test_end).values
    ev_s, ev_e = S[in_test], E[in_test]
    va_nom_local = val_nom[va]

    detectors = [
        ("z-score baseline", "z"),
        ("IsolationForest alone", "if"),
        ("LSTM recon error (level)", "lstm_err"),
        ("LSTM integral term (Node13)", "lstm_int"),
        ("Node13 ensemble P(F)", "p"),
    ]
    rows = []
    print(f"\nTEST period: {valid.sum():,} scored bins, {test_days:.1f} days, "
          f"{len(ev_s)} labelled anomaly events, {pos[valid].sum():,} positive bins\n")
    for name, key in detectors:
        s = st[key]
        auc = roc_auc_score(pos[valid], s[valid]) if 0 < pos[valid].sum() < valid.sum() else float("nan")
        ap_ = average_precision_score(pos[valid], s[valid]) if pos[valid].sum() else float("nan")
        thr = float(np.quantile(sv[key][va_nom_local], args.quantile))
        r = operating_point(s >= thr, valid, pos, pos_m, t_te, bin_ns, ev_s, ev_e, test_days)
        rows.append({"detector": name, "roc_auc": auc, "pr_auc": ap_, "threshold": thr, **r})
    for name, thr in [("Node13 stock rule P>=0.50", 0.50), ("Node13 stock rule P>=0.75 (FR-3)", 0.75)]:
        r = operating_point(st["p"] >= thr, valid, pos, pos_m, t_te, bin_ns, ev_s, ev_e, test_days)
        rows.append({"detector": name, "roc_auc": float("nan"), "pr_auc": float("nan"), "threshold": thr, **r})

    res = pd.DataFrame(rows)
    base_rate = pos[valid].mean() if valid.sum() else float("nan")
    print(f"(PR-AUC of a random detector = base rate = {base_rate:.4f})")
    cols = ["roc_auc", "pr_auc", "precision", "recall", "f1", "event_recall", "median_delay_min", "false_alarms_per_day"]
    heads = ["ROC-AUC", "PR-AUC", "prec", "recall", "F1", "ev.recall", "delay_min", "FA/day"]
    print(f"{'detector':34s}" + "".join(f"{h:>10s}" for h in heads))
    for _, r in res.iterrows():
        print(f"{r.detector:34s}" + "".join(f"{fmt(float(r[c])):>10s}" for c in cols))
    print("\nevents caught (of {}): ".format(len(ev_s)) +
          "; ".join(f"{r.detector}: {int(r.events_caught)}" for _, r in res.iterrows()))

    # ---- Node13 diagnostics: how does the ensemble behave on NOMINAL data? --- #
    nom_te = te & ~any_ev
    nom_local = nom_te[te]
    print("\nNode13 ensemble on NOMINAL bins (median / share above 0.50 / share above 0.75):")
    for lbl, sc, m in [("validation", sv, va_nom_local), ("test", st, nom_local)]:
        if m.sum():
            print(f"  {lbl:10s} IF score {np.median(sc['if_score'][m]):.3f} | "
                  f"integral {np.median(sc['lstm_int'][m]):.3f} | "
                  f"P(F) {np.median(sc['p'][m]):.3f} | "
                  f">=0.50: {(sc['p'][m] >= 0.5).mean():.1%} | >=0.75: {(sc['p'][m] >= 0.75).mean():.1%}")

    out = args.out or f"results_{ch}.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
