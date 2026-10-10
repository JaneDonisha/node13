"""
node13_v2/walkforward.py - label-free, drift-tolerant detection, evaluated WALK-FORWARD.

How a live Node13 would behave month after month:
  every --step-days, for every channel:
    1. take the previous --window-days of telemetry (no labels, nothing from the future),
    2. re-pick each detector's alarm threshold so that false-alarm EPISODES in that window stay within
       --fa-per-week (scores in that window are computed exactly as they would have been live),
    3. apply them to the next --step-days of data.
Detectors:
    z_roll  level check against a baseline from the previous --baseline-days, updated DAILY (tracks drift)
    roll_z  change vs the trailing 24 h (already adaptive)
    ANY     alarm if either fires
Labels are used ONLY to score the result.

    python -m node13_v2.walkforward --channels 44
    python -m node13_v2.walkforward --channels all --cleanup
    python -m node13_v2.walkforward --channels all --cleanup --test-start 2005-01-01   (longer test)

Writes results_v3\\per_channel.csv, summary.csv, events_detail.csv and alerts_all.csv (the last is
in the format node13_v2.incidents reads:  python -m node13_v2.incidents --alerts results_v3\\alerts_all.csv)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .bench import chance_recall, ensure_pickle, event_hits, parse_channels
from .core import (FEATURES, MIN_NS, WEEK_NS, _robust, build_bins, choose_threshold, episodes,
                   interval_mask, rolling_robust_z, to_ns)

DETS = ["z_roll", "roll_z", "ANY"]
I_MIN, I_MAX, I_MEAN = FEATURES.index("min"), FEATURES.index("max"), FEATURES.index("mean")
DAY_NS = 24 * 60 * MIN_NS


def level_z(X: np.ndarray, med: float, mad: float) -> np.ndarray:
    return np.maximum(np.abs(X[:, I_MIN] - med), np.abs(X[:, I_MAX] - med)) / mad


def daily_baseline_z(times: np.ndarray, X: np.ndarray, baseline_days: int, floor: float, min_bins: int = 500) -> np.ndarray:
    """Level check of every bin against a baseline from the PREVIOUS `baseline_days`, updated daily.
    Past-only, so it can run live; short baseline -> follows slow drift without lag."""
    n = len(times)
    z = np.zeros(n)
    day = times // DAY_NS
    starts = np.r_[0, np.where(np.diff(day) != 0)[0] + 1]
    ends = np.r_[starts[1:], n]
    bw = baseline_days * DAY_NS
    for s_, e_ in zip(starts, ends):
        lo = int(np.searchsorted(times, day[s_] * DAY_NS - bw))
        if s_ - lo < min_bins:
            continue
        m = X[lo:s_, I_MEAN]
        med = float(np.median(m))
        mad = max(1.4826 * float(np.median(np.abs(m - med))), floor, 1e-12)
        z[s_:e_] = np.maximum(np.abs(X[s_:e_, I_MIN] - med), np.abs(X[s_:e_, I_MAX] - med)) / mad
    return z


def run_channel(ch: int, a, labels: pd.DataFrame, types: pd.DataFrame):
    name = f"channel_{ch}"
    pkl, extracted = ensure_pickle(ch, a.data, a.work, a.sevenzip)
    series = pd.read_pickle(pkl).iloc[:, 0]
    if a.cleanup and extracted:
        os.remove(pkl)
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    test_start = pd.Timestamp(a.test_start)
    test_end = pd.Timestamp(a.test_end) if a.test_end else series.index[-1] + pd.Timedelta(days=1)
    series = series[series.index < test_end]
    bin_ns = a.bin_min * MIN_NS
    bins = build_bins(series, a.bin_min)
    del series
    times = to_ns(bins.index)
    X = bins.values
    n = len(bins)

    lab = labels[labels.Channel == name].merge(types, on="ID", how="left").reset_index(drop=True)
    lab["Start"] = pd.to_datetime(lab.StartTime, utc=True).dt.tz_localize(None)
    lab["End"] = pd.to_datetime(lab.EndTime, utc=True).dt.tz_localize(None)
    is_anom = (lab.Category == "Anomaly").to_numpy()
    S, E = to_ns(lab.Start), to_ns(lab.End)
    margin = a.margin_min * MIN_NS
    anom_ev = interval_mask(times, bin_ns, S[is_anom], E[is_anom], 0)
    anom_ev_m = interval_mask(times, bin_ns, S[is_anom], E[is_anom], margin)
    ignore = interval_mask(times, bin_ns, S[~is_anom], E[~is_anom], 0) & ~anom_ev

    # trailing-24h change detector (adaptive by construction); noise floors from the first 180 days
    n0 = max(int(np.searchsorted(times, times[0] + 180 * DAY_NS)), 500)
    floor_m = 0.1 * _robust(X[:n0, I_MEAN])[1]
    floor_s = 0.1 * _robust(X[:n0, FEATURES.index("std")])[1]
    win = int(24 * 60 / a.bin_min)
    roll_all = np.maximum(rolling_robust_z(bins["mean"], win, floor_m), rolling_robust_z(bins["std"], win, floor_s))

    zr_all = daily_baseline_z(times, X, a.baseline_days, 0.1 * _robust(X[:n0, I_MEAN])[1])
    rng = np.random.default_rng(a.seed)
    W_ns = a.window_days * DAY_NS
    budgets = a.budgets
    norm = {(d, b): np.zeros(n, dtype=np.float32) for d in ("z_roll", "roll_z") for b in budgets}
    alarm = {(d, b): np.zeros(n, bool) for d in ("z_roll", "roll_z") for b in budgets}
    scored = np.zeros(n, bool)
    ts_ns = to_ns([test_start])[0]
    end_ns = to_ns([test_end])[0]
    step_ns = a.step_days * DAY_NS
    ms = ts_ns
    while ms < end_ns:
        me = min(ms + step_ns, end_ns)
        lo, mid, hi = (int(np.searchsorted(times, v)) for v in (ms - W_ns, ms, me))
        if mid - lo >= 2000 and hi > mid:
            allm = np.ones(mid - lo, bool)
            tw = times[lo:mid]
            ztr = zr_all[lo:mid] + 1e-7 * (zr_all[lo:mid].std() + 1e-12) * rng.standard_normal(mid - lo)
            rtr = roll_all[lo:mid] + 1e-7 * (roll_all[lo:mid].std() + 1e-12) * rng.standard_normal(mid - lo)
            zm = zr_all[mid:hi]
            for b in budgets:
                thr_z = choose_threshold(ztr, tw, allm, bin_ns, b, a.n_cand)
                thr_r = choose_threshold(rtr, tw, allm, bin_ns, b, a.n_cand)
                norm[("z_roll", b)][mid:hi] = zm / thr_z
                norm[("roll_z", b)][mid:hi] = roll_all[mid:hi] / thr_r
                alarm[("z_roll", b)][mid:hi] = zm >= thr_z
                alarm[("roll_z", b)][mid:hi] = roll_all[mid:hi] >= thr_r
            scored[mid:hi] = True
        ms = me
    for b in budgets:
        norm[("ANY", b)] = np.maximum(norm[("z_roll", b)], norm[("roll_z", b)])
        alarm[("ANY", b)] = alarm[("z_roll", b)] | alarm[("roll_z", b)]

    valid = scored & ~ignore
    if valid.sum() < 1000:
        raise RuntimeError("too little scored test data")
    t_v = times[valid]
    weeks = valid.sum() * bin_ns / WEEK_NS
    in_test = is_anom & (lab.Start >= test_start).to_numpy() & (lab.Start < test_end).to_numpy()
    ev_s, ev_e, ev_ids = S[in_test], E[in_test], lab.ID.to_numpy()[in_test]
    ev_start, ev_dur_h = lab.Start[in_test].to_numpy(), (E[in_test] - S[in_test]) / (60 * MIN_NS)
    y = anom_ev[valid]
    both = 0 < y.sum() < len(y)
    crng = np.random.default_rng(a.seed + 1)

    rows, hits, arows = [], {}, []
    for b in budgets:
        for d in DETS:
            al = alarm[(d, b)] & valid
            hit, delay = event_hits(times[al], ev_s, ev_e, bin_ns)
            hits[(d, b)] = hit
            ai = np.where(al)[0]
            s_i, e_i = episodes(times[ai])
            n_ep = len(s_i)
            true_ep = sum(bool(anom_ev_m[ai[x:y_ + 1]].any()) for x, y_ in zip(s_i, e_i))
            nv = norm[(d, b)][valid]
            rows.append({
                "channel": name, "budget": b, "detector": d,
                "roc_auc": roc_auc_score(y, nv) if both else float("nan"),
                "pr_auc": average_precision_score(y, nv) if y.sum() else float("nan"),
                "events": len(ev_s), "events_caught": int(hit.sum()),
                "chance_recall": chance_recall(al[valid], t_v, ev_s, ev_e, bin_ns, crng),
                "alert_prec": (true_ep / n_ep) if n_ep else float("nan"), "episodes": n_ep, "weeks": weeks,
                "fa_per_week_test": (n_ep - true_ep) / weeks, "time_in_alarm": float(al.sum() / valid.sum()),
                "median_delay_min": delay,
            })
        ai = np.where(alarm[("ANY", b)] & valid)[0]
        s_i, e_i = episodes(times[ai])
        for x, y_ in zip(s_i, e_i):
            idx = ai[x:y_ + 1]
            pk = int(idx[np.argmax(norm[("ANY", b)][idx])])
            which = "both" if (alarm[("z_roll", b)][pk] and alarm[("roll_z", b)][pk]) else (
                "z_roll" if alarm[("z_roll", b)][pk] else "roll_z")
            arows.append({"budget": b, "channel": name, "start": bins.index[idx[0]], "end": bins.index[idx[-1]],
                          "bins": len(idx), "peak_ratio": round(float(norm[("ANY", b)][pk]), 2), "detector": which})
    alerts = pd.DataFrame(arows)
    detail = pd.concat([pd.DataFrame({"channel": name, "budget": b, "event_id": ev_ids, "start": ev_start,
                                      "duration_h": ev_dur_h, **{f"hit_{d}": hits[(d, b)] for d in DETS}})
                        for b in budgets], ignore_index=True)
    return rows, alerts, detail, {"events": len(ev_s), "weeks": weeks, "base": float(y.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", default="44")
    ap.add_argument("--data", default=os.path.join("data", "ESA-Mission1"))
    ap.add_argument("--work", default=os.path.join("data", "work"))
    ap.add_argument("--sevenzip", default=r"C:\Program Files\7-Zip\7z.exe")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--bin-min", type=int, default=5)
    ap.add_argument("--test-start", default="2010-01-01")
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--window-days", type=int, default=90, help="trailing data used to re-pick thresholds")
    ap.add_argument("--baseline-days", type=int, default=14, help="trailing data for the daily-updated level baseline")
    ap.add_argument("--step-days", type=int, default=30, help="how often to recalibrate")
    ap.add_argument("--fa-per-week", default="2,0.5,0.2,0.05", help="comma list of per-channel false-alarm budgets (episodes/week) to sweep")
    ap.add_argument("--n-cand", type=int, default=120)
    ap.add_argument("--margin-min", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results_v3")
    a = ap.parse_args()
    a.budgets = [float(x) for x in str(a.fa_per_week).split(",")]

    os.makedirs(a.out_dir, exist_ok=True)
    labels = pd.read_csv(os.path.join(a.data, "labels.csv"))
    types = pd.read_csv(os.path.join(a.data, "anomaly_types.csv"))[["ID", "Category"]]
    chinfo = pd.read_csv(os.path.join(a.data, "channels.csv")).set_index("Channel")

    all_rows, all_alerts, all_detail = [], [], []
    for ch in parse_channels(a.channels, chinfo):
        name = f"channel_{ch}"
        if name in chinfo.index and str(chinfo.loc[name, "Categorical"]).upper() == "YES":
            print(f"{name}: categorical - skipped")
            continue
        t0 = time.time()
        try:
            rows, alerts, detail, info = run_channel(ch, a, labels, types)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
            continue
        all_rows += rows; all_alerts.append(alerts); all_detail.append(detail)
        line = "  ".join(f"b={b:g}: ANY {next(r for r in rows if r['budget'] == b and r['detector'] == 'ANY')['events_caught']}/{info['events']}"
                         f" inAlarm {next(r for r in rows if r['budget'] == b and r['detector'] == 'ANY')['time_in_alarm']:.1%}" for b in a.budgets)
        print(f"{name:11s} {info['events']:3d} events | {line}   ({time.time() - t0:.0f}s)")

    if not all_rows:
        sys.exit("No channels completed.")
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(a.out_dir, "per_channel.csv"), index=False)
    detail = pd.concat(all_detail, ignore_index=True)
    detail.to_csv(os.path.join(a.out_dir, "events_detail.csv"), index=False)
    allal = pd.concat([x for x in all_alerts if len(x)], ignore_index=True) if any(len(x) for x in all_alerts) else pd.DataFrame()
    allal.to_csv(os.path.join(a.out_dir, "alerts_all.csv"), index=False)
    for b in a.budgets:
        (allal[allal.budget == b] if len(allal) else allal).drop(columns="budget", errors="ignore").to_csv(
            os.path.join(a.out_dir, f"alerts_b{b:g}.csv"), index=False)

    n_ch = res.channel.nunique()
    weeks = res.weeks.mean()
    print("\n" + "=" * 100 + f"\nWALK-FORWARD OPERATING CURVE over {n_ch} channels (recalibrated every {a.step_days} d on the previous {a.window_days} d; no labels used)\n" + "=" * 100)
    print(f"{'budget':>7s} {'detector':8s}{'caught':>10s}{'chance':>8s}{'distinct':>9s}{'PR-AUC':>8s}{'ROC':>7s}{'alertPrec':>10s}{'FA/wk/ch':>9s}{'alerts/wk':>10s}{'inAlarm':>9s}{'ch>10%':>7s}")
    for b in a.budgets:
        rb = res[res.budget == b]
        db = detail[detail.budget == b]
        dist = db.groupby("event_id")[[f"hit_{d}" for d in DETS]].any()
        for d in DETS:
            r = rb[rb.detector == d]
            print(f"{b:7g} {d:8s}{str(int(r.events_caught.sum())) + '/' + str(int(r.events.sum())):>10s}"
                  f"{(r.chance_recall * r.events).sum():8.1f}{str(int(dist['hit_' + d].sum())) + '/' + str(len(dist)):>9s}"
                  f"{r.pr_auc.mean():8.3f}{r.roc_auc.mean():7.3f}{r.alert_prec.mean():10.2f}{r.fa_per_week_test.mean():9.2f}"
                  f"{r.episodes.sum() / weeks:10.1f}{r.time_in_alarm.mean():9.1%}{int((r.time_in_alarm > 0.10).sum()):7d}")
        print()
    print("caught = channel-level events (an event labelled on several channels counts on each); distinct = events caught on at least one of its labelled channels;")
    print("alerts/wk = alert episodes per week summed over ALL channels (what an operator would see).")
    print(f"alerts per budget -> {a.out_dir}\\alerts_b<budget>.csv   (next: python -m node13_v2.incidents --alerts {a.out_dir}\\alerts_b0.2.csv)")


if __name__ == "__main__":
    main()
