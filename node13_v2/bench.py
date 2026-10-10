"""
node13_v2/bench.py - multi-channel benchmark of Node13 detector v2 on ESA-ADB Mission 1.

Run from the folder that contains `data\\` and `node13_v2\\`:
    python -m node13_v2.bench --channels 44                     (one channel)
    python -m node13_v2.bench --channels 41-46
    python -m node13_v2.bench --channels all --cleanup          (every numeric channel; deletes each
                                                                 extracted file after use to save disk)

Split by DATE: train < val-start <= validation < test-start <= test.  Training uses only bins that
overlap no labelled event.  Alarm thresholds are chosen on VALIDATION so that false-alarm episodes
stay under --fa-per-week; the TEST period is only measured, never tuned on.
Labels: Category == "Anomaly" is positive; other categories (rare events, gaps) are ignored in
testing and excluded from training.

Honest-metrics notes
  * One anomaly event is usually labelled on SEVERAL channels (same ID).  The summary therefore also
    reports DISTINCT events (an event counts once, "caught" if any channel's detector caught it).
  * Event recall is flattering when events are long (a random alarm inside a 20-day event "catches"
    it).  Every row also reports `chance`: the recall obtained by placing the same alarm episodes
    at RANDOM positions.  Only recall well above chance means anything.
  * `alert_prec` = share of alarm episodes in the test period that overlap a real anomaly event.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import (DETECTORS, MIN_NS, WEEK_NS, Calibrator, ChannelModel, build_alerts, build_bins,
                   choose_threshold, episodes, false_alarm_rate, fuse, interval_mask, to_ns)


def parse_channels(spec: str, chinfo: pd.DataFrame) -> List[int]:
    if spec.strip().lower() == "all":
        nums = [int(re.search(r"(\d+)$", n).group(1)) for n in chinfo.index]
        return sorted(nums)
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def ensure_pickle(ch: int, data_dir: str, work_dir: str, sevenzip: str):
    p = os.path.join(work_dir, f"channel_{ch}")
    if os.path.exists(p):
        return p, False
    z = os.path.join(data_dir, "channels", f"channel_{ch}.zip")
    if not os.path.exists(z):
        raise FileNotFoundError(z)
    os.makedirs(work_dir, exist_ok=True)
    subprocess.run([sevenzip, "e", z, f"-o{work_dir}", "-y"], check=True, stdout=subprocess.DEVNULL)
    return p, True


def event_hits(alarm_t, ev_s, ev_e, bin_ns):
    hit, delays = np.zeros(len(ev_s), dtype=bool), []
    for k, (s, e) in enumerate(zip(ev_s, ev_e)):
        i0 = np.searchsorted(alarm_t, s - bin_ns, side="right")
        i1 = np.searchsorted(alarm_t, e, side="right")
        if i1 > i0:
            hit[k] = True
            delays.append(max(0, alarm_t[i0] - s) / MIN_NS)
    return hit, (float(np.median(delays)) if delays else float("nan"))


def chance_recall(alarm_v, t_v, ev_s, ev_e, bin_ns, rng, reps=200):
    """Expected event recall if the same alarm EPISODES (same lengths) were placed at random."""
    a = np.where(alarm_v)[0]
    if len(a) == 0 or len(ev_s) == 0:
        return 0.0
    s_i, e_i = episodes(t_v[a])
    L = (a[e_i] - a[s_i] + 1).astype(int)
    nv = len(alarm_v)
    starts = rng.integers(0, max(nv - int(L.max()), 1), size=(reps, len(L)))
    rec = []
    for s, e in zip(ev_s, ev_e):
        i0 = np.searchsorted(t_v, s - bin_ns, side="right")
        i1 = np.searchsorted(t_v, e, side="right")
        rec.append(((starts < i1) & (starts + L[None, :] > i0)).any(axis=1).mean())
    return float(np.mean(rec))


def run_channel(ch: int, a, labels: pd.DataFrame, types: pd.DataFrame):
    name = f"channel_{ch}"
    pkl, extracted = ensure_pickle(ch, a.data, a.work, a.sevenzip)
    series = pd.read_pickle(pkl).iloc[:, 0]
    if a.cleanup and extracted:
        os.remove(pkl)
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    val_start, test_start = pd.Timestamp(a.val_start), pd.Timestamp(a.test_start)
    test_end = pd.Timestamp(a.test_end) if a.test_end else series.index[-1] + pd.Timedelta(days=1)
    series = series[series.index < test_end]
    bin_ns = a.bin_min * MIN_NS
    bins = build_bins(series, a.bin_min)
    del series
    times = to_ns(bins.index)
    X = bins.values

    lab = labels[labels.Channel == name].merge(types, on="ID", how="left").reset_index(drop=True)
    lab["Start"] = pd.to_datetime(lab.StartTime, utc=True).dt.tz_localize(None)
    lab["End"] = pd.to_datetime(lab.EndTime, utc=True).dt.tz_localize(None)
    is_anom = (lab.Category == "Anomaly").to_numpy()
    S, E = to_ns(lab.Start), to_ns(lab.End)
    margin = a.margin_min * MIN_NS
    any_ev = interval_mask(times, bin_ns, S, E, margin)
    anom_ev = interval_mask(times, bin_ns, S[is_anom], E[is_anom], 0)
    anom_ev_m = interval_mask(times, bin_ns, S[is_anom], E[is_anom], margin)
    ignore = interval_mask(times, bin_ns, S[~is_anom], E[~is_anom], 0) & ~anom_ev

    idx = bins.index
    train_mask = (idx < val_start) & ~any_ev
    if train_mask.sum() < 2000:
        raise RuntimeError("too few nominal training bins")
    model = ChannelModel.fit(X[train_mask], window_bins=int(24 * 60 / a.bin_min), seed=a.seed)

    off = int(np.searchsorted(idx.values, (val_start - pd.Timedelta(days=2)).to_datetime64()))
    sb = bins.iloc[off:]
    raw = model.score(sb)
    t_s = times[off:]
    va = ((idx >= val_start) & (idx < test_start))[off:]
    te = (idx >= test_start)[off:]
    val_nom = va & ~any_ev[off:]
    ign, pos, pos_m = ignore[off:], anom_ev[off:], anom_ev_m[off:]
    if val_nom.sum() < 1000:
        raise RuntimeError("too few nominal validation bins")

    sur, cals = {}, {}
    for d in DETECTORS:
        cals[d] = Calibrator(raw[d][val_nom])
        sur[d] = cals[d].surprise(raw[d])
    fused, which = fuse(sur)
    cap = cals["iforest"].cap
    sur["FUSED"] = fused

    valid = te & ~ign
    t_v = t_s[valid]
    test_weeks = valid.sum() * bin_ns / WEEK_NS
    in_test = is_anom & (lab.Start >= test_start).to_numpy() & (lab.Start < test_end).to_numpy()
    ev_s, ev_e = S[in_test], E[in_test]
    ev_ids = lab.ID.to_numpy()[in_test]
    ev_start = lab.Start[in_test].to_numpy()
    ev_dur_h = (ev_e - ev_s) / (60 * MIN_NS)
    y = pos[valid]
    rng = np.random.default_rng(a.seed)

    rows, hits = [], {}
    for d in DETECTORS + ["FUSED"]:
        s = sur[d]
        tau = choose_threshold(s, t_s, val_nom, bin_ns, a.fa_per_week)
        va_fa = false_alarm_rate(s, tau, t_s, val_nom, bin_ns)
        alarm = (s >= tau) & valid
        hit, delay = event_hits(t_s[alarm], ev_s, ev_e, bin_ns)
        hits[d] = hit
        ai = np.where(alarm)[0]
        s_i, e_i = episodes(t_s[ai])
        n_ep = len(s_i)
        true_ep = sum(bool(pos_m[ai[x:y_ + 1]].any()) for x, y_ in zip(s_i, e_i))
        both = 0 < y.sum() < len(y)
        rows.append({
            "channel": name, "detector": d,
            "roc_auc": roc_auc_score(y, s[valid]) if both else float("nan"),
            "pr_auc": average_precision_score(y, s[valid]) if y.sum() else float("nan"),
            "events": len(ev_s), "events_caught": int(hit.sum()),
            "chance_recall": chance_recall(alarm[valid], t_v, ev_s, ev_e, bin_ns, rng),
            "alert_prec": (true_ep / n_ep) if n_ep else float("nan"), "alert_episodes": n_ep,
            "median_delay_min": delay, "fa_per_week_val": va_fa,
            "fa_per_week_test": (n_ep - true_ep) / test_weeks if test_weeks > 0 else float("nan"),
            "threshold_surprise": tau, "max_test_surprise": float(s[valid].max()) if valid.any() else float("nan"),
            "val_tie_share": 1.0 - len(np.unique(np.round(raw[d][val_nom], 12))) / val_nom.sum() if d in raw else float("nan"),
        })

    alerts = build_alerts(name, sb, model, raw, sur, fused, which, rows[-1]["threshold_surprise"],
                          valid, t_s, 1440.0 / a.bin_min, cap)
    detail = pd.DataFrame({"channel": name, "event_id": ev_ids, "start": ev_start, "duration_h": ev_dur_h,
                           **{f"hit_{d}": hits[d] for d in DETECTORS + ["FUSED"]}})
    return rows, alerts, detail, {"test_events": len(ev_s), "test_pos_bins": int(y.sum()),
                                  "test_weeks": test_weeks, "base_rate": float(y.mean()) if len(y) else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", default="41-46", help='e.g. 44 | 41-46 | 12,44 | all')
    ap.add_argument("--data", default=os.path.join("data", "ESA-Mission1"))
    ap.add_argument("--work", default=os.path.join("data", "work"))
    ap.add_argument("--sevenzip", default=r"C:\Program Files\7-Zip\7z.exe")
    ap.add_argument("--cleanup", action="store_true", help="delete each extracted channel file after loading it")
    ap.add_argument("--bin-min", type=int, default=5)
    ap.add_argument("--val-start", default="2009-01-01")
    ap.add_argument("--test-start", default="2010-01-01")
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--fa-per-week", type=float, default=2.0, help="false-alarm episode budget per week")
    ap.add_argument("--margin-min", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results_v2")
    a = ap.parse_args()

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
        all_rows += rows
        all_alerts.append(alerts)
        all_detail.append(detail)
        f = {r["detector"]: r for r in rows}
        print(f"\n=== {name} ===  {info['test_events']} test events, {info['test_pos_bins']:,} positive bins, "
              f"{info['test_weeks']:.0f} weeks, random PR-AUC {info['base_rate']:.4f}  ({time.time() - t0:.0f}s)")
        print(f"  {'detector':9s}{'ROC':>6s}{'PR':>7s}{'caught':>8s}{'chance':>8s}{'alertPrec':>10s}{'FA/wk val':>10s}{'FA/wk test':>11s}{'tau':>6s}{'maxTest':>8s}")
        for r in rows:
            print(f"  {r['detector']:9s}{r['roc_auc']:6.3f}{r['pr_auc']:7.3f}"
                  f"{str(r['events_caught']) + '/' + str(r['events']):>8s}{r['chance_recall'] * r['events']:8.1f}"
                  f"{r['alert_prec']:10.2f}{r['fa_per_week_val']:10.2f}{r['fa_per_week_test']:11.2f}"
                  f"{r['threshold_surprise']:6.2f}{r['max_test_surprise']:8.2f}")
        alerts.to_csv(os.path.join(a.out_dir, f"alerts_{name}.csv"), index=False)

    if not all_rows:
        sys.exit("No channels completed.")
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(a.out_dir, "per_channel.csv"), index=False)
    detail = pd.concat(all_detail, ignore_index=True)
    detail.to_csv(os.path.join(a.out_dir, "events_detail.csv"), index=False)

    order = DETECTORS + ["FUSED"]
    g = res.groupby("detector")
    summ = pd.DataFrame({
        "chan_events": g.events.sum(), "caught": g.events_caught.sum(),
        "chance": (res.chance_recall * res.events).groupby(res.detector).sum(),
        "mean_pr_auc": g.pr_auc.mean(), "mean_roc_auc": g.roc_auc.mean(),
        "alert_prec": g.alert_prec.mean(), "fa_val": g.fa_per_week_val.mean(), "fa_test": g.fa_per_week_test.mean(),
    }).reindex(order)
    summ.to_csv(os.path.join(a.out_dir, "summary.csv"))

    n_ch = res.channel.nunique()
    print("\n" + "=" * 84 + f"\nSUMMARY over {n_ch} channels (false-alarm budget {a.fa_per_week:g} episodes/week)\n" + "=" * 84)
    print("Per-channel event counts (an event labelled on several channels is counted on each):")
    print(f"{'detector':9s}{'caught':>10s}{'chance':>9s}{'PR-AUC':>9s}{'ROC':>7s}{'alertPrec':>10s}{'FA/wk val':>10s}{'FA/wk test':>11s}")
    for d, r in summ.iterrows():
        print(f"{d:9s}{str(int(r.caught)) + '/' + str(int(r.chan_events)):>10s}{r.chance:9.1f}{r.mean_pr_auc:9.3f}"
              f"{r.mean_roc_auc:7.3f}{r.alert_prec:10.2f}{r.fa_val:10.2f}{r.fa_test:11.2f}")
    dist = detail.groupby("event_id")[[f"hit_{d}" for d in order]].any()
    print(f"\nDISTINCT events (each real event counted once; caught if ANY channel's detector caught it):")
    print(f"  {len(dist)} distinct anomaly events in the test period")
    for d in order:
        print(f"  {d:9s} caught {int(dist['hit_' + d].sum())}/{len(dist)}")
    missed = detail[~detail[[f"hit_{d}" for d in order]].any(axis=1)].event_id.unique()
    print(f"  missed by every detector on every channel: {len(missed)} -> see events_detail.csv")
    pv = res.pivot(index="channel", columns="detector", values="pr_auc")
    best = pv[DETECTORS].max(axis=1)
    print(f"\nFUSED PR-AUC beats the best single detector on {int((pv['FUSED'] > best).sum())} of {len(pv)} channels; "
          f"z_level is best-or-tied on {int((pv['z_level'] >= best - 1e-12).sum())}")
    allal = pd.concat([x for x in all_alerts if len(x)], ignore_index=True) if any(len(x) for x in all_alerts) else pd.DataFrame()
    allal.to_csv(os.path.join(a.out_dir, "alerts_all.csv"), index=False)
    print(f"{len(allal)} alerts -> {a.out_dir}\\alerts_all.csv ;  per-event detail -> {a.out_dir}\\events_detail.csv")


if __name__ == "__main__":
    main()
