"""
node13_v2/incidents.py - group per-channel alerts into INCIDENTS and measure them.

Idea: a real spacecraft event usually shows up on several channels at once, while false alarms
on different channels are mostly independent.  So instead of one alert per channel an incident is a
period when at least k channels are alerting at the same time.

Reads results_v2\\alerts_all.csv (written by node13_v2.bench) and the ESA labels.  No data
reloading, runs in seconds.

    python -m node13_v2.incidents
    python -m node13_v2.incidents --ks 1,3,5,10 --window-min 60

Truth: a DISTINCT anomaly event (all rows with the same ID merged, Category == "Anomaly") in the
test period.  An incident is "true" if it overlaps an event (+/- margin).  An event is "caught" if
any incident overlaps it.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _ns(ts: pd.Series) -> np.ndarray:
    """Datetimes -> int64 NANOSECONDS, whatever resolution pandas parsed them at (ns/us/ms/s)."""
    return pd.DatetimeIndex(ts).values.astype("datetime64[ns]").astype("int64")


def build_incidents(al: pd.DataFrame, k: int, pad: pd.Timedelta) -> pd.DataFrame:
    """Incident = maximal time range where >= k DIFFERENT channels have an active alert.
    Each alert is active over [start - pad, end + 5 min + pad].  Unlike chaining alerts together,
    this cannot snowball: unrelated alerts only count if they genuinely overlap in time."""
    ev_t, ev_d, ev_c = [], [], []
    for ch, g in al.groupby("channel"):
        iv = sorted(zip(_ns(g.start - pad), _ns(g.end + pd.Timedelta(minutes=5) + pad)))
        cs, ce = iv[0]
        merged = []
        for s_, e_ in iv[1:]:
            if s_ <= ce:
                ce = max(ce, e_)
            else:
                merged.append((cs, ce)); cs, ce = s_, e_
        merged.append((cs, ce))
        for s_, e_ in merged:
            ev_t += [s_, e_]; ev_d += [1, -1]; ev_c += [ch, ch]
    t = np.array(ev_t, dtype="int64"); d = np.array(ev_d)
    order = np.lexsort((d, t))                   # at equal times, ends (-1) before starts (+1)
    t, d = t[order], d[order]
    conc = np.cumsum(d)
    rows, start = [], None
    for i in range(len(t)):
        if conc[i] >= k and start is None:
            start = t[i]
        elif conc[i] < k and start is not None:
            rows.append((start, t[i])); start = None
    # merge touching ranges
    out = []
    for s_, e_ in rows:
        if out and s_ <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e_)
        else:
            out.append([s_, e_])
    inc = pd.DataFrame(out, columns=["start", "end"])
    for c in ("start", "end"):
        inc[c] = pd.to_datetime(inc[c], unit="ns")
    return inc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", default=os.path.join("results_v2", "alerts_all.csv"))
    ap.add_argument("--data", default=os.path.join("data", "ESA-Mission1"))
    ap.add_argument("--test-start", default="2010-01-01")
    ap.add_argument("--test-end", default="2014-01-01")
    ap.add_argument("--window-min", type=int, default=30, help="alerts within this many minutes count as simultaneous")
    ap.add_argument("--margin-min", type=int, default=60)
    ap.add_argument("--ks", default="1,2,3,4,5,6,8,10")
    a = ap.parse_args()

    al = pd.read_csv(a.alerts, parse_dates=["start", "end"])
    lab = pd.read_csv(os.path.join(a.data, "labels.csv"))
    typ = pd.read_csv(os.path.join(a.data, "anomaly_types.csv"))[["ID", "Category"]]
    lab = lab.merge(typ, on="ID", how="left")
    lab["Start"] = pd.to_datetime(lab.StartTime, utc=True).dt.tz_localize(None)
    lab["End"] = pd.to_datetime(lab.EndTime, utc=True).dt.tz_localize(None)
    t0, t1 = pd.Timestamp(a.test_start), pd.Timestamp(a.test_end)
    other = lab[(lab.Category != "Anomaly") & (lab.Start >= t0) & (lab.Start < t1)].groupby("ID").agg(
        start=("Start", "min"), end=("End", "max")).reset_index()          # rare events, communication gaps, ...
    lab = lab[lab.Category == "Anomaly"].copy()
    ev = lab[(lab.Start >= t0) & (lab.Start < t1)].groupby("ID").agg(
        start=("Start", "min"), end=("End", "max"), channels=("Channel", "nunique")).reset_index()
    ev["dur_h"] = (ev.end - ev.start).dt.total_seconds() / 3600
    weeks = (t1 - t0).days / 7.0
    sanity = build_incidents(al, 1, pd.Timedelta(minutes=a.window_min / 2))
    if len(sanity) and not (al.start.min() - pd.Timedelta(days=1) <= sanity.start.min() <= al.start.min()):
        raise SystemExit("time-unit mismatch: incidents do not line up with the alerts. Please report this.")
    margin, pad = pd.Timedelta(minutes=a.margin_min), pd.Timedelta(minutes=a.window_min / 2)
    total_h = (t1 - t0).total_seconds() / 3600
    print(f"{len(al):,} channel alerts across {al.channel.nunique()} channels, {len(ev)} distinct anomaly events, "
          f"{weeks:.0f} weeks of test data.  Alerts count as simultaneous within {a.window_min} min.\n")
    print(f"{len(other)} rare-event / communication-gap records also fall in the test period; incidents overlapping them are 'explained'.\n")
    print(f"{'min ch':>7s}{'incidents':>10s}{'per wk':>7s}{'true':>6s}{'explained':>10s}{'unexpl/wk':>10s}"
          f"{'caught':>8s}{'chance':>8s}{'med delay m':>12s}{'max dur h':>10s}{'covered':>9s}")
    rng = np.random.default_rng(0)
    ev_s = (ev.start - margin).astype("datetime64[ns]").astype("int64").to_numpy() if len(ev) else np.array([], dtype="int64")
    ev_e = (ev.end + margin).astype("datetime64[ns]").astype("int64").to_numpy() if len(ev) else np.array([], dtype="int64")
    t0n, t1n = int(pd.Timestamp(t0).value), int(pd.Timestamp(t1).value)
    caught_by_k = {}
    for k in [int(x) for x in a.ks.split(",")]:
        inc = build_incidents(al, k, pad)
        dur_h = ((inc.end - inc.start).dt.total_seconds() / 3600) if len(inc) else pd.Series(dtype=float)
        is_true = np.zeros(len(inc), dtype=bool)
        hit, delays = np.zeros(len(ev), dtype=bool), []
        for j, e in enumerate(ev.itertuples()):
            m = (inc.start <= e.end + margin) & (inc.end >= e.start - margin)
            if m.any():
                hit[j] = True
                is_true |= m.to_numpy()
                delays.append(max(0.0, (inc.start[m].min() - e.start).total_seconds() / 60))
        caught_by_k[k] = hit
        n_true = int(is_true.sum())
        expl = np.zeros(len(inc), dtype=bool)
        for o in other.itertuples():
            expl |= ((inc.start <= o.end + margin) & (inc.end >= o.start - margin)).to_numpy()
        n_expl = int((expl & ~is_true).sum())
        n_unexpl = len(inc) - n_true - n_expl
        # chance: same incident durations placed at random times in the test period
        chance = 0.0
        if len(inc) and len(ev):
            D = ((inc.end - inc.start).astype("timedelta64[ns]").astype("int64")).to_numpy()
            rs = t0n + rng.random((100, len(D))) * np.maximum(t1n - t0n - D, 1)[None, :]
            re_ = rs + D[None, :]
            chance = float(np.mean([((rs <= ev_e[j]) & (re_ >= ev_s[j])).any(axis=1).mean() for j in range(len(ev))]) * len(ev))
        print(f"{k:>7d}{len(inc):>10d}{len(inc) / weeks:>7.2f}{n_true:>6d}{n_expl:>10d}{n_unexpl / weeks:>10.2f}"
              f"{str(int(hit.sum())) + '/' + str(len(ev)):>8s}{chance:>8.1f}"
              f"{(np.median(delays) if delays else float('nan')):>12.0f}"
              f"{(dur_h.max() if len(dur_h) else float('nan')):>10.1f}"
              f"{(dur_h.sum() / total_h if len(dur_h) else 0.0):>9.1%}")

    k_show = sorted(caught_by_k)[min(2, len(caught_by_k) - 1)]
    ev["caught"] = caught_by_k[k_show]
    print(f"\nEvents at k = {k_show} (sorted by duration):")
    print(ev.sort_values("dur_h")[["ID", "start", "dur_h", "channels", "caught"]].to_string(index=False))


if __name__ == "__main__":
    main()
