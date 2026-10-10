"""
node13_app/cli.py - run Node13 as a service: replay historical data day by day, or ingest new data.

  python -m node13_app.cli replay --channels 41-46 --from 2010-01-01 --to 2010-07-01
  python -m node13_app.cli ingest --channel channel_44 --file new_data.csv
  python -m node13_app.cli status
  python -m node13_app.cli reset

CSV format for ingest: two columns, timestamp (ISO, UTC) and value, with a header row.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from node13_v2.bench import ensure_pickle, parse_channels
from node13_v2.core import build_bins

from . import engine, store

BIN_MIN = 5


def _process(conn, ch: str, bins: pd.DataFrame, start, end, budget: float, chunk_days: int) -> dict:
    floors = store.get_floors(conn, ch)
    if floors is None:
        hist = bins[bins.index < pd.Timestamp(start)]
        floors = engine.make_floors(hist if len(hist) > 500 else bins)
    level, change = engine.compute_scores(bins, floors, BIN_MIN)
    res = engine.detect_range(bins, level, change, start, end, budget, chunk_days=chunk_days, bin_min=BIN_MIN)
    new = merged = 0
    for e in res["episodes"]:
        r = store.add_episode(conn, ch, e["start"], e["end"], e["bins"], e["ratio"], e["detector"], e["reason"],
                              e["context"], budget)
        new += r == "new"
        merged += r == "merged"
    store.set_floors(conn, ch, floors, bins.index[-1])
    store.save_history(conn, ch, bins, keep_days=150, bin_min=BIN_MIN)
    return {"new": new, "merged": merged, **{k: res[k] for k in ("calibrated_chunks", "skipped_chunks")}}


def cmd_replay(a):
    conn = store.connect(a.db)
    labels = types = None
    lp, tp = os.path.join(a.data, "labels.csv"), os.path.join(a.data, "anomaly_types.csv")
    if os.path.exists(lp) and os.path.exists(tp):
        labels = pd.read_csv(lp)
        types = pd.read_csv(tp)[["ID", "Category"]]
    chinfo = pd.read_csv(os.path.join(a.data, "channels.csv")).set_index("Channel")
    start, end = pd.Timestamp(a.from_), pd.Timestamp(a.to)
    for ch in parse_channels(a.channels, chinfo):
        name = f"channel_{ch}"
        if name in chinfo.index and str(chinfo.loc[name, "Categorical"]).upper() == "YES":
            print(f"{name}: categorical channel - skipped")
            continue
        t0 = time.time()
        try:
            pkl, extracted = ensure_pickle(ch, a.data, a.work, a.sevenzip)
            series = pd.read_pickle(pkl).iloc[:, 0]
            if a.cleanup and extracted:
                os.remove(pkl)
            series = series.sort_index()
            series = series[series.index < end]
            bins = build_bins(series, BIN_MIN)
            del series
            if a.reset:
                store.clear_channel(conn, name)
                conn.execute("DELETE FROM channels WHERE name=?", (name,))
                conn.commit()
            r = _process(conn, name, bins, start, end, a.budget, a.chunk_days)
            if labels is not None:
                lab = labels[labels.Channel == name].merge(types, on="ID", how="left")
                rows = [(x.ID, name, pd.to_datetime(x.StartTime, utc=True).tz_localize(None).strftime(store.FMT),
                         pd.to_datetime(x.EndTime, utc=True).tz_localize(None).strftime(store.FMT), x.Category)
                        for x in lab.itertuples()]
                conn.execute("DELETE FROM known_events WHERE channel=?", (name,))
                store.add_known_events(conn, rows)
            print(f"{name}: {r['new']} alerts ({r['merged']} merged) | recalibrated {r['calibrated_chunks']}x, "
                  f"{r['skipped_chunks']} chunks skipped (warm-up) | {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {e}")
    store.meta_set(conn, "budget", a.budget)
    store.meta_set(conn, "replay_from", a.from_)
    store.meta_set(conn, "replay_to", a.to)
    print(f"\nDone. Open the dashboard:  streamlit run node13_app\\dashboard.py")


def cmd_ingest(a):
    conn = store.connect(a.db)
    ch = a.channel
    if a.file.lower().endswith(".csv"):
        df = pd.read_csv(a.file)
        series = pd.Series(pd.to_numeric(df.iloc[:, 1]).to_numpy(),
                           index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0], utc=True).dt.tz_localize(None)))
    else:
        series = pd.read_pickle(a.file).iloc[:, 0]
    series = series.sort_index().astype("float64")
    new_bins = build_bins(series, BIN_MIN)
    if new_bins.empty:
        sys.exit("no data in file")
    hist = store.load_history(conn, ch)
    if len(hist):
        hist = hist[hist.index < new_bins.index[0]]
        if hist.index[-1] > new_bins.index[0]:
            sys.exit("new data overlaps stored history")
    bins = pd.concat([hist, new_bins])
    start, end = new_bins.index[0], new_bins.index[-1] + pd.Timedelta(minutes=BIN_MIN)
    budget = float(store.meta_get(conn, "budget", a.budget))
    r = _process(conn, ch, bins, start, end, budget, a.chunk_days)
    print(f"{ch}: ingested {len(new_bins):,} bins ({start} -> {end}); {r['new']} new alerts, {r['merged']} extended; "
          f"history held: {len(bins):,} bins" + ("  [warm-up: needs ~7+ days of history before alerts can be raised]" if r["calibrated_chunks"] == 0 else ""))


def cmd_status(a):
    conn = store.connect(a.db)
    al = store.alerts_df(conn)
    print(f"database: {a.db}")
    print(f"channels with history: {conn.execute('SELECT COUNT(*) c FROM channels').fetchone()['c']}")
    print(f"alerts: {len(al)}   by status: {al.status.value_counts().to_dict() if len(al) else {}}")
    print(f"budget (alert episodes/week/channel per detector): {store.meta_get(conn, 'budget', 'n/a')}")


def cmd_reset(a):
    if os.path.exists(a.db):
        os.remove(a.db)
    print(f"removed {a.db}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("NODE13_DB", "node13.db"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="feed ESA-ADB telemetry through Node13 as if it arrived live")
    r.add_argument("--channels", default="44")
    r.add_argument("--from", dest="from_", default="2010-01-01")
    r.add_argument("--to", default="2010-07-01")
    r.add_argument("--budget", type=float, default=0.2, help="alert episodes/week/channel per detector in normal data")
    r.add_argument("--chunk-days", type=int, default=7)
    r.add_argument("--data", default=os.path.join("data", "ESA-Mission1"))
    r.add_argument("--work", default=os.path.join("data", "work"))
    r.add_argument("--sevenzip", default=r"C:\Program Files\7-Zip\7z.exe")
    r.add_argument("--cleanup", action="store_true")
    r.add_argument("--reset", action="store_true", help="drop existing alerts for these channels first")
    r.set_defaults(fn=cmd_replay)

    i = sub.add_parser("ingest", help="ingest new telemetry for one channel (CSV: timestamp,value)")
    i.add_argument("--channel", required=True)
    i.add_argument("--file", required=True)
    i.add_argument("--budget", type=float, default=0.2)
    i.add_argument("--chunk-days", type=int, default=1)
    i.set_defaults(fn=cmd_ingest)

    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("reset", help="delete the database").set_defaults(fn=cmd_reset)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
