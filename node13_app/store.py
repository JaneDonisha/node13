"""node13_app/store.py - SQLite storage: channel state, recent history, alerts, operator feedback."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels(name TEXT PRIMARY KEY, floor_mean REAL, floor_std REAL, last_data TEXT);
CREATE TABLE IF NOT EXISTS bins(channel TEXT, t TEXT, mean REAL, std REAL, min REAL, max REAL, slope REAL, mad_diff REAL,
                                PRIMARY KEY(channel, t));
CREATE TABLE IF NOT EXISTS alerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT, start TEXT, end TEXT, bins INTEGER, ratio REAL,
    detector TEXT, reason TEXT, context TEXT, budget REAL, status TEXT DEFAULT 'new', note TEXT DEFAULT '',
    created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS known_events(event_id TEXT, channel TEXT, start TEXT, end TEXT, category TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""
FEATURES = ["mean", "std", "min", "max", "slope", "mad_diff"]
FMT = "%Y-%m-%d %H:%M:%S"


def connect(path: str) -> sqlite3.Connection:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _ts(x) -> str:
    return pd.Timestamp(x).strftime(FMT)


# ---- channel state / history ------------------------------------------------
def get_floors(conn, ch: str) -> Optional[dict]:
    r = conn.execute("SELECT floor_mean, floor_std FROM channels WHERE name=?", (ch,)).fetchone()
    return None if r is None or r["floor_mean"] is None else {"mean": r["floor_mean"], "std": r["floor_std"]}


def set_floors(conn, ch: str, floors: dict, last_data) -> None:
    conn.execute("INSERT INTO channels(name, floor_mean, floor_std, last_data) VALUES(?,?,?,?) "
                 "ON CONFLICT(name) DO UPDATE SET floor_mean=excluded.floor_mean, floor_std=excluded.floor_std, "
                 "last_data=excluded.last_data", (ch, floors["mean"], floors["std"], _ts(last_data)))
    conn.commit()


def save_history(conn, ch: str, bins: pd.DataFrame, keep_days: int = 150, bin_min: int = 5) -> None:
    tail = bins.iloc[-int(keep_days * 24 * 60 / bin_min):]
    conn.execute("DELETE FROM bins WHERE channel=?", (ch,))
    rows = [(ch, t.strftime(FMT), *map(float, r)) for t, r in zip(tail.index, tail[FEATURES].to_numpy())]
    conn.executemany("INSERT OR REPLACE INTO bins VALUES(?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def load_history(conn, ch: str) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM bins WHERE channel=? ORDER BY t", conn, params=(ch,))
    if df.empty:
        return pd.DataFrame(columns=FEATURES, index=pd.DatetimeIndex([], name="t"))
    df["t"] = pd.to_datetime(df["t"])
    return df.set_index("t")[FEATURES]


# ---- alerts ---------------------------------------------------------------------
def add_episode(conn, ch: str, start, end, nbins: int, ratio: float, detector: str, reason: str,
                context: str, budget: float, merge_gap_h: float = 1.0) -> str:
    """Insert an alert; if it continues the channel's latest alert (gap <= merge_gap_h) extend that one."""
    last = conn.execute("SELECT id, end, bins, ratio FROM alerts WHERE channel=? ORDER BY end DESC LIMIT 1", (ch,)).fetchone()
    now = pd.Timestamp.now().strftime(FMT)
    if last is not None and pd.Timestamp(start) <= pd.Timestamp(last["end"]) + pd.Timedelta(hours=merge_gap_h) \
            and pd.Timestamp(end) >= pd.Timestamp(last["end"]):
        conn.execute("UPDATE alerts SET end=?, bins=?, ratio=?, context=?, updated=? WHERE id=?",
                     (_ts(end), last["bins"] + nbins, max(last["ratio"], ratio), context, now, last["id"]))
        conn.commit()
        return "merged"
    conn.execute("INSERT INTO alerts(channel,start,end,bins,ratio,detector,reason,context,budget,created,updated) "
                 "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                 (ch, _ts(start), _ts(end), nbins, float(ratio), detector, reason, context, budget, now, now))
    conn.commit()
    return "new"


def alerts_df(conn, with_context: bool = False) -> pd.DataFrame:
    """All alerts, newest first. The (large) context JSON is left out unless asked for."""
    cols = "id, channel, start, end, bins, ratio, detector, reason, budget, status, note, created, updated"
    if with_context:
        cols += ", context"
    df = pd.read_sql_query(f"SELECT {cols} FROM alerts ORDER BY start DESC", conn)
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c])
    return df


def get_context(conn, alert_id: int) -> str:
    r = conn.execute("SELECT context FROM alerts WHERE id=?", (int(alert_id),)).fetchone()
    return "" if r is None or r["context"] is None else r["context"]


def set_status(conn, alert_id: int, status: str, note: str = "") -> None:
    set_status_many(conn, [alert_id], status, note)


def set_status_many(conn, alert_ids, status: str, note: str = "") -> None:
    now = pd.Timestamp.now().strftime(FMT)
    conn.executemany("UPDATE alerts SET status=?, note=?, updated=? WHERE id=?",
                     [(status, note, now, int(i)) for i in alert_ids])
    conn.commit()


def clear_channel(conn, ch: str) -> None:
    conn.execute("DELETE FROM alerts WHERE channel=?", (ch,))
    conn.execute("DELETE FROM known_events WHERE channel=?", (ch,))
    conn.commit()


# ---- benchmark ground truth (only for the ESA replay demo) -------------------------
def add_known_events(conn, rows) -> None:
    conn.executemany("INSERT INTO known_events VALUES(?,?,?,?,?)", rows)
    conn.commit()


def known_events_df(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM known_events", conn)
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c])
    return df


def meta_get(conn, key: str, default=None):
    r = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return default if r is None else r["value"]


def meta_set(conn, key: str, value) -> None:
    conn.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
    conn.commit()
