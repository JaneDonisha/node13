"""
Node13 - telemetry anomaly triage for spacecraft operations.
Run:  streamlit run node13_app\\dashboard.py
"""
from __future__ import annotations

import html
import json
import os
import sys

# `streamlit run node13_app\\dashboard.py` does not put the project folder on the import path - do it here.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from node13_app import store
from node13_v2.incidents import build_incidents

DB = os.environ.get("NODE13_DB", "node13.db")
RESULTS = os.environ.get("NODE13_RESULTS", "results_v3")

BLUE, AMBER, RED, SLATE, GREEN = "#3987e5", "#e8a317", "#d03b3b", "#8a99ad", "#2f9e5b"
TIERS = ["Low", "Medium", "High"]
TIER_COLOR = {"Low": SLATE, "Medium": AMBER, "High": RED}
STATUS_LABEL = {"new": "Needs review", "real": "Real", "expected": "Expected", "false": "False alarm"}

st.set_page_config(page_title="Node13 - telemetry triage", page_icon=":material/satellite_alt:", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1500px;}
h1 {font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0;}
.n13-sub {opacity: .65; font-size: .95rem; margin: .1rem 0 1.1rem 0;}
.n13-kpi {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: .85rem 1rem; background: rgba(128,128,128,.06);}
.n13-kpi .v {font-size: 1.9rem; font-weight: 650; line-height: 1.1;}
.n13-kpi .l {font-size: .82rem; opacity: .7; margin-top: .15rem;}
.n13-kpi .s {font-size: .78rem; opacity: .55; margin-top: .1rem;}
.pill {display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: .74rem; font-weight: 600; color: #fff; vertical-align: middle;}
.chip {display:inline-block; padding: 0 7px; margin: 0 4px 3px 0; border-radius: 5px; font-size: .74rem;
       background: rgba(128,128,128,.16); font-variant-numeric: tabular-nums;}
.n13-reason {font-size: .82rem; opacity: .75; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin-top:.2rem;}
[class*="st-key-card_"] {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: .55rem .8rem .5rem .8rem; margin-bottom: .1rem;
                         background: rgba(128,128,128,.05);}
[class*="st-key-card_sel_"] {border-color: #3987e5; box-shadow: inset 3px 0 0 #3987e5; background: rgba(57,135,229,.09);}
[class*="st-key-card_"] {gap: .3rem;}
[class*="st-key-card_"] button {padding: 0; min-height: 0; justify-content: flex-start; font-weight: 600; font-size: .98rem;}
[class*="st-key-card_"] button div, [class*="st-key-card_"] button p {text-align: left; justify-content: flex-start;}
.n13-h {font-size: 1.05rem; font-weight: 650; margin: .2rem 0 .3rem 0;}
.n13-card {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: .9rem 1.1rem; background: rgba(128,128,128,.05); height: 100%;}
.n13-card b {display:block; margin-bottom: .25rem;}
.n13-card.warn {border-left: 4px solid #d03b3b;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------- helpers
def esc(x) -> str:
    return html.escape(str(x))


def tier(ratio: float) -> str:
    return "High" if ratio >= 5 else "Medium" if ratio >= 2 else "Low"


def pill(t: str) -> str:
    return f'<span class="pill" style="background:{TIER_COLOR[t]}">{t}</span>'


def chnum(c: str) -> int:
    return int("".join(ch for ch in c if ch.isdigit()) or 0)


def short(c: str) -> str:
    return c.replace("channel_", "ch\u00a0")


def kpi(col, value, label, sub=""):
    col.markdown(f'<div class="n13-kpi"><div class="v">{value}</div><div class="l">{esc(label)}</div>'
                 f'<div class="s">{esc(sub)}</div></div>', unsafe_allow_html=True)


conn = store.connect(DB)
alerts = store.alerts_df(conn)
known = store.known_events_df(conn)

# ---------------------------------------------------------------------------------------------- header
st.markdown("# Node13")
st.markdown('<div class="n13-sub">Telemetry anomaly triage. Scores incoming telemetry, raises explained alerts for human review. '
            'It commands nothing and predicts no failures.</div>', unsafe_allow_html=True)

if alerts.empty:
    st.info("No alerts yet. Feed some telemetry first, for example:  \n"
            "`python -m node13_app.cli replay --channels 41-46 --from 2010-01-01 --to 2010-07-01`")
    st.stop()

# ---------------------------------------------------------------------------------------------- sidebar filters
with st.sidebar:
    st.markdown("### Filters")
    last = alerts.end.max().date()
    as_of = st.date_input("Data up to", value=last, min_value=alerts.start.min().date(), max_value=last,
                          help="Pretend 'now' is this date. Handy when replaying history.")
    status_f = st.multiselect("Status", list(STATUS_LABEL), default=["new"], format_func=STATUS_LABEL.get)
    chans = sorted(alerts.channel.unique(), key=chnum)
    chan_f = st.multiselect("Channels", chans, placeholder="All channels", format_func=short)
    sev = st.select_slider("Severity", ["All", "Medium and up", "High only"], value="All")
    min_ch = st.slider("Channels alerting together", 1, 10, 1,
                       help="Show only incidents where at least this many channels alert at the same time. "
                            "Several channels moving together is usually more meaningful than one.")
    sort = st.radio("Sort by", ["Highest score", "Newest", "Most channels"])
    st.divider()
    st.caption(f"Database: `{os.path.basename(DB)}`")

min_ratio = {"All": 0.0, "Medium and up": 2.0, "High only": 5.0}[sev]
upto = pd.Timestamp(as_of) + pd.Timedelta(days=1)
in_time = alerts[alerts.start <= upto]
view = in_time[in_time.status.isin(status_f) & (in_time.ratio >= min_ratio)]
if chan_f:
    view = view[view.channel.isin(chan_f)]

# ---------------------------------------------------------------------------------------------- incidents
incs = []
if len(view):
    base = build_incidents(view[["channel", "start", "end"]], min_ch, pd.Timedelta(minutes=15))
    for r in base.itertuples():
        hit = view[(view.start <= r.end) & (view.end >= r.start)]
        top = hit.sort_values("ratio", ascending=False).iloc[0]
        incs.append({"key": f"{hit.start.min():%Y%m%d%H%M}", "start": hit.start.min(), "end": hit.end.max(), "n_ch": hit.channel.nunique(),
                     "n_alerts": len(hit), "score": float(hit.ratio.max()), "tier": tier(hit.ratio.max()),
                     "channels": sorted(hit.channel.unique(), key=chnum), "reason": top.reason, "ids": hit.id.tolist(),
                     "n_new": int((hit.status == "new").sum())})
inc = pd.DataFrame(incs)
if len(inc):
    inc = inc.sort_values({"Highest score": "score", "Newest": "start", "Most channels": "n_ch"}[sort], ascending=False,
                          kind="stable").reset_index(drop=True)

reviewed = int((in_time.status != "new").sum())
c1, c2, c3, c4 = st.columns(4)
kpi(c1, len(inc), "Incidents in view", f"{len(view)} alerts after filters")
kpi(c2, int((inc.tier == "High").sum()) if len(inc) else 0, "High severity", "score 5x threshold or more")
kpi(c3, view.channel.nunique(), "Channels alerting", f"of {alerts.channel.nunique()} monitored")
kpi(c4, f"{reviewed} / {len(in_time)}", "Alerts reviewed", "marked real, expected or false")
st.write("")

tab_q, tab_h, tab_v, tab_a = st.tabs(["Triage", "Channel health", "Validation", "About and limits"])

# ================================================================================================ TRIAGE
with tab_q:
    if inc.empty:
        st.success("Nothing to show with these filters. Widen the filters in the sidebar, or include reviewed alerts under Status.")
    else:
        keys = inc.key.tolist()
        if st.session_state.get("sel") not in keys:
            st.session_state["sel"] = keys[0]
        sel = st.session_state["sel"]

        # ---- pulse chart: every incident as one lollipop
        st.markdown('<div class="n13-h">When incidents happened</div>', unsafe_allow_html=True)
        pc = inc.assign(tier=pd.Categorical(inc.tier, TIERS), when=inc.start, selected=(inc.key == sel),
                        score_txt=inc.score.round(1), zero=0)
        ymax = max(3, int(pc.n_ch.max()))
        x = alt.X("when:T", title=None)
        y = alt.Y("n_ch:Q", title="Channels", scale=alt.Scale(domain=[0, ymax + 0.5]), axis=alt.Axis(tickMinStep=1, grid=True))
        col = alt.Color("tier:N", scale=alt.Scale(domain=TIERS, range=[TIER_COLOR[t] for t in TIERS]),
                        legend=alt.Legend(title="Severity", orient="top", direction="horizontal"))
        tip = [alt.Tooltip("when:T", title="Start", format="%d %b %Y %H:%M"), alt.Tooltip("n_ch:Q", title="Channels"),
               alt.Tooltip("score_txt:Q", title="Top score (x threshold)"), alt.Tooltip("tier:N", title="Severity")]
        stems = alt.Chart(pc).mark_rule(strokeWidth=1.5, opacity=.55).encode(x=x, y=alt.Y("n_ch:Q"), y2="zero:Q", color=col)
        dots = alt.Chart(pc).mark_circle(size=90, opacity=1).encode(x=x, y=y, color=col, tooltip=tip)
        ring = alt.Chart(pc[pc.selected]).mark_point(shape="circle", size=330, filled=False, strokeWidth=2, color=BLUE).encode(x=x, y="n_ch:Q")
        st.altair_chart(alt.layer(stems, dots, ring).properties(height=190), width="stretch")

        # ---- master / detail
        left, right = st.columns([1, 1.75], gap="large")

        with left:
            n_show = 40
            st.markdown(f'<div class="n13-h">Incidents ({len(inc)})</div>', unsafe_allow_html=True)
            with st.container(height=760, border=False):
                for i, r in inc.head(n_show).iterrows():
                    is_sel = r.key == sel
                    with st.container(key=f"card{'_sel' if is_sel else ''}_{r.key}"):
                        st.button(f"{r.start:%d %b %Y, %H:%M}", key=f"b_{r.key}", type="tertiary", width="stretch",
                                  on_click=lambda k=r.key: st.session_state.update(sel=k))
                        chips = "".join(f'<span class="chip">{esc(short(c))}</span>' for c in r.channels[:6])
                        if r.n_ch > 6:
                            chips += f'<span class="chip">+{r.n_ch - 6}</span>'
                        dur = (r.end - r.start).total_seconds() / 60
                        meta = f"{r.n_ch} channel{'s' if r.n_ch != 1 else ''} · {dur:.0f} min · score {r.score:.1f}x"
                        st.markdown(f'{pill(r.tier)} <span style="font-size:.8rem;opacity:.7">{esc(meta)}</span><br>{chips}'
                                    f'<div class="n13-reason">{esc(r.reason)}</div>', unsafe_allow_html=True)
                if len(inc) > n_show:
                    st.caption(f"Showing the first {n_show} of {len(inc)}. Narrow the filters to see the rest.")

        with right:
            r = inc[inc.key == sel].iloc[0]
            hit = view[view.id.isin(r.ids)].sort_values("ratio", ascending=False)
            dur_min = (r.end - r.start).total_seconds() / 60
            st.markdown(f'<div class="n13-h" style="font-size:1.25rem">{r.start:%A %d %B %Y, %H:%M} {pill(r.tier)}</div>', unsafe_allow_html=True)
            if r.n_ch > 1:
                summary = (f"{r.n_ch} channels left their normal range at the same time for about {dur_min:.0f} minutes. "
                           f"The strongest was {short(hit.iloc[0].channel)} at {r.score:.1f}x its alarm threshold.")
            else:
                summary = (f"{short(hit.iloc[0].channel)} left its normal range for about {dur_min:.0f} minutes, "
                           f"reaching {r.score:.1f}x its alarm threshold.")
            st.write(summary)
            if len(known):
                kk = known[(known.end >= r.start - pd.Timedelta(hours=1)) & (known.start <= r.end + pd.Timedelta(hours=1))]
                kk = kk[kk.channel.isin(r.channels)]
                if len(kk):
                    st.info("Replay demo only: ESA annotated this period as **" + ", ".join(sorted(set(kk.category))) +
                            "**. Real operators would not have this label.")

            # small multiples: one chart per channel, shared time window
            shown = hit.drop_duplicates("channel").head(6)
            cols = st.columns(2 if len(shown) > 1 else 1)
            for j, a in enumerate(shown.itertuples()):
                ctx = store.get_context(conn, a.id)
                with cols[j % len(cols)]:
                    st.markdown(f'<div style="font-weight:600;font-size:.9rem">{esc(short(a.channel))} '
                                f'<span style="opacity:.6;font-weight:400">· {a.ratio:.1f}x · {esc(a.detector)}</span></div>',
                                unsafe_allow_html=True)
                    if not ctx:
                        st.caption("No context stored.")
                        continue
                    c = json.loads(ctx)
                    df = pd.DataFrame({"t": pd.to_datetime(c["t"]), "mean": c["mean"], "min": c["min"], "max": c["max"]})
                    ax = alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=5, labelFlush=False))
                    ay = alt.Y("mean:Q", title=None, scale=alt.Scale(zero=False))
                    layers = [alt.Chart(pd.DataFrame({"s": [min(a.start, a.end - pd.Timedelta(minutes=30))], "e": [a.end + pd.Timedelta(minutes=5)]})).mark_rect(color=AMBER, opacity=.35).encode(x="s:T", x2="e:T"),
                              alt.Chart(pd.DataFrame({"s": [a.start]})).mark_rule(color=RED, strokeWidth=1.5, strokeDash=[4, 3]).encode(x="s:T"),
                              alt.Chart(df).mark_area(opacity=.18, color=BLUE).encode(x=ax, y=alt.Y("min:Q", scale=alt.Scale(zero=False), title=None), y2="max:Q"),
                              alt.Chart(df).mark_line(color=BLUE, strokeWidth=1.6).encode(x=ax, y=ay)]
                    ke = known[(known.channel == a.channel) & (known.end >= df.t.min()) & (known.start <= df.t.max())] if len(known) else known
                    if len(ke):
                        layers.insert(0, alt.Chart(ke).mark_rect(color=GREEN, opacity=.16).encode(x="start:T", x2="end:T"))
                    st.altair_chart(alt.layer(*layers).properties(height=130), width="stretch")
            st.caption("Blue line: 5-minute mean, shaded band: min to max. Amber band and red dashed line: when the alert started (short alerts are drawn wider so they stay visible)."
                       + (" Green: ESA's own annotation (replay demo only)." if len(known) else "")
                       + (f" Showing 6 of {hit.channel.nunique()} channels." if hit.channel.nunique() > 6 else ""))

            with st.expander(f"Why each channel fired ({len(hit)})", expanded=r.n_ch == 1):
                for a in hit.itertuples():
                    st.markdown(f"**{short(a.channel)}** · {a.detector} · {a.ratio:.1f}x  \n{a.reason}")
                st.caption(f"Thresholds are re-set weekly from the previous 90 days so that, on normal data, each detector raises "
                           f"about {hit.budget.iloc[0]:g} alert(s) per week per channel.")

            st.markdown('<div class="n13-h" style="margin-top:.8rem">Your review</div>', unsafe_allow_html=True)
            with st.form(f"fb_{sel}", border=True):
                choice = st.radio("Verdict", ["real", "expected", "false"], horizontal=True, format_func=STATUS_LABEL.get,
                                  help="Real: genuine anomaly. Expected: known or planned behaviour. False alarm: nothing to see.")
                note = st.text_input("Note (optional)", placeholder="e.g. scheduled thruster burn")
                go = st.form_submit_button(f"Save for all {len(hit)} alert{'s' if len(hit) != 1 else ''}", type="primary")
                if go:
                    store.set_status_many(conn, r.ids, choice, note)
                    st.toast(f"Saved: {STATUS_LABEL[choice]}", icon=":material/check_circle:")
                    st.rerun()
            st.caption("Your verdicts are stored. They do not retrain anything yet.")

            with st.expander("What-if: contract exposure (hypothetical, not a prediction)"):
                n = st.number_input("Contracts running on this spacecraft", 0, 50, 3)
                v = st.number_input("Average contract value (EUR)", 0, 100_000_000, 1_000_000, step=100_000)
                d = st.slider("SLA penalty if service is breached", 0.0, 1.0, 0.1, 0.01)
                st.write(f"Exposure if this anomaly led to a breach of all {n} contracts: **EUR {n * v * d:,.0f}**")
                st.caption("Plain arithmetic on numbers you enter. Node13 does not estimate the chance of a breach.")

# ================================================================================================ HEALTH
with tab_h:
    t_end = alerts.end.max()
    span_days = max((t_end - alerts.start.min()).days, 1)
    weeks = span_days / 7
    g = alerts.assign(dur_h=(alerts.end - alerts.start).dt.total_seconds() / 3600).groupby("channel")
    hl = pd.DataFrame({"Alerts": g.size(), "Per week": (g.size() / weeks).round(2),
                       "Time in alarm": (g.dur_h.sum() / (span_days * 24)),
                       "Real": g.status.apply(lambda s: int((s == "real").sum())),
                       "Expected": g.status.apply(lambda s: int((s == "expected").sum())),
                       "False alarm": g.status.apply(lambda s: int((s == "false").sum())),
                       "Needs review": g.status.apply(lambda s: int((s == "new").sum()))})
    hl["Note"] = np.where(hl["Time in alarm"] > 0.10, "In alarm over 10% of the time - baseline may have shifted", "")
    hl = hl.reset_index().sort_values("Alerts", ascending=False)
    hl["Channel"] = hl.pop("channel").str.replace("channel_", "ch ")
    hl = hl[["Channel"] + [c for c in hl.columns if c != "Channel"]]

    st.markdown('<div class="n13-h">Alerts per channel per week</div>', unsafe_allow_html=True)
    hm = alerts.assign(week=alerts.start.dt.to_period("W").dt.start_time, ch=alerts.channel.map(short)) \
               .groupby(["ch", "week"]).size().reset_index(name="alerts")
    hm["week_end"] = hm.week + pd.Timedelta(days=7)
    order = [short(c) for c in sorted(alerts.channel.unique(), key=chnum)]
    heat = alt.Chart(hm).mark_rect(stroke="rgba(128,128,128,.15)").encode(
        x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %d")), x2="week_end:T",
        y=alt.Y("ch:N", sort=order, title=None),
        color=alt.Color("alerts:Q", scale=alt.Scale(range=["#d6e6fb", "#9cc3f5", "#6aa5ee", "#3987e5", "#1f64b8", "#12447f"]),
                        legend=alt.Legend(title="Alerts", orient="right", format="d", tickMinStep=1)),
        tooltip=[alt.Tooltip("ch:N", title="Channel"), alt.Tooltip("week:T", title="Week of", format="%d %b %Y"), alt.Tooltip("alerts:Q", title="Alerts")])
    st.altair_chart(heat.properties(height=max(120, 28 * len(order))), width="stretch")
    st.dataframe(hl, hide_index=True, width="stretch", column_config={
        "Channel": st.column_config.TextColumn("Channel", width="small"),
        "Time in alarm": st.column_config.ProgressColumn("Time in alarm", format="percent", min_value=0, max_value=0.2,
                                                         help="Share of the period this channel was alerting. Bar full = 20% or more."),
        "Note": st.column_config.TextColumn("Note", width="large")})
    st.caption(f"{len(alerts)} alerts over {span_days} days of replayed or ingested data.")

# ================================================================================================ VALIDATION
with tab_v:
    st.markdown('<div class="n13-h">Measured performance on real ESA telemetry</div>', unsafe_allow_html=True)
    pcf, edf = os.path.join(RESULTS, "per_channel.csv"), os.path.join(RESULTS, "events_detail.csv")
    if os.path.exists(pcf) and os.path.exists(edf):
        res, det = pd.read_csv(pcf), pd.read_csv(edf)
        legacy = "budget" not in res.columns
        if legacy:
            res["budget"] = 2.0
            det["budget"] = 2.0
        if "weeks" not in res.columns:
            res["weeks"] = 206.0   # ESA-ADB Mission 1 test period 2010-2013
        rows = []
        for b in sorted(res.budget.unique(), reverse=True):
            r_ = res[(res.budget == b) & (res.detector == "ANY")]
            d_ = det[det.budget == b]
            if "hit_ANY" not in d_.columns or r_.empty:
                continue
            ev = d_.groupby("event_id")["hit_ANY"].any()
            chance = float(r_.chance_recall.mean()) if "chance_recall" in r_.columns else np.nan
            rows.append({"Alert budget": b, "Alerts per week": float(r_.episodes.sum() / r_.weeks.mean()),
                         "Events caught": int(ev.sum()), "Events": int(len(ev)),
                         "Caught by chance": chance * len(ev) if chance == chance else np.nan,
                         "Time in alarm": float(r_.time_in_alarm.mean()),
                         "Alerts overlapping a labelled anomaly": float(r_.alert_prec.mean())})
        vt = pd.DataFrame(rows)
        if len(vt):
            tot = int(vt.Events.max())
            k1, k2, k3 = st.columns(3)
            best = vt.sort_values("Alerts per week").iloc[0]
            kpi(k1, f"{int(vt.Events.max())}", "Distinct anomaly events", "ESA-ADB Mission 1, test 2010-2013")
            kpi(k2, f"{int(best['Events caught'])} / {tot}", "Caught at the strictest budget", f"{best['Alerts per week']:.0f} alerts per week, all channels")
            kpi(k3, f"{int(vt.sort_values('Alerts per week').iloc[-1]['Events caught'])} / {tot}", "Caught at the loosest budget",
                f"{vt['Alerts per week'].max():.0f} alerts per week")
            st.write("")
            long = pd.concat([vt.assign(series="Node13", caught=vt["Events caught"]),
                              vt.dropna(subset=["Caught by chance"]).assign(series="Random alarms, same rate", caught=lambda d: d["Caught by chance"])])
            xs = alt.X("Alerts per week:Q", title="Alerts per week across all channels (lower is quieter)", scale=alt.Scale(type="log", domain=[max(1, vt["Alerts per week"].min() * 0.7), vt["Alerts per week"].max() * 1.3]))
            ln = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2).encode(
                x=xs, y=alt.Y("caught:Q", title=f"Events caught (of {tot})", scale=alt.Scale(domain=[0, tot])),
                color=alt.Color("series:N", scale=alt.Scale(domain=["Node13", "Random alarms, same rate"], range=[BLUE, SLATE]),
                                legend=alt.Legend(title=None, orient="top")),
                strokeDash=alt.StrokeDash("series:N", legend=None, scale=alt.Scale(domain=["Node13", "Random alarms, same rate"], range=[[1, 0], [5, 4]])),
                tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("Alerts per week:Q", format=".1f"),
                         alt.Tooltip("caught:Q", title="Events caught", format=".1f"), alt.Tooltip("Alert budget:Q")])
            st.altair_chart(ln.properties(height=300), width="stretch")
            show = vt.assign(**{"Events caught": vt["Events caught"].astype(str) + " / " + vt["Events"].astype(str)}).drop(columns=["Events"])
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "Alert budget": st.column_config.NumberColumn("Budget per channel per week", format="%g"),
                "Alerts per week": st.column_config.NumberColumn(format="%.1f"),
                "Caught by chance": st.column_config.NumberColumn(format="%.1f"),
                "Time in alarm": st.column_config.NumberColumn(format="percent"),
                "Alerts overlapping a labelled anomaly": st.column_config.NumberColumn(format="percent")})
            if legacy:
                st.warning("These results come from a single alert budget. Run `python -m node13_v2.walkforward --channels all --cleanup` "
                           "to add the budget sweep and see the trade-off curve.")
        st.caption(f"{res.channel.nunique()} channels. Thresholds were recalibrated monthly from the previous 90 days only; no labels were used. "
                   "An event counts as caught if an alert on one of its labelled channels overlaps it. 'Overlapping a labelled anomaly' understates "
                   "usefulness because alerts on rare nominal events and communication gaps count as non-matches.")
    else:
        st.info(f"No benchmark results found in `{RESULTS}`. Run `python -m node13_v2.walkforward --channels all --cleanup` to create them.")
    st.markdown('<div class="n13-card warn"><b>Read this before quoting any number</b>'
                'Only 34 distinct anomaly events exist in the test period, from a single ESA mission, so treat results as indicative. '
                'Sharp, short events are caught reliably; slow multi-hour drifts are mostly missed. Event-level counts are inflated for long events, '
                'which is why every result is compared with random alarm placement.</div>', unsafe_allow_html=True)

# ================================================================================================ ABOUT
with tab_a:
    a1, a2 = st.columns(2, gap="large")
    a1.markdown('<div class="n13-card"><b>What Node13 is</b>A triage aid for spacecraft operations. It watches telemetry next to the limit '
                'checks operators already use, and raises an explained alert when a channel changes in a way that is unusual for <i>that channel</i>. '
                'Alerts at the same time on several channels are grouped into one incident.</div>', unsafe_allow_html=True)
    a2.markdown('<div class="n13-card"><b>How it works</b>1. Telemetry is cut into 5-minute bins (mean, spread, min, max).<br>'
                '2. <i>Level check</i>: each bin versus the channel\'s own level over the previous 14 days.<br>'
                '3. <i>Change check</i>: each bin versus the previous 24 hours.<br>'
                '4. Alarm thresholds are re-set from the previous 90 days to hold the alert rate on normal data within a budget.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="n13-card warn"><b>What it does not do</b>'
                'It does not predict failures days ahead and gives no failure probabilities. '
                'It has not been validated on any spacecraft other than the ESA missions in the public benchmark. '
                'It does not use commanding context, so planned manoeuvres can look like anomalies: mark them <i>expected</i>. '
                'Operator verdicts are stored but do not retrain anything yet. The contract what-if is plain arithmetic on numbers you enter.</div>',
                unsafe_allow_html=True)
