"""Streamlit viewer for the AI Review Insight Engine. Reads precomputed files only: no LLM calls, no API keys, no torch.

Run locally:  streamlit run dashboard/app.py
Deploy:       Streamlit Community Cloud, main file dashboard/app.py. It installs dashboard/requirements.txt (light),
              not the repo's full requirements.txt (torch, BERTopic).
"""
import json
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import ui  # page design: moodboard palette, Notion-like layout, Revi the mascot (dashboard/ui.py)

ROOT = Path(__file__).resolve().parent.parent  # repository root (data/, results/, analysis/, eval/, BRIEF.md)
REPO_URL = "https://github.com/Abel1712/review-insight-engine"

st.set_page_config(page_title="Rapido Review Insights", page_icon="🛵", layout="wide")


@st.cache_data
def load():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    themes = pd.read_csv(ROOT / "data" / "themes.csv")
    sample = pd.read_csv(ROOT / "data" / "sample_final.csv")
    return metrics, themes, sample


metrics, themes, sample = load()
by_app = metrics["themes"]["by_app"]
ex, ve, ev = metrics["extract"], metrics["verify"], metrics["evaluation"]["ours"]
mode = ui.theme_type()
SURFACE, MUTE = ui.SURFACE[mode], ui.MUTE[mode]  # chart rings and annotations must match the page


def trend_chart() -> alt.LayerChart:
    """Monthly share of rider reviews mentioning tip / extra-money demands, 95% CIs, the CCPA order date and the
    pre-order average. Points sit mid-month, so the mid-September order date cuts through September's point."""
    T = metrics["trend"]
    s = T["series"]["Tip / extra-money demands"]
    df = pd.DataFrame([{
        "date": f"{m}-15", "share": v["share"], "lo": v["ci95"][0], "hi": v["ci95"][1],
        "month": pd.Timestamp(f"{m}-01").strftime("%B") + (" (partial month)" if m == T["partial_month"] else ""),
        "reviews": f"{v['k']} of {v['n']}", "partial": m == T["partial_month"]} for m, v in s["by_month"].items()])
    months = [pd.Timestamp(d) for d in df["date"]]
    x = alt.X("date:T", title=None,
              scale=alt.Scale(domain=[alt.DateTime(year=2026, month=3, date=1), alt.DateTime(year=2026, month=10, date=31)]),
              axis=alt.Axis(values=[alt.DateTime(year=d.year, month=d.month, date=15) for d in months], format="%b",
                            grid=False, ticks=False, labelPadding=8))
    yscale = alt.Scale(domain=[0, 0.3])
    y = alt.Y("share:Q", title=None, scale=yscale, axis=alt.Axis(format="%", tickCount=4, domain=False, ticks=False))
    base = alt.Chart(df).encode(x=x)
    band = base.mark_area(color=ui.ACCENT, opacity=0.12, interpolate="monotone").encode(
        y=alt.Y("lo:Q", scale=yscale), y2="hi:Q")
    line = base.mark_line(color=ui.ACCENT, strokeWidth=2, interpolate="monotone", strokeCap="round").encode(y=y)
    hover = alt.selection_point(on="pointerover", nearest=True, fields=["date"], clear="pointerout", empty=False)
    points = base.mark_point(filled=True, opacity=1, strokeWidth=2).encode(
        y=y,
        fill=alt.when(alt.datum.partial).then(alt.value(SURFACE)).otherwise(alt.value(ui.ACCENT)),
        stroke=alt.when(alt.datum.partial).then(alt.value(ui.ACCENT)).otherwise(alt.value(SURFACE)),
        size=alt.when(hover).then(alt.value(150)).otherwise(alt.value(80)),
        tooltip=[alt.Tooltip("month:N"), alt.Tooltip("share:Q", format=".1%", title="share of rider reviews"),
                 alt.Tooltip("reviews:N", title="reviews mentioning it"),
                 alt.Tooltip("lo:Q", format=".1%", title="95% CI low"), alt.Tooltip("hi:Q", format=".1%", title="95% CI high")],
    ).add_params(hover)
    order = pd.Timestamp(T["ccpa_order_date"])
    ccpa = alt.Chart(pd.DataFrame({"date": [T["ccpa_order_date"]], "label": [f"CCPA order · {order.day} {order:%b}"]}))
    ccpa_rule = ccpa.mark_rule(color=MUTE, strokeWidth=1).encode(x="date:T")
    ccpa_text = ccpa.mark_text(align="left", baseline="top", dx=6, color=MUTE, fontSize=11.5).encode(
        x="date:T", y=alt.value(2), text="label:N")
    pre = alt.Chart(pd.DataFrame({"a": ["2026-03-01"], "b": ["2026-08-31"], "y": [s["pre_order_share"]],
                                  "label": [f"Mar–Aug average {s['pre_order_share']:.1%}"]}))
    pre_rule = pre.mark_rule(color=MUTE, strokeWidth=1).encode(x="a:T", x2="b:T", y=alt.Y("y:Q", scale=yscale))
    pre_text = pre.mark_text(align="left", baseline="bottom", dx=2, dy=-5, color=MUTE, fontSize=11.5).encode(
        x="a:T", y=alt.Y("y:Q", scale=yscale), text="label:N")
    return (alt.layer(band, pre_rule, pre_text, ccpa_rule, ccpa_text, line, points)
            .properties(height=300, background="transparent").configure_view(strokeWidth=0))


# ---------- header ----------
ui.inject_css()
ui.hero("What should Rapido fix first?", "fix first?",
        f"I'm <b>Revi</b>. I read {ex['n_reviews']} Google Play reviews of Rapido's rider and captain apps, split them "
        "into individual complaints, checked every one against the review's own words, and ranked them. "
        "Relative signals from reviewers, not population rates.",
        ["Google Play reviews", "Mar–Oct 2026", "Rider + Captain apps"])
ui.kpis([
    ("Reviews analysed", ex["n_reviews"], "int",
     f"rider {by_app['rider']['n_reviews']} · captain {by_app['captain']['n_reviews']}", "mint", ui.P["sage"]),
    ("Complaints & praise found", ve["n_kept"], "int",
     f"{ve['n_in']} extracted, {ve['n_in'] - ve['n_kept']} failed the checks", "blush", ui.P["salmon"]),
    ("Reviews with 2+ issues", ex["share_reviews_2plus_aspects"], "pct",
     f"{ex['avg_aspects_per_review']:.2f} issues per review on average", "mustard", ui.P["mustard"]),
    ("Complaints found (recall)", ev["phrase_level"]["recall"], "pct",
     f"precision {ev['precision_strict']:.1%} · {metrics['evaluation']['n_test_reviews']} unseen reviews", "sage",
     ui.P["forest"]),
])

ui.section("Explore by app")
app = st.segmented_control("App", ["rider", "captain"], default="rider", required=True, label_visibility="collapsed",
                           format_func=lambda a: "🛵 Rider (customers)" if a == "rider" else "🪖 Captain (drivers)")
switched = st.session_state.get("last_app") != app  # a little celebration when you switch apps
st.session_state["last_app"] = app
ui.confetti("🛵" if app == "rider" else "🪖", switched)
data = by_app[app]
who = "Riders" if app == "rider" else "Captains"

tab_issues, tab_safety, tab_trend, tab_eval, tab_brief = st.tabs(
    ["Top issues", "Safety", "Tip & price trend", "Accuracy", "Brief"])

with tab_issues, st.container(key="sheet-issues"):  # content sits on a raised sheet
    neg = pd.DataFrame(data["negative"])
    top = neg.iloc[0]
    ui.say(f"{who}' #1 complaint: <b>{escape(top['theme'])}</b>, in {top['frequency']:.1%} of {app} reviews. "
           "Pick a theme below to read what people actually wrote.", "mint")
    ui.section("Complaints", f"Complaint themes in the {app} app, ranked by priority",
               "priority = share of this app's reviews mentioning the theme × average severity (1–5)", ui.P["orange"])
    hover = alt.selection_point(on="pointerover", clear="pointerout", fields=["theme"])
    chart = (alt.Chart(neg.head(12))
             .mark_bar(color=ui.ACCENT, cornerRadiusEnd=4, size=18)
             .encode(x=alt.X("priority:Q", title="priority (share × severity)", axis=alt.Axis(tickCount=5, domain=False)),
                     y=alt.Y("theme:N", sort="-x", title=None,
                             axis=alt.Axis(labelLimit=320, ticks=False, domain=False, labelPadding=8)),
                     opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0.35)),
                     tooltip=[alt.Tooltip("theme:N"), alt.Tooltip("frequency:Q", format=".1%", title="% of reviews"),
                              alt.Tooltip("avg_severity:Q", format=".2f", title="avg severity"),
                              alt.Tooltip("avg_stars:Q", format=".1f", title="avg stars"),
                              alt.Tooltip("priority:Q", format=".3f")])
             .add_params(hover)
             .properties(height=30 * min(len(neg), 12), background="transparent")
             .configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")

    pick = st.selectbox("Open a theme to see what users actually wrote", neg["theme"].tolist())
    rows = themes[(themes["app"] == app) & (themes["theme"] == pick)]
    row = neg[neg["theme"] == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews mentioning it", f"{row['frequency']:.1%}")
    c2.metric("Average severity", f"{row['avg_severity']:.2f}")
    c3.metric("Average stars", f"{row['avg_stars']:.1f}")
    st.dataframe(rows[["phrase", "evidence", "impact", "month"]].rename(
        columns={"phrase": "extracted complaint", "evidence": "exact words from the review"}),
        width="stretch", hide_index=True)

    pos = pd.DataFrame(data["positive"])
    if len(pos):
        ui.section("What's working", f"What {who.lower()} praise", "share of this app's reviews", ui.P["sage"])
        ui.good([(f"{r.frequency:.1%}", escape(r.theme)) for r in pos.itertuples()])

with tab_safety, st.container(key="sheet-safety"):  # content sits on a raised sheet
    s = metrics["themes"]["safety"][app]
    ui.say(f"Safety comes first and is never ranked by a formula: <b>{s['n_reviews']} {app} reviews "
           f"({s['share_of_reviews']:.1%})</b> describe a safety issue.", "blush")
    ui.section("Safety", "Safety issues, listed separately",
               "A review is flagged when the extractor, or the cross-model tie-breaker (OR rule), finds a safety or "
               "harassment issue backed by a verbatim quote.", ui.P["terracotta"])
    st.metric(f"{app.capitalize()} reviews with a safety issue", f"{s['n_reviews']} ({s['share_of_reviews']:.1%})")
    for quote in s["examples"]:
        ui.quote(escape(quote))

with tab_trend, st.container(key="sheet-trend"):  # content sits on a raised sheet
    tr = metrics["trend"]["series"]["Tip / extra-money demands"]
    ui.say(f"Before the CCPA order, <b>{tr['pre_order_share']:.1%}</b> of rider reviews mentioned tips or extra "
           "money. It looks lower since, but it's too early to call. This is the metric to watch.", "mustard")
    if app == "captain":
        ui.note("This chart covers the rider app, where tip and extra-money demands are reported.", "🛵")
    ui.section("Trend", "Rider complaints about tips and extra money, around the CCPA order",
               "share of rider reviews per month · shaded band = 95% confidence interval · hover a point for counts",
               ui.P["mustard"])
    st.altair_chart(trend_chart(), width="stretch")
    st.markdown(f"Before the order (Mar–Aug), **{tr['pre_order_share']:.1%}** of rider reviews mentioned tip or "
                f"extra-money demands (95% CI {tr['pre_order_ci95'][0]:.1%}–{tr['pre_order_ci95'][1]:.1%}). "
                "The post-order window is only a few weeks with ~35 reviews per month, and overall complaint "
                "share also fell, so **no causal claim**: this is the metric to watch.")

with tab_eval, st.container(key="sheet-eval"):  # content sits on a raised sheet
    ui.say(f"On {metrics['evaluation']['n_test_reviews']} reviews I had never seen, I found "
           f"<b>{ev['phrase_level']['recall']:.1%}</b> of the complaints, and <b>{ev['precision_strict']:.1%}</b> "
           "of what I reported was real.", "sage")
    ui.section("Accuracy", "How accurate is it?", dot=ui.P["forest"])
    st.markdown((ROOT / "eval" / "results.md").read_text(encoding="utf-8").split("\n", 1)[1])
    js = metrics.get("judge_spotcheck")
    if js:
        st.caption(f"LLM judge vs human spot-check: {js['n_agree']}/{js['n']} agreed "
                   f"(95% CI {js['agreement_ci95'][0]:.0%}–{js['agreement_ci95'][1]:.0%}).")
    c = metrics["consistency"]["results"]["0.7"]
    st.caption(f"Cross-model consistency (Qwen vs gpt-oss on 50 reviews): {c['aspect_agreement']:.1%} of extracted "
               f"complaints matched; {c['impact_agreement']:.1%} of matched complaints got the same impact label.")

with tab_brief, st.container(key="sheet-brief"):  # content sits on a raised sheet
    ui.say("Here's the one-page brief: what to fix first, and how to measure whether the fix worked.", "blush")
    brief = (ROOT / "BRIEF.md").read_text(encoding="utf-8")
    st.download_button("Download BRIEF.md", brief, file_name="BRIEF.md", mime="text/markdown")
    before, _, after = brief.partition("![trend](analysis/figures/trend_tipping.png)")
    st.markdown(before)
    if after:  # Markdown image links don't resolve in Streamlit, so show the chart directly
        st.altair_chart(trend_chart(), width="stretch")
        st.markdown(after)

ui.footer("Every number on this page is read from <code>results/metrics.json</code>.",
          f'<a href="{REPO_URL}" target="_blank" rel="noopener">Code &amp; method on GitHub ↗</a>')
ui.theme_script()
