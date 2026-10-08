"""Streamlit viewer for the AI Review Insight Engine. Reads precomputed files only: no LLM calls, no API keys, no torch.

Run locally:  streamlit run dashboard/app.py
Deploy:       Streamlit Community Cloud, main file dashboard/app.py. It installs dashboard/requirements.txt (light),
              not the repo's full requirements.txt (torch, BERTopic).
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent  # repository root (data/, results/, analysis/, eval/, BRIEF.md)
BLUE, INK_2 = "#2a78d6", "#52514e"

st.set_page_config(page_title="Rapido Review Insights", page_icon="🛵", layout="wide")


@st.cache_data
def load():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    themes = pd.read_csv(ROOT / "data" / "themes.csv")
    sample = pd.read_csv(ROOT / "data" / "sample_final.csv")
    return metrics, themes, sample


metrics, themes, sample = load()
by_app = metrics["themes"]["by_app"]

st.title("Rapido Review Insights")
st.caption(f"{metrics['extract']['n_reviews']} Google Play reviews of the Rapido rider and captain apps, "
           "Mar–Oct 2026, split into individual complaints by an LLM, verified, clustered and ranked. "
           "Relative signals from reviewers, not population rates.")

app = st.radio("App", ["rider", "captain"], horizontal=True,
               format_func=lambda a: f"{'Rider (customers)' if a == 'rider' else 'Captain (drivers)'}")
data = by_app[app]

tab_issues, tab_safety, tab_trend, tab_eval, tab_brief = st.tabs(
    ["Top issues", "Safety", "Tip & price trend", "Accuracy", "Brief"])

with tab_issues:
    neg = pd.DataFrame(data["negative"])
    st.subheader(f"Complaint themes in the {app} app, ranked by priority")
    st.caption("priority = share of this app's reviews mentioning the theme × average severity (1–5)")
    chart = (alt.Chart(neg.head(12))
             .mark_bar(color=BLUE, cornerRadiusEnd=4, height=16)
             .encode(x=alt.X("priority:Q", title="priority"),
                     y=alt.Y("theme:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                     tooltip=[alt.Tooltip("theme:N"), alt.Tooltip("frequency:Q", format=".1%", title="% of reviews"),
                              alt.Tooltip("avg_severity:Q", format=".2f", title="avg severity"),
                              alt.Tooltip("avg_stars:Q", format=".1f", title="avg stars"),
                              alt.Tooltip("priority:Q", format=".3f")])
             .properties(height=28 * min(len(neg), 12)))
    st.altair_chart(chart, use_container_width=True)

    pick = st.selectbox("Open a theme to see what users actually wrote", neg["theme"].tolist())
    rows = themes[(themes["app"] == app) & (themes["theme"] == pick)]
    row = neg[neg["theme"] == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews mentioning it", f"{row['frequency']:.1%}")
    c2.metric("Average severity", f"{row['avg_severity']:.2f}")
    c3.metric("Average stars", f"{row['avg_stars']:.1f}")
    st.dataframe(rows[["phrase", "evidence", "impact", "month"]].rename(
        columns={"phrase": "extracted complaint", "evidence": "exact words from the review"}),
        use_container_width=True, hide_index=True)

    st.subheader("What's working")
    pos = pd.DataFrame(data["positive"])
    if len(pos):
        st.dataframe(pos[["theme", "frequency"]].assign(frequency=pos["frequency"].map("{:.1%}".format))
                     .rename(columns={"frequency": "% of reviews"}), hide_index=True, use_container_width=True)

with tab_safety:
    s = metrics["themes"]["safety"][app]
    st.subheader("Safety issues: listed separately, never ranked by the formula")
    st.metric(f"{app.capitalize()} reviews with a safety issue", f"{s['n_reviews']} ({s['share_of_reviews']:.1%})")
    for quote in s["examples"]:
        st.markdown(f"> {quote}")
    st.caption("A review is flagged when the extractor, or the cross-model tie-breaker (OR rule), finds a safety "
               "or harassment issue backed by a verbatim quote.")

with tab_trend:
    st.subheader("Rider complaints about tips/extra money, around the CCPA order")
    st.image(str(ROOT / "analysis" / "figures" / "trend_tipping.png"), use_container_width=True)
    tr = metrics["trend"]["series"]["Tip / extra-money demands"]
    st.markdown(f"Before the order (Mar–Aug), **{tr['pre_order_share']:.1%}** of rider reviews mentioned tip or "
                f"extra-money demands (95% CI {tr['pre_order_ci95'][0]:.1%}–{tr['pre_order_ci95'][1]:.1%}). "
                "The post-order window is only a few weeks with ~35 reviews per month, and overall complaint "
                "share also fell, so **no causal claim**: this is the metric to watch.")
    if app == "captain":
        st.info("The trend chart covers the rider app, where tip and extra-money demands are reported.")

with tab_eval:
    st.subheader("How accurate is it?")
    st.markdown((ROOT / "eval" / "results.md").read_text(encoding="utf-8").split("\n", 1)[1])
    js = metrics.get("judge_spotcheck")
    if js:
        st.caption(f"LLM judge vs human spot-check: {js['n_agree']}/{js['n']} agreed "
                   f"(95% CI {js['agreement_ci95'][0]:.0%}–{js['agreement_ci95'][1]:.0%}).")
    c = metrics["consistency"]["results"]["0.7"]
    st.caption(f"Cross-model consistency (Qwen vs gpt-oss on 50 reviews): {c['aspect_agreement']:.1%} of extracted "
               f"complaints matched; {c['impact_agreement']:.1%} of matched complaints got the same impact label.")

with tab_brief:
    brief = (ROOT / "BRIEF.md").read_text(encoding="utf-8")
    st.download_button("Download BRIEF.md", brief, file_name="BRIEF.md", mime="text/markdown")
    before, _, after = brief.partition("![trend](analysis/figures/trend_tipping.png)")
    st.markdown(before)
    if after:  # Markdown image links don't resolve in Streamlit, so show the chart directly
        st.image(str(ROOT / "analysis" / "figures" / "trend_tipping.png"), use_container_width=True)
        st.markdown(after)
