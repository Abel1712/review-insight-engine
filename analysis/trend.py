"""Step 8: monthly share of tip and price-pressure complaints (rider app), with 95% Wilson intervals.

Framing rule: the post-order window is a few weeks. Never claim the CCPA order caused a change; report the
pre-order level, say post-order data is too thin to call, and name the metric a PM should watch.
Run:  python analysis/trend.py  ->  analysis/figures/trend_tipping.png, metrics.json["trend"]
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CFG, DATA, FIGURES, update_metrics  # noqa: E402
from plotstyle import INK_2, MUTED, SERIES, plt  # noqa: E402

T = CFG["trend"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def monthly(themes: pd.DataFrame, sample: pd.DataFrame, app: str) -> pd.DataFrame:
    n_by_month = sample[sample["app"] == app].groupby("month").size()
    neg = themes[(themes["app"] == app) & (themes["sentiment"] == "neg")]
    rows = []
    for name, pattern in T["series"].items():
        hit = neg[neg["phrase"].str.contains(pattern, case=False, regex=True)]
        k_by_month = hit.groupby("month")["review_id"].nunique()
        for m, n in n_by_month.items():
            k = int(k_by_month.get(m, 0))
            lo, hi = wilson(k, int(n))
            rows.append({"series": name, "month": m, "k": k, "n": int(n), "share": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def other_spikes(themes: pd.DataFrame, sample: pd.DataFrame, app: str) -> dict:
    """Mention rate of every negative theme per month, to check whether a change is part of a broader shift."""
    n = sample[sample["app"] == app].groupby("month").size()
    neg = themes[(themes["app"] == app) & (themes["sentiment"] == "neg")]
    any_neg = neg.groupby("month")["review_id"].nunique().reindex(n.index, fill_value=0) / n
    return {m: round(float(v), 4) for m, v in any_neg.items()}


def plot(df: pd.DataFrame, partial_month: str) -> None:
    months = sorted(df["month"].unique())
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for color, (name, g) in zip(SERIES, df.groupby("series", sort=False)):
        g = g.set_index("month").reindex(months)
        ax.fill_between(x, g["lo"] * 100, g["hi"] * 100, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, g["share"] * 100, color=color, marker="o", label=name)
        ax.annotate(name, (x[-1], g["share"].iloc[-1] * 100), textcoords="offset points", xytext=(8, 0),
                    color=INK_2, fontsize=9, va="center")
    ccpa = date.fromisoformat(T["ccpa_order_date"])
    pos = months.index(ccpa.strftime("%Y-%m")) + (ccpa.day - 1) / 30 - 0.5  # mid-month inside the month slot
    ax.axvline(pos, color=MUTED, linewidth=1.2, linestyle="--")
    ax.annotate("CCPA order: pre-ride tipping\ndark patterns (mid-Sep 2026)", (pos, ax.get_ylim()[1]),
                textcoords="offset points", xytext=(-6, -4), ha="right", va="top", color=INK_2, fontsize=8.5)
    ax.set_xticks(x, [m + (" (partial)" if m == partial_month else "") for m in months], rotation=0, fontsize=8.5)
    ax.set_ylabel("% of rider reviews in the sample")
    ax.set_title("Rider complaints about tips/extra money and fares above the quote, by month")
    n_lo, n_hi = int(df["n"].min()), int(df["n"].max())
    ax.text(0, -0.22, f"Shaded bands: 95% Wilson intervals ({n_lo}-{n_hi} rider reviews per month). Review date is "
            "when the review was posted, not the ride date.", transform=ax.transAxes, color=MUTED, fontsize=7.5)
    ax.legend(loc="upper left")
    ax.set_xlim(-0.4, len(months) - 0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "trend_tipping.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    themes = pd.read_csv(DATA / "themes.csv")
    sample = pd.read_csv(DATA / "sample_final.csv")
    partial = pd.Timestamp.now(tz=CFG["collect"]["timezone"]).strftime("%Y-%m")
    df = monthly(themes, sample, "rider")
    plot(df, partial)

    order_month = date.fromisoformat(T["ccpa_order_date"]).strftime("%Y-%m")
    summary = {}
    for name, g in df.groupby("series", sort=False):
        pre = g[g["month"] < order_month]
        summary[name] = {
            "by_month": {r.month: {"k": r.k, "n": r.n, "share": round(r.share, 4), "ci95": [round(r.lo, 4), round(r.hi, 4)]}
                         for r in g.itertuples()},
            "pre_order_share": round(pre["k"].sum() / pre["n"].sum(), 4),
            "pre_order_ci95": [round(v, 4) for v in wilson(int(pre["k"].sum()), int(pre["n"].sum()))],
            "pre_order_months": f"{pre['month'].min()}..{pre['month'].max()}",
        }
        print(f"{name}: pre-order ({summary[name]['pre_order_months']}) {summary[name]['pre_order_share']:.1%} "
              f"[{summary[name]['pre_order_ci95'][0]:.1%}-{summary[name]['pre_order_ci95'][1]:.1%}]; "
              + ", ".join(f"{m}: {v['k']}/{v['n']}" for m, v in summary[name]["by_month"].items()))
    spikes = other_spikes(themes, sample, "rider")
    print("Share of rider reviews with ANY complaint, by month (context for spikes):", spikes)
    update_metrics(trend={"app": "rider", "series_regex": T["series"], "ccpa_order_date": T["ccpa_order_date"],
                          "partial_month": partial, "series": summary, "any_complaint_share_by_month": spikes,
                          "app_version_split": "not done: 45 reviews per month is too few to split further",
                          "framing": "post-order window too short to call; no causal claim"})
    print("Saved analysis/figures/trend_tipping.png")


if __name__ == "__main__":
    main()
