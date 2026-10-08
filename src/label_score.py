"""Step 6 (QualIT step D): name each cluster, merge duplicates, and score themes per app.

frequency = % of the app's sampled reviews that mention the theme (each theme counted once per review)
priority  = frequency x avg_severity (negative themes). Safety aspects are listed separately, never ranked.
Run:  python src/label_score.py   ->  data/themes.csv, metrics.json["themes"]
"""
import numpy as np
import pandas as pd
from pydantic import BaseModel
from rapidfuzz import fuzz

from config import CFG, DATA, update_metrics
from embed import embed
from llm import complete_json

L = CFG["label"]

LABEL_PROMPT = """These short phrases come from Rapido app reviews ({app_note}). They were grouped together because
they are similar. Name the shared theme for a product manager.
Vocabulary: in Indian bike-taxi usage "rider" or "driver" usually means the CAPTAIN (the person driving), not the
customer. Customers are "customers" or "passengers". Name the theme from that point of view.
If the phrases cover several issues, name the most common one.
Return JSON: {{"label": "3-6 word theme name", "description": "one sentence describing the theme"}}
Be specific (e.g. "Captains demand extra cash" rather than "Payment issues"). Polarity: {polarity}.
Phrases:
{phrases}"""

APP_NOTE = {None: "rider app (customers) and captain app (drivers)",
            "rider": "the RIDER app, written by customers", "captain": "the CAPTAIN app, written by drivers"}
MIN_APP_PHRASES = 3  # an app-specific theme name needs at least this many of that app's phrases


class ThemeName(BaseModel):
    label: str
    description: str


def name_cluster(phrases: list[str], polarity: str, app: str | None = None) -> tuple[ThemeName, str]:
    prompt = LABEL_PROMPT.format(app_note=APP_NOTE[app], polarity=polarity,
                                 phrases="\n".join(f"- {p}" for p in phrases))
    return complete_json(prompt, ThemeName, role=L["role"], verbose=False)


def label_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """One shared name per cluster (used for merging), plus an app-specific name from that app's own phrases."""
    rows = []
    for cid, g in df.groupby("cluster_id"):
        polarity = "complaints" if cid.startswith("neg") else "praise"
        top = g.nsmallest(L["phrases_per_cluster"], "dist_to_centroid")["phrase"].tolist()
        name, model = name_cluster(top, polarity)
        row = {"cluster_id": cid, "label": name.label, "description": name.description,
               "n_phrases": len(g), "label_model": model}
        for app in ("rider", "captain"):
            ga = g[g["app"] == app]
            if len(ga) >= MIN_APP_PHRASES:
                n_app, _ = name_cluster(ga.nsmallest(L["phrases_per_cluster"], "dist_to_centroid")["phrase"].tolist(),
                                        polarity, app)
                row[f"label_{app}"], row[f"description_{app}"] = n_app.label, n_app.description
            else:
                row[f"label_{app}"], row[f"description_{app}"] = name.label, name.description
        rows.append(row)
    return pd.DataFrame(rows)


def merge_duplicates(df: pd.DataFrame, names: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Merge same-polarity clusters whose NAMES mean the same thing (or whose centroids are unusually close
    relative to all pairs AND whose names are loosely similar). The larger cluster survives; no chain merges."""
    vecs = embed(df["phrase"].tolist(), task="Clustering", verbose=False)
    cent = {cid: vecs[(df["cluster_id"] == cid).values].mean(axis=0) for cid in names["cluster_id"]}
    cent = {cid: v / np.linalg.norm(v) for cid, v in cent.items()}
    names = names.sort_values("n_phrases", ascending=False).reset_index(drop=True)
    name_vec = dict(zip(names["cluster_id"], embed(names["label"].tolist(), "STS", verbose=False)))
    pairs = [(a, b) for i, a in names.iterrows() for _, b in names.iloc[i + 1:].iterrows()
             if a.cluster_id[:3] == b.cluster_id[:3]]
    all_cos = [float(cent[a.cluster_id] @ cent[b.cluster_id]) for a, b in pairs]
    cut = float(np.percentile(all_cos, L["merge_centroid_pct"])) if all_cos else 1.0
    target, merges = {cid: cid for cid in names["cluster_id"]}, []
    for (a, b), cos in zip(pairs, all_cos):
        if target[a.cluster_id] != a.cluster_id or target[b.cluster_id] != b.cluster_id:
            continue  # no chains: both must still be unmerged roots
        label_cos = float(name_vec[a.cluster_id] @ name_vec[b.cluster_id])
        ratio = fuzz.token_set_ratio(a.label.lower(), b.label.lower())
        if (label_cos >= L["merge_label_cosine"] or ratio >= L["merge_label_ratio"]
                or (cos >= cut and label_cos >= L["merge_centroid_min_label_cosine"])):
            target[b.cluster_id] = a.cluster_id
            merges.append({"merged": f"{b.cluster_id} '{b.label}'", "into": f"{a.cluster_id} '{a.label}'",
                           "label_cosine": round(label_cos, 3), "label_ratio": ratio,
                           "centroid_cosine": round(cos, 3), "centroid_cut": round(cut, 3)})
    df = df.assign(theme_id=df["cluster_id"].map(target))
    labels = names.set_index("cluster_id")
    df["theme_shared"] = df["theme_id"].map(labels["label"])  # one name across apps (for comparison)
    # Display name per row = the app-specific name of its (surviving) theme, from that app's own phrases.
    df["theme"] = [labels.loc[t, f"label_{a}"] for t, a in zip(df["theme_id"], df["app"])]
    df["theme_description"] = [labels.loc[t, f"description_{a}"] for t, a in zip(df["theme_id"], df["app"])]
    return df, merges


def score(themes: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Per app: frequency (% of that app's sampled reviews), avg severity, avg stars, priority."""
    stars = sample.set_index("review_id")["rating"]
    out = {}
    for app, n_reviews in sample.groupby("app").size().items():
        t = themes[themes["app"] == app]
        rows = []
        for (theme, sentiment), g in t.groupby(["theme", "sentiment"]):
            reviews = g["review_id"].unique()
            freq = len(reviews) / n_reviews
            sev = float(g["severity"].mean()) if sentiment == "neg" else None
            avg_stars = float(stars.reindex(reviews).mean())
            rows.append({"theme": theme, "sentiment": sentiment, "n_reviews": len(reviews),
                         "frequency": round(freq, 4), "avg_severity": round(sev, 2) if sev else None,
                         "priority": round(freq * sev, 4) if sev else None, "avg_stars": round(avg_stars, 2),
                         "flag_high_severity_high_stars": bool(sev and sev >= L["flag_min_severity"]
                                                               and avg_stars >= L["flag_min_stars"]),
                         "examples": g.nsmallest(2, "dist_to_centroid")["evidence"].astype(str).str[:140].tolist()})
        r = pd.DataFrame(rows)
        neg = r[r["sentiment"] == "neg"].sort_values("priority", ascending=False)
        pos = r[r["sentiment"] == "pos"].sort_values("frequency", ascending=False)
        top_by_priority = neg["theme"].head(5).tolist()
        top_by_frequency = neg.sort_values("frequency", ascending=False)["theme"].head(5).tolist()
        out[app] = {"n_reviews": int(n_reviews), "negative": neg.to_dict("records"), "positive": pos.to_dict("records"),
                    "robustness_top5_overlap_priority_vs_frequency": len(set(top_by_priority) & set(top_by_frequency)),
                    "top5_by_priority": top_by_priority, "top5_by_frequency": top_by_frequency}
    return out


def safety_section(themes: pd.DataFrame, sample: pd.DataFrame) -> dict:
    s = themes[themes["is_safety"]]
    return {app: {"n_aspects": int((s["app"] == app).sum()),
                  "n_reviews": int(s.loc[s["app"] == app, "review_id"].nunique()),
                  "share_of_reviews": round(s.loc[s["app"] == app, "review_id"].nunique() / n, 4),
                  "examples": s.loc[s["app"] == app, "evidence"].astype(str).str[:160].head(5).tolist()}
            for app, n in sample.groupby("app").size().items()}


def main() -> None:
    df = pd.read_csv(DATA / "aspects_clustered.csv")
    sample = pd.read_csv(DATA / "sample_final.csv")
    names = label_clusters(df)
    themes, merges = merge_duplicates(df, names)
    themes.to_csv(DATA / "themes.csv", index=False, encoding="utf-8")
    scores, safety = score(themes, sample), safety_section(themes, sample)

    for app, s in scores.items():
        print(f"\n=== {app.upper()} ({s['n_reviews']} reviews): top complaint themes by priority ===")
        for r in s["negative"][:8]:
            flag = "  <- high severity but high stars" if r["flag_high_severity_high_stars"] else ""
            print(f"  {r['priority']:.3f} = {r['frequency']:5.1%} x {r['avg_severity']:.2f}  "
                  f"({r['avg_stars']:.1f} stars)  {r['theme']}{flag}")
        print(f"  robustness: {s['robustness_top5_overlap_priority_vs_frequency']}/5 of the top 5 are the same "
              "whether ranked by priority or by frequency alone")
        print("  what's working: " + "; ".join(f"{r['theme']} ({r['frequency']:.1%})" for r in s["positive"][:4]))
    print("\nSafety (never ranked):", {a: f"{v['n_reviews']} reviews ({v['share_of_reviews']:.1%})" for a, v in safety.items()})
    print(f"Merges: {len(merges)}" + "".join(f"\n  {m['merged']} -> {m['into']}" for m in merges))

    update_metrics(themes={"n_clusters": int(names.shape[0]), "n_themes": int(themes["theme_id"].nunique()),
                           "n_negative_themes": int(themes.loc[themes["sentiment"] == "neg", "theme_id"].nunique()),
                           "merges": merges, "by_app": scores, "safety": safety,
                           "label_model": names["label_model"].iloc[0]})


if __name__ == "__main__":
    main()
