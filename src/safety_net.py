"""Step 2c: safety-net signals for every sampled review: keyword hits + meaning similarity to seed complaints.

Signals never decide anything; in Step 3 they only send a review to the secondary-model tie-breaker.
The similarity threshold is set later from gold_dev (every dev review labeled safety/money must be flagged).
"""
import re

import numpy as np
import pandas as pd

from config import CFG, DATA, EVAL, update_metrics
from embed import embed

S = CFG["safety_net"]


WORDCHAR = r"[\wऀ-ॿ]"  # \w misses Hindi vowel signs, so "डर" would match inside "ऑर्डर"


def _pattern(terms: list[str]) -> re.Pattern:
    """Whole-word / whole-phrase match (multi-word terms tolerate any whitespace). Single words of
    prefix_match_min_len+ letters also match word starts, so "fraud" catches "fraudsters"."""
    parts = []
    for t in terms:
        body = r"\s+".join(map(re.escape, t.lower().split()))
        prefix_ok = " " not in t and len(t) >= S["prefix_match_min_len"]
        parts.append(body + (rf"{WORDCHAR}*" if prefix_ok else ""))
    return re.compile(rf"(?<!{WORDCHAR})(?:" + "|".join(parts) + rf")(?!{WORDCHAR})", re.IGNORECASE)


PATTERNS = {cat: _pattern(terms) for cat, terms in S["keywords"].items()}
SEEDS = [s for cat in S["seeds"] for s in S["seeds"][cat]]
SEED_CATEGORY = [cat for cat in S["seeds"] for _ in S["seeds"][cat]]


def keyword_hits(text: str) -> dict[str, list[str]]:
    hits = {cat: sorted({m.group(0).lower() for m in p.finditer(text)}) for cat, p in PATTERNS.items()}
    return {cat: h for cat, h in hits.items() if h}


def calibrate(smp: pd.DataFrame) -> tuple[float | None, dict]:
    """Threshold from gold_dev: every dev review with a critical gold item must be flagged by keyword OR similarity."""
    path = EVAL / "gold_dev.csv"
    if not path.exists():
        return None, {"status": "gold_dev.csv missing"}
    dev = pd.read_csv(path, encoding="utf-8-sig").merge(smp, on=["review_id", "app"], suffixes=("", "_s"))
    crit = re.compile("|".join(map(re.escape, S["gold_critical_terms"])), re.IGNORECASE)
    dev["critical"] = dev["gold_complaints"].str.contains(crit)
    need = dev[dev["critical"] & ~dev["keyword_flag"]]       # critical reviews keywords didn't catch
    if need.empty:
        return None, {"status": "keywords alone caught every critical dev review"}
    threshold = round(float(need["contrast_score"].min()) - S["threshold_margin"], 4)
    print("Calibration on gold_dev (critical reviews and how they are caught):")
    for r in dev[dev["critical"]].sort_values("contrast_score").itertuples():
        how = "keyword" if r.keyword_flag else f"similarity ({r.contrast_score:+.2f})"
        print(f"  {how:<20} {r.gold_complaints[:80]}")
    print(f"  -> threshold = lowest uncaught score {need['contrast_score'].min():+.3f} - margin "
          f"{S['threshold_margin']} = {threshold:+.3f}\n")
    return threshold, {"n_dev": len(dev), "n_critical_dev": int(dev["critical"].sum()),
                       "n_caught_by_keyword": int((dev["critical"] & dev["keyword_flag"]).sum())}


def main() -> None:
    smp = pd.read_csv(DATA / "sample.csv")

    hits = smp["text"].map(keyword_hits)
    smp["keyword_flag"] = hits.map(bool)
    smp["keyword_categories"] = hits.map(lambda h: ";".join(h))
    smp["keyword_hits"] = hits.map(lambda h: ";".join(w for ws in h.values() for w in ws))

    reviews = embed(smp["text"].tolist(), task="STS")
    seeds = embed(SEEDS, task="STS")
    anchors = embed(S["contrast_anchors"], task="STS")
    sims = reviews @ seeds.T                      # cosine similarity (all vectors are unit length)
    smp["max_seed_sim"] = sims.max(axis=1).round(4)
    smp["nearest_seed"] = [SEEDS[i] for i in sims.argmax(axis=1)]
    smp["nearest_seed_category"] = [SEED_CATEGORY[i] for i in sims.argmax(axis=1)]
    # Contrast: how much closer the review is to a complaint seed than to plain praise ("good driver").
    smp["contrast_score"] = (sims.max(axis=1) - (reviews @ anchors.T).max(axis=1)).round(4)

    threshold, calib = calibrate(smp)
    smp["similarity_flag"] = (smp["contrast_score"] >= threshold) if threshold is not None else pd.NA

    cols = ["review_id", "app", "keyword_flag", "keyword_categories", "keyword_hits",
            "max_seed_sim", "nearest_seed", "nearest_seed_category", "contrast_score", "similarity_flag"]
    smp[cols].to_csv(DATA / "safety_signals.csv", index=False, encoding="utf-8")

    print("Keyword flag rate by app and category:")
    for app, g in smp.groupby("app"):
        cats = {c: f"{g['keyword_categories'].str.contains(c).mean():.1%}" for c in PATTERNS}
        print(f"  {app:<8} any: {g['keyword_flag'].mean():.1%}  {cats}")

    print("\nContrast score percentiles by app:")
    for app, g in smp.groupby("app"):
        q = np.percentile(g["contrast_score"], [10, 25, 50, 75, 90, 99])
        print(f"  {app:<8} p10 {q[0]:+.2f}  p25 {q[1]:+.2f}  p50 {q[2]:+.2f}  p75 {q[3]:+.2f}  p90 {q[4]:+.2f}  p99 {q[5]:+.2f}")

    print("\nHighest contrast scores WITHOUT any keyword (what the similarity signal adds):")
    for r in smp[~smp["keyword_flag"]].nlargest(8, "contrast_score").itertuples():
        print(f"  {r.contrast_score:+.2f} [{r.app}] {r.text[:95]!r}\n        ~ seed: {r.nearest_seed}")

    print("\nKeyword hits with the LOWEST contrast scores (what keywords add):")
    for r in smp[smp["keyword_flag"]].nsmallest(5, "contrast_score").itertuples():
        print(f"  {r.contrast_score:+.2f} [{r.app}] hits={r.keyword_hits!r}  {r.text[:85]!r}")

    update_metrics(safety_net={
        "keyword_flag_rate": {a: round(float(g["keyword_flag"].mean()), 4) for a, g in smp.groupby("app")},
        "keyword_category_rate": {a: {c: round(float(g["keyword_categories"].str.contains(c).mean()), 4)
                                      for c in PATTERNS} for a, g in smp.groupby("app")},
        "n_seeds": len(SEEDS),
        "similarity_threshold": threshold,
        "calibration": calib,
        "similarity_flag_rate": None if threshold is None else
            {a: round(float(g["similarity_flag"].mean()), 4) for a, g in smp.groupby("app")},
        "any_flag_rate": None if threshold is None else
            {a: round(float((g["keyword_flag"] | g["similarity_flag"]).mean()), 4) for a, g in smp.groupby("app")},
    })
    if threshold is not None:
        for app, g in smp.groupby("app"):
            print(f"Flag rate {app:<8} keyword {g['keyword_flag'].mean():.1%} | similarity "
                  f"{g['similarity_flag'].mean():.1%} | either {(g['keyword_flag'] | g['similarity_flag']).mean():.1%}")
    print(f"\nSaved data/safety_signals.csv. Contrast threshold: {threshold}")


if __name__ == "__main__":
    main()
