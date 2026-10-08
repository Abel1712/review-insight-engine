"""Step 4 (QualIT step B): hallucination checks on extracted aspects.

1. Evidence check (deterministic): the aspect's `evidence` quote must actually appear in the review.
2. Meaning check: the phrase must be close in meaning to its own evidence. Threshold = 5th percentile of
   phrase-evidence similarity among known-good gold_dev aspects (those matching a gold item).

The evidence check is also used by the Step 3 bake-off to score each candidate model.
Run:  python src/verify.py   ->  data/aspects.csv
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from config import CFG, DATA, EVAL, update_metrics

V = CFG["verify"]
GOOD_MATCH = 0.7  # an extracted dev aspect counts as "known good" if it matches a gold item at cosine >= this


def normalize(text: str) -> str:
    text = str(text).lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def evidence_found(evidence: str, review: str, min_ratio: float = V["evidence_min_ratio"]) -> bool:
    """True if the evidence appears in the review (fuzzy, to tolerate tiny copy slips like a dropped comma)."""
    e, r = normalize(evidence), normalize(review)
    if not e:
        return False
    if len(e) > len(r) * 1.1:          # "evidence" longer than the review itself: compare whole strings
        return fuzz.ratio(e, r) >= min_ratio
    return fuzz.partial_ratio(e, r) >= min_ratio  # best-matching substring of the review


def phrase_evidence_similarity(aspects: pd.DataFrame) -> np.ndarray:
    from embed import embed  # imported here so the bake-off can use evidence_found without loading the model
    p = embed(aspects["phrase"].tolist(), "STS", verbose=False)
    e = embed(aspects["evidence"].astype(str).tolist(), "STS", verbose=False)
    return (p * e).sum(axis=1)  # row-wise cosine (unit vectors)


def meaning_threshold(aspects: pd.DataFrame) -> tuple[float, int]:
    """5th percentile of phrase-evidence similarity among dev aspects that match a gold item."""
    from embed import embed
    dev = pd.read_csv(EVAL / "gold_dev.csv", encoding="utf-8-sig")
    good = []
    for r in dev.itertuples():
        items = [] if str(r.gold_complaints).strip().lower() == "none" else \
            [p.strip() for p in str(r.gold_complaints).split(";") if p.strip()]
        mine = aspects[aspects["review_id"] == r.review_id]
        if not items or mine.empty:
            continue
        best = (embed(mine["phrase"].tolist(), "STS", verbose=False) @ embed(items, "STS", verbose=False).T).max(axis=1)
        good.append(mine[best >= GOOD_MATCH])
    good = pd.concat(good)
    return round(float(np.percentile(phrase_evidence_similarity(good), 5)), 4), len(good)


def main() -> None:
    raw = pd.read_csv(DATA / "aspects_raw.csv")
    texts = dict(zip(*pd.read_csv(DATA / "sample_final.csv")[["review_id", "text"]].values.T))

    raw["evidence_ok"] = [evidence_found(a.evidence, texts[a.review_id]) for a in raw.itertuples()]
    raw["phrase_evidence_sim"] = phrase_evidence_similarity(raw).round(4)
    threshold, n_good = meaning_threshold(raw[raw["evidence_ok"]])
    raw["meaning_ok"] = raw["phrase_evidence_sim"] >= threshold
    # Safety is fail-safe: a SAFETY aspect with genuine evidence is never dropped by the (softer) meaning check.
    raw.loc[raw["is_safety"] & raw["evidence_ok"], "meaning_ok"] = True

    dropped_ev = raw[~raw["evidence_ok"]]
    dropped_mean = raw[raw["evidence_ok"] & ~raw["meaning_ok"]]
    kept = raw[raw["evidence_ok"] & raw["meaning_ok"]].drop(columns=["evidence_ok", "meaning_ok"])
    kept.to_csv(DATA / "aspects.csv", index=False, encoding="utf-8")

    print(f"aspects in: {len(raw)} | dropped by evidence check: {len(dropped_ev)} | "
          f"dropped by meaning check: {len(dropped_mean)} (threshold {threshold:+.3f}, from {n_good} known-good dev aspects)"
          f" | kept: {len(kept)}")
    print("\nExamples dropped by the EVIDENCE check (quote not found in review):")
    for a in dropped_ev.head(5).itertuples():
        print(f"  [{a.app}] {a.phrase!r}  evidence {str(a.evidence)[:60]!r}\n      review: {texts[a.review_id][:90]!r}")
    print("\nExamples dropped by the MEANING check (phrase doesn't match its own quote):")
    for a in dropped_mean.nsmallest(5, "phrase_evidence_sim").itertuples():
        print(f"  {a.phrase_evidence_sim:+.2f} [{a.app}] {a.phrase!r}  <-  {str(a.evidence)[:70]!r}")

    update_metrics(verify={
        "n_in": len(raw), "n_kept": len(kept),
        "evidence_min_ratio": V["evidence_min_ratio"], "n_dropped_evidence": len(dropped_ev),
        "meaning_threshold": threshold, "meaning_threshold_from_n_good_dev_aspects": n_good,
        "n_dropped_meaning": len(dropped_mean),
        "dropped_examples": [{"phrase": a.phrase, "evidence": str(a.evidence)[:120], "check": c}
                             for c, d in (("evidence", dropped_ev), ("meaning", dropped_mean))
                             for a in d.head(5).itertuples()],
    })


if __name__ == "__main__":
    main()
