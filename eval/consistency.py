"""Cross-model consistency test (plan 2b.9): main vs secondary model on 50 random NON-gold reviews.

Per review, phrases are paired one-to-one by meaning (greedy, cosine >= t). Reports:
  aspect agreement = 2 x matched pairs / (main phrases + secondary phrases)
  impact agreement = share of matched negative pairs with the same impact label
Plan rule: below ~80% agreement -> improve the prompt (on gold_dev only) before the full run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA, EVAL, SEED, update_metrics  # noqa: E402
from embed import embed  # noqa: E402
from extract import PROMPT_VERSION, consistency_set, extract  # noqa: E402

THRESHOLDS = (0.6, 0.7, 0.8)


def pair_up(a: pd.DataFrame, b: pd.DataFrame, t: float) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing of phrases (highest similarity first) above threshold t."""
    if a.empty or b.empty:
        return []
    sims = embed(a["phrase"].tolist(), "STS", verbose=False) @ embed(b["phrase"].tolist(), "STS", verbose=False).T
    pairs, used_a, used_b = [], set(), set()
    for i, j in sorted(np.ndindex(sims.shape), key=lambda ij: -sims[ij]):
        if sims[i, j] < t:
            break
        if i not in used_a and j not in used_b:
            pairs.append((i, j)); used_a.add(i); used_b.add(j)
    return pairs


def main() -> None:
    gold_ids = set(pd.read_csv(EVAL / "gold_100.csv", encoding="utf-8-sig")["review_id"])
    test = consistency_set(pd.read_csv(DATA / "sample.csv"), gold_ids)  # same 50 reviews the final run includes

    main_asp, main_rev, _ = extract(test, role="main")
    sec_asp, sec_rev, _ = extract(test, role="secondary")
    m_model, s_model = main_rev["model"].dropna().iloc[0], sec_rev["model"].dropna().iloc[0]

    results = {}
    for t in THRESHOLDS:
        matched = total = same_impact = neg_pairs = 0
        for rid in test["review_id"]:
            a = main_asp[main_asp["review_id"] == rid].reset_index(drop=True) if len(main_asp) else pd.DataFrame()
            b = sec_asp[sec_asp["review_id"] == rid].reset_index(drop=True) if len(sec_asp) else pd.DataFrame()
            total += len(a) + len(b)
            for i, j in pair_up(a, b, t):
                matched += 1
                if a.loc[i, "sentiment"] == "neg" and b.loc[j, "sentiment"] == "neg":
                    neg_pairs += 1
                    same_impact += int(a.loc[i, "impact"] == b.loc[j, "impact"])
        results[t] = {"aspect_agreement": round(2 * matched / total, 3) if total else None,
                      "impact_agreement": round(same_impact / neg_pairs, 3) if neg_pairs else None,
                      "matched_pairs": matched, "neg_pairs": neg_pairs}

    both_empty = sum(1 for rid in test["review_id"]
                     if (main_asp["review_id"] == rid).sum() == 0 and (sec_asp["review_id"] == rid).sum() == 0)
    print(f"\n{m_model} ({len(main_asp)} phrases) vs {s_model} ({len(sec_asp)} phrases) on {len(test)} non-gold reviews; "
          f"both returned nothing for {both_empty}")
    for t, r in results.items():
        print(f"  @{t}: aspect agreement {r['aspect_agreement']:.1%}  impact agreement {r['impact_agreement']:.1%} "
              f"({r['matched_pairs']} pairs, {r['neg_pairs']} negative)")
    verdict = "PASS" if results[0.7]["aspect_agreement"] >= 0.8 else "BELOW 80%: improve the prompt on gold_dev"
    print(f"  -> {verdict} (plan rule: >= ~80% at the match threshold)")

    update_metrics(consistency={"main": m_model, "secondary": s_model, "prompt_version": PROMPT_VERSION,
                                "n_reviews": len(test), "results": {str(t): r for t, r in results.items()},
                                "verdict_at_0.7": verdict})


if __name__ == "__main__":
    main()
