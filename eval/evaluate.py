"""Step 7: evaluate our method on the 70 held-out gold_test reviews (never used for tuning).

Phrase level: recall (gold items found) and precision (phrases that match gold; plus a judge pass on unmatched
phrases, since the gold labels may miss real issues). Theme level: a gold item is covered if it matches one of
the theme labels of that review's phrases (comparable with BERTopic, which gives each review one topic).
Writes eval/judge_spotcheck.csv (20 judge decisions for Abel) and metrics.json["evaluation"]["ours"].
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bakeoff import gold_items  # noqa: E402
from config import CFG, DATA, EVAL, SEED, update_metrics  # noqa: E402
from matching import JUDGE_LOG, judge_real, match_many  # noqa: E402


def recall_by_review(test: pd.DataFrame, mats: list) -> pd.DataFrame:
    rows = []
    for r, m in zip(test.itertuples(), mats):
        n = len(gold_items(r.gold_complaints))
        found = int(m.any(axis=1).sum()) if n and m.size else 0
        rows.append({"review_id": r.review_id, "n_gold": n, "found": found})
    return pd.DataFrame(rows)


def summarize(rec: pd.DataFrame) -> dict:
    multi = rec[rec["n_gold"] >= 2]
    return {"recall": round(float(rec["found"].sum() / rec["n_gold"].sum()), 4),
            "recall_multi_complaint_reviews": round(float(multi["found"].sum() / multi["n_gold"].sum()), 4),
            "n_gold_items": int(rec["n_gold"].sum()), "n_multi_reviews": len(multi)}


def main() -> None:
    test = pd.read_csv(EVAL / "gold_test.csv", encoding="utf-8-sig")
    themes = pd.read_csv(DATA / "themes.csv")
    texts = dict(zip(test["review_id"], test["text"]))

    phrase_cases, theme_cases = [], []
    for r in test.itertuples():
        mine = themes[themes["review_id"] == r.review_id]
        phrase_cases.append((gold_items(r.gold_complaints), mine["phrase"].tolist()))
        theme_cases.append((gold_items(r.gold_complaints), sorted(mine["theme"].dropna().unique().tolist())))

    phrase_mats = match_many(phrase_cases, "phrase")
    theme_mats = match_many(theme_cases, "theme")

    # Precision: matched phrases / all phrases; then a judge pass on the unmatched ones.
    n_phrases = sum(len(c[1]) for c in phrase_cases)
    matched = sum(int(m.any(axis=0).sum()) for m in phrase_mats if m.size)
    unmatched = [(texts[r.review_id], p) for r, (g, ps), m in zip(test.itertuples(), phrase_cases, phrase_mats)
                 for j, p in enumerate(ps) if not (m.size and m[:, j].any())]
    real = sum(judge_real(unmatched)) if unmatched else 0

    ours = {"phrase_level": summarize(recall_by_review(test, phrase_mats)),
            "theme_level": summarize(recall_by_review(test, theme_mats)),
            "precision_strict": round(matched / n_phrases, 4) if n_phrases else None,
            "precision_with_judge": round((matched + real) / n_phrases, 4) if n_phrases else None,
            "n_phrases": n_phrases, "n_unmatched_judged_real": int(real), "n_judge_decisions": len(JUDGE_LOG)}

    log = pd.DataFrame(JUDGE_LOG)
    if len(log):
        spot = log.sample(n=min(CFG["evaluate"]["spotcheck_n"], len(log)), random_state=SEED)
        spot.assign(abel_agrees="").to_csv(EVAL / "judge_spotcheck.csv", index=False, encoding="utf-8-sig")
        from spotcheck import XLSX, make as make_spotcheck
        if not XLSX.exists():  # never overwrite a sheet Abel may already have filled in
            make_spotcheck("judge")
    update_metrics(evaluation={"gold_labeled_by": "claude (AI), not hand-labeled", "n_test_reviews": len(test),
                               "match_high": CFG["evaluate"]["match_high"], "match_low": CFG["evaluate"]["match_low"],
                               "ours": ours})
    print(f"OUR METHOD on {len(test)} gold_test reviews:")
    for k, v in ours.items():
        print(f"  {k}: {v}")
    print(f"Wrote eval/judge_spotcheck.csv ({min(CFG['evaluate']['spotcheck_n'], len(log))} decisions for Abel)")


if __name__ == "__main__":
    main()
