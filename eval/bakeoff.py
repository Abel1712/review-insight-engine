"""Step 3 model bake-off: every configured candidate extracts the 30 gold_dev reviews; score and rank them.

Main = best Groq model. Secondary = best model from a different family than the main (for independence).
Results: eval/model_bakeoff.md and metrics.json["bakeoff"]. Re-runs are served from the LLM cache.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CFG, DATA, EVAL, update_metrics  # noqa: E402
from embed import embed  # noqa: E402
from extract import extract  # noqa: E402
from verify import evidence_found  # noqa: E402

THRESHOLDS = (0.6, 0.7, 0.8)


def family(model: str) -> str:
    return model.split("/")[0] if "/" in model else model.split("-")[0]  # openai, qwen, gemini


def gold_items(s: str) -> list[str]:
    return [] if str(s).strip().lower() == "none" else [p.strip() for p in str(s).split(";") if p.strip()]


def recall_at(dev: pd.DataFrame, aspects: pd.DataFrame) -> dict[float, float]:
    """Share of gold items matched (cosine >= t) by some extracted phrase from the SAME review."""
    hits = {t: 0 for t in THRESHOLDS}
    total = 0
    for r in dev.itertuples():
        items = gold_items(r.gold_complaints)
        phrases = aspects.loc[aspects["review_id"] == r.review_id, "phrase"].tolist() if len(aspects) else []
        total += len(items)
        if not items or not phrases:
            continue
        best = (embed(items, "STS", verbose=False) @ embed(phrases, "STS", verbose=False).T).max(axis=1)
        for t in THRESHOLDS:
            hits[t] += int((best >= t).sum())
    return {t: hits[t] / total for t in THRESHOLDS}


def main() -> None:
    dev = pd.read_csv(EVAL / "gold_dev.csv", encoding="utf-8-sig")
    dev = dev.merge(pd.read_csv(DATA / "sample.csv")[["review_id", "month"]], on="review_id")
    texts = dict(zip(dev["review_id"], dev["text"]))
    n_gold = sum(len(gold_items(s)) for s in dev["gold_complaints"])

    candidates = {c["model"]: c["provider"] for role in ("main", "secondary") for c in CFG["llm"]["roles"][role]}
    rows, outputs = [], {}
    for model, provider in candidates.items():
        print(f"\n=== {provider}/{model} ===")
        start = time.time()
        try:
            aspects, reviews, stats = extract(dev, model=model, verbose=True)
        except Exception as exc:  # a candidate failing is a result, not a crash
            print(f"FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            rows.append({"model": model, "provider": provider, "family": family(model), "error": type(exc).__name__})
            continue
        seconds = time.time() - start
        outputs[model] = aspects
        ev = [evidence_found(a.evidence, texts[a.review_id]) for a in aspects.itertuples()] if len(aspects) else []
        rec = recall_at(dev, aspects)
        rows.append({
            "model": model, "provider": provider, "family": family(model),
            **{f"recall@{t}": round(v, 3) for t, v in rec.items()},
            "first_pass_valid": round(stats["first_pass_valid"] / stats["n_reviews"], 3),
            "failed": len(stats["failed"]),
            "evidence_found": round(float(np.mean(ev)), 3) if ev else None,
            "phrases": len(aspects), "unsure_reviews": int(reviews["unsure"].fillna(False).sum()),
            "seconds": round(seconds, 1),
        })

    table = pd.DataFrame(rows)
    ok = table[table.get("error").isna()] if "error" in table else table
    ok = ok.sort_values(["recall@0.7", "evidence_found", "first_pass_valid"], ascending=False)
    main_pick = ok[ok["provider"] == "groq"].iloc[0]
    secondary_pick = ok[ok["family"] != main_pick["family"]].iloc[0]

    print(f"\nGold items in gold_dev: {n_gold}")
    print(ok.drop(columns=[c for c in ("error",) if c in ok]).to_string(index=False))
    print(f"\nRanking stable across thresholds? "
          + ", ".join(f"@{t}: {list(ok.sort_values(f'recall@{t}', ascending=False)['model'])}" for t in THRESHOLDS))
    print(f"\n-> MAIN: {main_pick['model']}   SECONDARY (different family): {secondary_pick['model']}")

    md = ["# Model bake-off (gold_dev, 30 reviews)", "",
          f"Gold labels: {n_gold} items, LLM-generated (see gold_labeling_notes.md). "
          "Recall = share of gold items matched by an extracted phrase from the same review (cosine >= t).", "",
          ok.drop(columns=[c for c in ("error",) if c in ok]).to_markdown(index=False), "",
          f"**Main:** {main_pick['model']} (best Groq model by recall@0.7, then evidence accuracy)  ",
          f"**Secondary:** {secondary_pick['model']} (best model from a different family)"]
    failed = table[table["error"].notna()] if "error" in table else table.iloc[0:0]
    if len(failed):
        md += ["", "**Could not be scored:** " + ", ".join(f"{r.model} ({r.error})" for r in failed.itertuples())]
    (EVAL / "model_bakeoff.md").write_text("\n".join(md), encoding="utf-8")
    update_metrics(bakeoff={"n_dev_reviews": len(dev), "n_gold_items": n_gold,
                            "results": ok.drop(columns=[c for c in ("error",) if c in ok]).to_dict("records"),
                            "main": main_pick["model"], "secondary": secondary_pick["model"],
                            "not_scored": failed["model"].tolist() if len(failed) else []})
    return outputs


if __name__ == "__main__":
    main()
