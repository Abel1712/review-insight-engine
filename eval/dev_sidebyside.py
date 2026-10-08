"""Step 3 prompt check: pinned main model's output on gold_dev, side by side with the gold labels.

Writes eval/dev_sidebyside.md (for Abel's glance) and prints gold items the model missed (best cosine < 0.7).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA, EVAL  # noqa: E402
from embed import embed  # noqa: E402
from extract import PROMPT_VERSION, extract  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bakeoff import gold_items  # noqa: E402


def main() -> None:
    dev = pd.read_csv(EVAL / "gold_dev.csv", encoding="utf-8-sig")
    dev = dev.merge(pd.read_csv(DATA / "sample.csv")[["review_id", "month"]], on="review_id")
    aspects, reviews, stats = extract(dev, verbose=False)
    model = reviews["model"].dropna().iloc[0]

    md = [f"# gold_dev side by side: {model}, prompt {PROMPT_VERSION}", "",
          "| # | app | review | gold (reference) | extracted (impact) |", "|---|---|---|---|---|"]
    misses = []
    for i, r in enumerate(dev.itertuples(), 1):
        mine = aspects[aspects["review_id"] == r.review_id]
        shown = "<br>".join(f"{a.phrase} ({a.impact or 'pos'})" for a in mine.itertuples()) or "*(none)*"
        text = " ".join(str(r.text).split()).replace("|", "/")
        md.append(f"| {i} | {r.app} | {text[:220]} | {r.gold_complaints.replace(';', '<br>')} | {shown} |")
        items = gold_items(r.gold_complaints)
        if items:
            if len(mine):
                sims = embed(items, "STS", verbose=False) @ embed(mine["phrase"].tolist(), "STS", verbose=False).T
                for item, row in zip(items, sims):
                    if row.max() < 0.7:
                        misses.append((i, item, mine["phrase"].iloc[row.argmax()], row.max()))
            else:
                misses += [(i, item, "(nothing extracted)", 0.0) for item in items]
    (EVAL / "dev_sidebyside.md").write_text("\n".join(md), encoding="utf-8")

    from bakeoff import recall_at
    print(f"recall on gold_dev: " + ", ".join(f"@{t}: {v:.3f}" for t, v in recall_at(dev, aspects).items()))
    extra = len(aspects) - sum(len(gold_items(s)) for s in dev["gold_complaints"])
    print(f"{model}: {len(aspects)} phrases for {sum(len(gold_items(s)) for s in dev['gold_complaints'])} gold items "
          f"({extra:+d}); unsure reviews: {int(reviews['unsure'].fillna(False).sum())}; "
          f"impact mix: {aspects['impact'].fillna('pos').value_counts().to_dict()}")
    print(f"\nGold items without a close match (cosine < 0.7): {len(misses)}")
    for i, item, best, s in misses:
        print(f"  #{i:<2} gold: {item:<55} best: {best} ({s:.2f})")
    print("\nWrote eval/dev_sidebyside.md")


if __name__ == "__main__":
    main()
