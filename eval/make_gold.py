"""Step 2b: build the gold set Abel labels by hand, then split it into dev (tuning) and test (final scores).

  python eval/make_gold.py          -> eval/gold_100.csv (50 rider + 50 captain, gold_complaints empty)
  python eval/make_gold.py --split  -> after labeling: eval/gold_dev.csv (30) + eval/gold_test.csv (70)

gold_complaints format: short phrases separated by ";"  e.g.  captain demanded tip; app crashed at payment
Write "none" for a review with no complaint or praise.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA, EVAL, SEED, update_metrics  # noqa: E402

GOLD = EVAL / "gold_100.csv"
PER_APP, DEV_PER_APP = 50, 15


def make() -> None:
    if GOLD.exists():
        labeled = pd.read_csv(GOLD, encoding="utf-8-sig")["gold_complaints"].notna().sum()
        if labeled:
            sys.exit(f"{GOLD.name} already has {labeled} labeled rows. Refusing to overwrite your work.")
    smp = pd.read_csv(DATA / "sample.csv")
    gold = (smp.groupby("app").sample(n=PER_APP, random_state=SEED)
               .sample(frac=1, random_state=SEED))  # shuffle so rider/captain are interleaved
    gold = gold[["review_id", "app", "text"]].assign(gold_complaints="")
    gold.to_csv(GOLD, index=False, encoding="utf-8-sig")  # -sig so Excel shows Hindi/emoji correctly
    print(f"Wrote {GOLD} ({len(gold)} rows: {gold['app'].value_counts().to_dict()})")


def split() -> None:
    gold = pd.read_csv(GOLD, encoding="utf-8-sig")
    missing = gold["gold_complaints"].isna() | (gold["gold_complaints"].astype(str).str.strip() == "")
    if missing.any():
        sys.exit(f"{missing.sum()} rows still unlabeled (write 'none' if a review has nothing). Rows: "
                 f"{list(gold.index[missing] + 2)[:20]}")  # +2 = spreadsheet row numbers
    dev = gold.groupby("app").sample(n=DEV_PER_APP, random_state=SEED)
    test = gold.drop(dev.index)
    dev.to_csv(EVAL / "gold_dev.csv", index=False, encoding="utf-8-sig")
    test.to_csv(EVAL / "gold_test.csv", index=False, encoding="utf-8-sig")
    n_items = gold["gold_complaints"].map(lambda s: 0 if str(s).strip().lower() == "none"
                                          else len([p for p in str(s).split(";") if p.strip()]))
    labeled_by = gold["labeled_by"].iloc[0] if "labeled_by" in gold else "Abel (by hand)"
    update_metrics(gold={"labeled_by": labeled_by, "n_reviews": len(gold), "n_dev": len(dev), "n_test": len(test),
                         "n_gold_items": int(n_items.sum()),
                         "share_reviews_2plus_gold_items": round(float((n_items >= 2).mean()), 4)})
    print(f"dev: {len(dev)} {dev['app'].value_counts().to_dict()} | test: {len(test)} "
          f"{test['app'].value_counts().to_dict()} | gold items: {n_items.sum()}")


if __name__ == "__main__":
    split() if "--split" in sys.argv else make()
