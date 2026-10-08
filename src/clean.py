"""Step 2: clean the raw scrape and draw a seeded sample of up to N reviews per month per app.

Rules (each counted in metrics.json): duplicate review_id, outside month range, empty text,
no letters (emoji-only), all-generic words, spam (identical text under 5+ review IDs).
No length filter: short reviews often carry real issues.
"""
import re

import pandas as pd

from config import CFG, DATA, RAW, SEED, update_metrics

C = CFG["clean"]
GENERIC = {w.lower() for w in C["generic_words"]}
WORD = re.compile(r"[^\W\d_]+")  # runs of letters in any script (Latin, Devanagari, ...)


def words(text: str) -> list[str]:
    return WORD.findall(re.sub(r"['’]", "", text.lower()))  # "it's" -> "its", not "it" + "s"


def content_free_reason(text) -> str | None:
    """Why a review carries no specific content, or None if it should be kept."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    w = words(text)
    if not w:
        return "no_letters"           # emoji-only, "10/10", "!!!"
    if all(x in GENERIC for x in w):
        return "all_generic"          # "worst app", "bahut accha", "5 star"
    return None


def clean_app(df: pd.DataFrame, app: str, drops: dict, no_issue: dict, spam_log: list) -> pd.DataFrame:
    d = drops[app] = {"raw": len(df)}

    before = len(df)
    df = df.drop_duplicates("review_id")
    d["duplicate_review_id"] = before - len(df)

    before = len(df)
    df = df[(df["month"] >= CFG["collect"]["start_month"]) & (df["month"] <= CFG["collect"]["end_month"])]
    d["outside_month_range"] = before - len(df)

    # Content-free reviews: counted per month (before sampling) as a signal, then removed.
    reasons = df["text"].map(content_free_reason)
    for rule in ("empty", "no_letters", "all_generic"):
        d[rule] = int((reasons == rule).sum())
    totals = df.groupby("month").size()
    removed = df[reasons.notna()].groupby("month").size().reindex(totals.index, fill_value=0)
    no_issue[app] = {m: {"no_specific_issue": int(removed[m]), "total_reviews": int(totals[m]),
                         "share": round(removed[m] / totals[m], 4)} for m in totals.index}
    df = df[reasons.isna()]

    # Spam: identical text (case/space-normalized) under 5+ different review IDs -> keep the first.
    # Only long texts count: short phrases ("good ride", "app not working") repeat naturally at this volume.
    norm = df["text"].str.lower().str.split().str.join(" ")
    counts = norm.map(norm.value_counts())
    repeated = counts >= C["spam_repeat_threshold"]
    long_enough = norm.map(lambda t: len(words(t))) >= C["spam_min_words"]
    is_repeat = norm.duplicated(keep="first") & repeated & long_enough
    for text, n in norm[repeated & long_enough].value_counts().items():
        spam_log.append({"app": app, "text": text[:120], "copies": int(n), "dropped": int(n) - 1})
    d["spam_repeats"] = int(is_repeat.sum())
    d["short_repeats_kept"] = int((norm.duplicated(keep="first") & repeated & ~long_enough).sum())
    df = df[~is_repeat]

    d["clean"] = len(df)
    return df


def sample(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n, parts, info = C["per_month_per_app"], [], {}
    for (app, month), grp in clean.groupby(["app", "month"]):
        take = grp.sample(n=min(n, len(grp)), random_state=SEED)
        parts.append(take)
        info.setdefault(app, {})[month] = {"available": len(grp), "sampled": len(take),
                                           "flag_under_target": len(grp) < n}
    out = pd.concat(parts).sort_values(["app", "month", "at"]).reset_index(drop=True)
    out["n_words"] = out["text"].map(lambda t: len(words(t)))
    return out, info


def main() -> None:
    drops, no_issue, spam_log, frames = {}, {}, [], []
    for app in CFG["apps"]:
        raw = pd.read_csv(RAW / f"{app}.csv")
        frames.append(clean_app(raw, app, drops, no_issue, spam_log))
    clean = pd.concat(frames)
    smp, info = sample(clean)
    cols = ["review_id", "app", "month", "at", "rating", "app_version", "n_words", "text"]
    smp[cols].to_csv(DATA / "sample.csv", index=False, encoding="utf-8")

    print("Rows per rule:")
    print(pd.DataFrame(drops).to_string())

    print("\nContent-free share per month (counted on ALL reviews, before sampling):")
    print(pd.DataFrame({a: {m: f"{v['share']:.1%} of {v['total_reviews']:,}" for m, v in ms.items()}
                        for a, ms in no_issue.items()}).sort_index(ascending=False).to_string())

    print(f"\nSpam texts (>= {C['spam_repeat_threshold']} identical copies): {len(spam_log)}")
    for s in spam_log[:10]:
        print(f"  [{s['app']}] x{s['copies']}: {s['text']!r}")

    partial = pd.Timestamp.now(tz=CFG["collect"]["timezone"]).strftime("%Y-%m")
    table = smp.pivot_table(index="month", columns="app", values="review_id", aggfunc="count", fill_value=0)
    print("\nSample per month per app (target "
          f"{C['per_month_per_app']}; * = under target, {partial} = partial month):")
    for m, row in table.sort_index(ascending=False).iterrows():
        marks = "  ".join(f"{a}: {row[a]:>3}{'*' if info[a][m]['flag_under_target'] else ' '}" for a in table.columns)
        print(f"  {m}  {marks}{'   (partial month)' if m == partial else ''}")
    print(f"\nTotal sampled: {len(smp):,}  ->  data/sample.csv")

    update_metrics(clean={
        "rows_per_rule": drops,
        "no_specific_issue_by_month": no_issue,
        "spam_texts": spam_log,
        "sample_by_month": info,
        "partial_month": partial,
        "n_sample": len(smp),
        "n_sample_by_app": smp["app"].value_counts().to_dict(),
        "share_short_reviews_lt4_words": round(float((smp["n_words"] < 4).mean()), 4),
    })


if __name__ == "__main__":
    main()
