"""Step 1: scrape Play Store reviews for both Rapido apps, newest first, dropping personal fields immediately.

Run:  python src/collect.py            (skips an app whose raw file already exists)
      python src/collect.py --force    (re-scrape)
"""
import random
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from google_play_scraper import Sort, reviews
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CFG, RAW, update_metrics

C = CFG["collect"]
TZ = ZoneInfo(C["timezone"])
START = datetime.strptime(C["start_month"], "%Y-%m").replace(tzinfo=TZ)  # first moment of the start month, IST
KEEP = ["reviewId", "content", "score", "at", "reviewCreatedVersion"]    # the ONLY fields that ever reach disk


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=4, max=60), reraise=True)
def _fetch_page(app_id: str, token):
    return reviews(app_id, lang=C["lang"], country=C["country"], sort=Sort.NEWEST,
                   count=C["batch_size"], continuation_token=token)


def _clean_record(r: dict, app: str) -> dict:
    """Privacy filter + IST month, applied in memory before anything is saved."""
    kept = {k: r.get(k) for k in KEEP}
    at_ist = kept.pop("at").astimezone(TZ)  # library gives naive local time; astimezone() reads it as local
    return {
        "review_id": kept["reviewId"],
        "app": app,
        "text": kept["content"],
        "rating": kept["score"],
        "at": at_ist.isoformat(),
        "month": at_ist.strftime("%Y-%m"),
        "app_version": kept["reviewCreatedVersion"],
    }


def _coverage_report(rows: list[dict], app: str) -> None:
    newest = datetime.fromisoformat(rows[0]["at"])
    oldest = datetime.fromisoformat(rows[-1]["at"])
    days = max((newest - oldest).total_seconds() / 86400, 1 / 24)
    per_day = len(rows) / days
    needed = per_day * (newest - START).total_seconds() / 86400
    print(f"  [coverage] {len(rows)} reviews span {days:.1f} days (~{per_day:,.0f}/day). "
          f"Reaching {C['start_month']} needs ~{needed:,.0f} reviews (cap {C['max_reviews_per_app']:,}).")
    if needed > C["max_reviews_per_app"]:
        reach = newest.timestamp() - C["max_reviews_per_app"] / per_day * 86400
        print(f"  [coverage] WARNING: the cap will stop {app} around "
              f"{datetime.fromtimestamp(reach, TZ):%Y-%m-%d}, short of {C['start_month']}.")


def collect_app(app: str, app_id: str) -> pd.DataFrame:
    rows, token, page, checked = [], None, 0, False
    stop_reason = "store ran out of reviews"
    print(f"\n=== {app} ({app_id}) ===")
    while True:
        batch, token = _fetch_page(app_id, token)
        page += 1
        rows.extend(_clean_record(r, app) for r in batch)  # raw `batch` (with names) is never stored
        if not batch:
            break
        oldest = datetime.fromisoformat(rows[-1]["at"])
        print(f"  page {page:3d}: {len(rows):6,} reviews, oldest {oldest:%Y-%m-%d %H:%M}")

        if not checked and len(rows) >= C["coverage_check_after"]:
            _coverage_report(rows, app)
            checked = True
        if oldest < START:
            stop_reason = f"passed start month {C['start_month']}"
            break
        if len(rows) >= C["max_reviews_per_app"]:
            stop_reason = f"hit cap of {C['max_reviews_per_app']:,}"
            break
        if token is None or getattr(token, "token", None) is None:
            break
        time.sleep(random.uniform(*C["delay_seconds"]))

    df = pd.DataFrame(rows)
    print(f"  stopped: {stop_reason}")
    df.attrs["stop_reason"] = stop_reason
    return df


def main(force: bool) -> None:
    summary = {}
    frames = []
    for app, spec in CFG["apps"].items():
        path = RAW / f"{app}.csv"
        if path.exists() and not force:
            print(f"\n=== {app}: using existing {path.name} (pass --force to re-scrape) ===")
            df = pd.read_csv(path)
            stop_reason = "loaded from existing file"
        else:
            df = collect_app(app, spec["app_id"])
            stop_reason = df.attrs["stop_reason"]
            df.to_csv(path, index=False, encoding="utf-8")
        frames.append(df)
        summary[app] = {"app_id": spec["app_id"], "n_reviews": len(df), "oldest": df["at"].min(),
                        "newest": df["at"].max(), "stop_reason": stop_reason}

    all_df = pd.concat(frames)
    table = all_df.pivot_table(index="month", columns="app", values="review_id", aggfunc="count", fill_value=0)
    table = table.sort_index(ascending=False)
    print("\nReviews per month per app (IST months; raw, before cleaning):")
    print(table.to_string())

    update_metrics(collect={
        "collected_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "lang": C["lang"], "country": C["country"], "timezone": C["timezone"],
        "apps": summary,
        "raw_reviews_per_month": {app: {m: int(n) for m, n in table[app].items() if n} for app in table.columns},
    })
    print("\nSaved data/raw/*.csv and collect stats in results/metrics.json")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
