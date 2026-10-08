"""Loads config.yaml + .env and defines shared paths, so every script reads settings from one place."""
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Windows without Developer Mode can't symlink; HF cache still works, so silence the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

with open(ROOT / "config.yaml", encoding="utf-8-sig") as f:  # -sig: tolerate a BOM from Windows editors
    CFG = yaml.safe_load(f)

SEED = CFG["seed"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "analysis" / "figures"
EVAL = ROOT / "eval"
METRICS = RESULTS / "metrics.json"

for d in (RAW, CACHE, RESULTS, FIGURES, EVAL):
    d.mkdir(parents=True, exist_ok=True)


def update_metrics(**values) -> dict:
    """Merge values into results/metrics.json, the single source of truth for every reported number."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    metrics.update(values)
    METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print("Root:          ", ROOT)
    print("Seed:          ", SEED)
    print("Rider app ID:  ", CFG["apps"]["rider"]["app_id"])
    print("Captain app ID:", CFG["apps"]["captain"]["app_id"])
    print("Groq key set:  ", bool(GROQ_API_KEY))
    print("HF token set:  ", bool(HF_TOKEN))
