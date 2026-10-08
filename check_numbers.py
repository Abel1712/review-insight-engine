"""Never invent numbers: every number in BRIEF.md and README.md must exist in results/metrics.json.

Ignored: text inside quotes (review excerpts), code blocks, links/image paths, years, list/heading numbers, and a
small allowlist of fixed constants (the 1-5 severity scale, "per 1,000 rides", "24 h").
Run:  python check_numbers.py   (exit code 1 if anything is unmatched)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOW = {1.0, 5.0, 1000.0, 24.0}
NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s?%)?(?![\w])")


def metric_values(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in metric_values(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in metric_values(x)]
    if isinstance(obj, str):  # numbers stored inside strings, e.g. month labels "2026-03" or "0.56..0.6"
        return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", obj)]
    return []


def strip(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", text)                   # links and images
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|`[^`\n]*`', " ", text)          # quotes and inline code
    text = re.sub(r"(?m)^\s*(#+\s*)?\d+\.\s", " ", text)                  # numbered headings / lists
    text = re.sub(r"\b(19|20)\d{2}(-\d{2})?\b", " ", text)                # years and year-months
    return text


def matches(x: float, decimals: int, is_pct: bool, values: list[float]) -> bool:
    for v in values:
        if is_pct and (round(v * 100, decimals) == x or round(v, decimals) == x):
            return True
        if not is_pct and (round(v, decimals) == x or round(v * 100, decimals) == x):
            return True
    return False


def main() -> None:
    values = metric_values(json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8")))
    bad = []
    for name in ("BRIEF.md", "README.md"):
        path = ROOT / name
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in NUMBER.finditer(strip(line)):
                raw, is_pct = m.group(1), bool(m.group(2))
                x = float(raw.replace(",", ""))
                decimals = len(raw.split(".")[1]) if "." in raw else 0
                if x in ALLOW or matches(x, decimals, is_pct, values):
                    continue
                bad.append(f"{name}:{lineno}: {raw}{'%' if is_pct else ''}  <- {line.strip()[:110]}")
    if bad:
        print(f"{len(bad)} number(s) not found in metrics.json:")
        print("\n".join(f"  {b}" for b in bad))
        sys.exit(1)
    print("OK: every number in BRIEF.md and README.md is backed by results/metrics.json")


if __name__ == "__main__":
    main()
