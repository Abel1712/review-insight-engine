"""Step 8: rider vs captain top complaint themes, side by side, plus LLM-suggested links as HYPOTHESES.

The links ("rider cancellations" <-> "captain fares too low") are co-occurrence-style hypotheses for a PM to test,
never findings. Run:  python analysis/compare_apps.py  ->  analysis/compare_apps.md, metrics.json["compare_apps"]
"""
import json
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import METRICS, ROOT, update_metrics  # noqa: E402
from llm import complete_json  # noqa: E402

PROMPT = """Rapido is a ride-hailing company in India. Below are the top complaint themes from its RIDER app
(customers) and its CAPTAIN app (drivers), each with the % of that app's reviews mentioning it.
Suggest up to 3 pairs where a rider complaint and a captain complaint could be two sides of the same underlying
problem (e.g. riders' "captains cancel rides" vs captains' "fares too low to be worth it"). These are hypotheses
for a product manager to test, not proven causes. Only pair themes that appear in the lists.
Return JSON: {{"pairs": [{{"rider_theme": "...", "captain_theme": "...", "hypothesis": "one sentence",
"how_to_test": "one sentence naming data Rapido could check"}}]}}
RIDER: {rider}
CAPTAIN: {captain}"""


class Pair(BaseModel):
    rider_theme: str
    captain_theme: str
    hypothesis: str
    how_to_test: str


class Pairs(BaseModel):
    pairs: list[Pair]


def main() -> None:
    by_app = json.loads(METRICS.read_text(encoding="utf-8"))["themes"]["by_app"]
    top = {app: by_app[app]["negative"][:5] for app in ("rider", "captain")}
    fmt = lambda rows: "; ".join(f"{r['theme']} ({r['frequency']:.0%})" for r in rows)  # noqa: E731
    result, model = complete_json(PROMPT.format(rider=fmt(top["rider"]), captain=fmt(top["captain"])),
                                  Pairs, role="secondary")
    valid_r, valid_c = {r["theme"] for r in top["rider"]}, {r["theme"] for r in top["captain"]}
    pairs = [p.model_dump() for p in result.pairs if p.rider_theme in valid_r and p.captain_theme in valid_c]

    md = ["# Rider vs captain: top complaint themes (by priority = frequency x severity)", "",
          "| # | Rider theme | % of rider reviews | Captain theme | % of captain reviews |", "|---|---|---|---|---|"]
    for i in range(5):
        r = top["rider"][i] if i < len(top["rider"]) else None
        c = top["captain"][i] if i < len(top["captain"]) else None
        cell = lambda t: (t["theme"], f"{t['frequency']:.1%}") if t else ("", "")  # noqa: E731
        (rt, rf), (ct, cf) = cell(r), cell(c)
        md.append(f"| {i + 1} | {rt} | {rf} | {ct} | {cf} |")
    md += ["", "## Possible links (hypotheses, not findings)", ""]
    md += [f"- **{p['rider_theme']}** (rider) <-> **{p['captain_theme']}** (captain): {p['hypothesis']} "
           f"*Test:* {p['how_to_test']}" for p in pairs] or ["- (none suggested)"]
    md += ["", f"Pairs suggested by {model}; they are co-occurrence hypotheses for a PM to test, not causal findings."]
    (ROOT / "analysis" / "compare_apps.md").write_text("\n".join(md), encoding="utf-8")
    update_metrics(compare_apps={"top5": {a: [r["theme"] for r in rows] for a, rows in top.items()},
                                 "hypotheses": pairs, "hypotheses_model": model})
    print("\n".join(md))


if __name__ == "__main__":
    main()
