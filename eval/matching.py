"""Shared matching for Step 7 (identical for our method and BERTopic, so the comparison is fair).

Cosine >= match_high -> match; < match_low -> no match; in between -> LLM judge (secondary model,
a different family from the extractor). Every judge decision is kept for Abel's spot-check.
"""
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CFG  # noqa: E402
from embed import embed  # noqa: E402
from llm import complete_json  # noqa: E402

E = CFG["evaluate"]
JUDGE_LOG: list[dict] = []

SAME_PROMPT = """For each numbered pair, does B describe the same specific issue or praise as A (allowing different
wording)? A is a reference label written by an annotator; B is a system output for the SAME app review.
Return JSON: {{"results": [{{"id": "P1", "same": true}}]}} with every pair id exactly once.
{pairs}"""

REAL_PROMPT = """For each numbered item, is the PHRASE a real issue or praise that the REVIEW actually expresses?
Return JSON: {{"results": [{{"id": "Q1", "real": true}}]}} with every item id exactly once.
{items}"""


class _Same(BaseModel):
    id: str
    same: bool


class _SameBatch(BaseModel):
    results: list[_Same]


class _Real(BaseModel):
    id: str
    real: bool


class _RealBatch(BaseModel):
    results: list[_Real]


def judge_same(pairs: list[tuple[str, str]], kind: str) -> list[bool]:
    out = []
    for s in range(0, len(pairs), E["judge_batch"]):
        chunk = pairs[s:s + E["judge_batch"]]
        text = "\n".join(f"P{i + 1}: A = {a!r} | B = {b!r}" for i, (a, b) in enumerate(chunk))
        res, model = complete_json(SAME_PROMPT.format(pairs=text), _SameBatch, role="secondary", verbose=False)
        got = {r.id: r.same for r in res.results}
        for i, (a, b) in enumerate(chunk):
            verdict = got.get(f"P{i + 1}", False)
            JUDGE_LOG.append({"kind": kind, "a": a, "b": b, "judge_same": verdict, "judge_model": model})
            out.append(verdict)
    return out


def judge_real(items: list[tuple[str, str]]) -> list[bool]:
    out = []
    for s in range(0, len(items), E["judge_batch"]):
        chunk = items[s:s + E["judge_batch"]]
        text = "\n".join(f"Q{i + 1}: REVIEW = {rv[:400]!r} | PHRASE = {ph!r}" for i, (rv, ph) in enumerate(chunk))
        res, _ = complete_json(REAL_PROMPT.format(items=text), _RealBatch, role="secondary", verbose=False)
        got = {r.id: r.real for r in res.results}
        out += [got.get(f"Q{i + 1}", False) for i in range(len(chunk))]
    return out


def match_many(cases: list[tuple[list[str], list[str]]], kind: str) -> list[np.ndarray]:
    """For each (gold items, system outputs) case, a boolean matrix [gold x system] of same-issue matches.
    All ambiguous pairs across all cases go to the judge together, in batches."""
    mats, middle = [], []
    for c, (gold, system) in enumerate(cases):
        if not gold or not system:
            mats.append(np.zeros((len(gold), len(system)), dtype=bool))
            continue
        sims = embed(gold, "STS", verbose=False) @ embed(system, "STS", verbose=False).T
        mats.append(sims >= E["match_high"])
        middle += [(c, i, j) for i, j in zip(*np.where((sims >= E["match_low"]) & (sims < E["match_high"])))]
    verdicts = judge_same([(cases[c][0][i], cases[c][1][j]) for c, i, j in middle], kind) if middle else []
    for (c, i, j), v in zip(middle, verdicts):
        mats[c][i, j] = v
    return mats
