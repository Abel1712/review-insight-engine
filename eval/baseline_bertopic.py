"""Step 7 baseline: BERTopic on the same 720 reviews, made fair (not a strawman):
same EmbeddingGemma embeddings, outliers reduced, topics named with the SAME prompt and model as our themes.
Each review gets ONE topic; theme-level recall on gold_test uses the same matching as our method.
Writes metrics.json["evaluation"]["bertopic"] and eval/results.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bakeoff import gold_items  # noqa: E402
from config import CFG, DATA, EVAL, METRICS, SEED, update_metrics  # noqa: E402
from embed import embed  # noqa: E402
from evaluate import recall_by_review, summarize  # noqa: E402
from label_score import MIN_APP_PHRASES, name_cluster  # noqa: E402
from matching import match_many  # noqa: E402


def fit_topics(docs: list[str], emb: np.ndarray):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    model = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=SEED),
                     hdbscan_model=HDBSCAN(min_cluster_size=CFG["evaluate"]["bertopic_min_topic_size"],
                                           metric="euclidean", prediction_data=True))
    topics, _ = model.fit_transform(docs, embeddings=emb)
    n_outliers = int(sum(t == -1 for t in topics))
    topics = model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb)
    return np.array(topics), n_outliers


def main() -> None:
    sample = pd.read_csv(DATA / "sample_final.csv")
    docs = sample["text"].astype(str).tolist()
    emb = embed(docs, task="Clustering")
    topics, n_outliers = fit_topics(docs, emb)
    sample["topic"] = topics

    # Name each topic per app from its most central reviews: same prompt, model and per-app naming as our themes.
    labels = {}
    apps = sample["app"].to_numpy()
    for t in sorted(set(topics)):
        idx_all = np.where(topics == t)[0]
        centre = emb[idx_all].mean(axis=0)
        for app in ("rider", "captain"):
            idx = idx_all[apps[idx_all] == app]
            if len(idx) < MIN_APP_PHRASES:
                idx, app_note = idx_all, None
            else:
                app_note = app
            near = idx[np.argsort(-(emb[idx] @ centre))][:CFG["label"]["phrases_per_cluster"]]
            name, _ = name_cluster([docs[i][:200] for i in near], "complaints or praise", app_note)
            labels[(t, app)] = name.label
    sample["topic_label"] = [labels[(t, a)] for t, a in zip(sample["topic"], sample["app"])]

    test = pd.read_csv(EVAL / "gold_test.csv", encoding="utf-8-sig")
    topic_of = dict(zip(sample["review_id"], sample["topic_label"]))
    cases = [(gold_items(r.gold_complaints), [topic_of[r.review_id]] if r.review_id in topic_of else [])
             for r in test.itertuples()]
    bert = summarize(recall_by_review(test, match_many(cases, "bertopic")))
    bert.update({"n_topics": int(len(set(topics))), "n_outliers_before_reduction": n_outliers})
    update_metrics(evaluation={**json.loads(METRICS.read_text(encoding="utf-8")).get("evaluation", {}),
                               "bertopic": bert})

    ours = json.loads(METRICS.read_text(encoding="utf-8"))["evaluation"]["ours"]
    pct = lambda v: f"{v:.1%}"  # noqa: E731
    md = ["# Evaluation: our method vs BERTopic (70 held-out gold_test reviews)", "",
          "Gold labels are LLM-generated, not hand-written (human spot-check: 20/20); see `gold_labeling_notes.md`.", "",
          "| Method | Recall (all gold items) | Recall on multi-complaint reviews | Precision |",
          "|---|---|---|---|",
          f"| Ours, phrase level | {pct(ours['phrase_level']['recall'])} | "
          f"{pct(ours['phrase_level']['recall_multi_complaint_reviews'])} | "
          f"{pct(ours['precision_strict'])} strict / {pct(ours['precision_with_judge'])} with judge |",
          f"| Ours, theme level | {pct(ours['theme_level']['recall'])} | "
          f"{pct(ours['theme_level']['recall_multi_complaint_reviews'])} | n/a |",
          f"| BERTopic (one topic per review) | {pct(bert['recall'])} | {pct(bert['recall_multi_complaint_reviews'])} | n/a |",
          "", f"Gold items: {bert['n_gold_items']}; multi-complaint reviews: {bert['n_multi_reviews']}. "
          f"BERTopic: {bert['n_topics']} topics, {bert['n_outliers_before_reduction']} outliers reassigned. "
          f"Matching: cosine >= {CFG['evaluate']['match_high']} match, < {CFG['evaluate']['match_low']} no match, "
          "in between judged by gpt-oss-120b (a different model family from the extractor)."]
    (EVAL / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
