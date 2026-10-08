"""Step 5 (QualIT step C): cluster verified phrases into themes with one layer of K-Means.

Negative and positive phrases are clustered separately (embeddings capture topic more than polarity).
Rider and captain phrases are clustered together, so themes are shared across apps. K = best silhouette.
Run:  python src/cluster.py   ->  data/aspects_clustered.csv, analysis/figures/silhouette.png
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import CFG, DATA, FIGURES, SEED, update_metrics
from embed import embed
from plotstyle import INK_2, SERIES, plt

C = CFG["cluster"]


def cluster_polarity(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    X = embed(df["phrase"].tolist(), task="Clustering")  # unit vectors: Euclidean K-Means ~ cosine
    k_cap = min(C["k_max"], len(df) // C["min_avg_cluster_size"])  # themes must average >= N phrases
    ks = range(C["k_min"], max(C["k_min"], k_cap) + 1)
    scores = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        scores[k] = round(float(silhouette_score(X, km.labels_, metric="cosine")), 4)
    top = max(scores.values())
    best_k = max(k for k, s in scores.items() if s >= top - C["tie_tolerance"])  # largest K among near-ties
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    df = df.copy()
    df["cluster_id"] = [f"{label}_{c:02d}" for c in km.labels_]
    df["dist_to_centroid"] = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1).round(4)
    print(f"{label}: {len(df)} phrases, best K = {best_k} (silhouette {scores[best_k]:.3f}); "
          f"sizes {sorted(df['cluster_id'].value_counts().tolist(), reverse=True)}")
    return df, {"n_phrases": len(df), "k": best_k, "silhouette": scores[best_k], "best_silhouette": top,
                "k_rule": f"largest K within {C['tie_tolerance']} of best; K <= n/{C['min_avg_cluster_size']}",
                "silhouette_by_k": scores}


def plot(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for color, (label, r) in zip(SERIES, results.items()):
        ks, ss = list(r["silhouette_by_k"]), list(r["silhouette_by_k"].values())
        ax.plot(ks, ss, color=color, marker="o", markersize=4, label=f"{label} phrases")
        ax.plot([r["k"]], [r["silhouette"]], marker="o", markersize=9, color=color,
                markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.annotate(f"{label}: K = {r['k']}", (r["k"], r["silhouette"]), textcoords="offset points",
                    xytext=(6, 8), color=INK_2, fontsize=9)
    ax.set_title("Silhouette score by number of clusters (higher = cleaner themes)")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("silhouette (cosine)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "silhouette.png")
    plt.close(fig)


def main() -> None:
    aspects = pd.read_csv(DATA / "aspects.csv")
    parts, results = [], {}
    for label, sentiment in (("neg", "neg"), ("pos", "pos")):
        out, info = cluster_polarity(aspects[aspects["sentiment"] == sentiment], label)
        parts.append(out)
        results[label] = info
    pd.concat(parts).to_csv(DATA / "aspects_clustered.csv", index=False, encoding="utf-8")
    plot(results)
    update_metrics(cluster=results)
    print("Saved data/aspects_clustered.csv and analysis/figures/silhouette.png")


if __name__ == "__main__":
    main()
