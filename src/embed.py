"""Shared EmbeddingGemma helper: embed(texts, task) -> L2-normalized float32 vectors, cached on disk.

Tasks are EmbeddingGemma's built-in prompts. Always compare vectors made with the SAME task:
  "STS"        semantic similarity (safety net, phrase-vs-evidence check, eval matching)
  "Clustering" grouping phrases into themes (Step 5)
"""
import hashlib

import numpy as np

from config import CACHE, CFG, HF_TOKEN

MODEL_NAME = CFG["embeddings"]["model"]
EMB_CACHE = CACHE / "emb"
EMB_CACHE.mkdir(parents=True, exist_ok=True)

_model = None
_stores: dict[str, dict[str, np.ndarray]] = {}  # task -> {text hash: vector}, loaded lazily


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # slow import, so only when needed

        print(f"[embed] loading {MODEL_NAME} on CPU (first run downloads ~1.2 GB)")
        _model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN, device="cpu")  # float32 by default
    return _model


def _store_path(task: str):
    return EMB_CACHE / f"{MODEL_NAME.replace('/', '__')}__{task}.npz"


def _load_store(task: str) -> dict[str, np.ndarray]:
    if task not in _stores:
        path = _store_path(task)
        if path.exists():
            data = np.load(path)
            _stores[task] = dict(zip(data["keys"].tolist(), data["vectors"]))
        else:
            _stores[task] = {}
    return _stores[task]


def _save_store(task: str) -> None:
    store = _stores[task]
    np.savez(_store_path(task), keys=np.array(list(store.keys())),
             vectors=np.stack(list(store.values())).astype(np.float32))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(texts: list[str], task: str = "STS", verbose: bool = True) -> np.ndarray:
    """Return an (n, 768) float32 array of unit-length vectors, one per text, in input order."""
    store = _load_store(task)
    hashes = [_hash(t) for t in texts]
    missing = list(dict.fromkeys(h_t for h_t in zip(hashes, texts) if h_t[0] not in store))  # unique, ordered

    if missing:
        model = _get_model()
        if task not in model.prompts:
            raise ValueError(f"Unknown task '{task}'. Available: {sorted(model.prompts)}")
        vectors = model.encode([t for _, t in missing], prompt_name=task, normalize_embeddings=True,
                               batch_size=32, convert_to_numpy=True, show_progress_bar=len(missing) > 64)
        for (h, _), v in zip(missing, vectors):
            store[h] = v.astype(np.float32)
        _save_store(task)
    if verbose:
        print(f"[embed] {len(texts)} texts, task={task}: {len(texts) - len(missing)} from cache, {len(missing)} computed")
    return np.stack([store[h] for h in hashes])


if __name__ == "__main__":
    import time

    samples = [
        "captain asked for a tip before the ride",
        "captain ne ride se pehle tip maanga",   # Hinglish, same meaning
        "app shows wrong fare at payment",
        "very smooth and fast bike ride",
    ]
    for run in (1, 2):
        start = time.time()
        vecs = embed(samples, task="STS")
        print(f"run {run}: shape {vecs.shape}, dtype {vecs.dtype}, {time.time() - start:.1f}s, "
              f"vector lengths {np.round(np.linalg.norm(vecs, axis=1), 3).tolist()}")

    print("\nCosine similarity (dot product of unit vectors):")
    sims = vecs @ vecs.T
    for i, t in enumerate(samples):
        print(f"  [{i}] " + "  ".join(f"{s:5.2f}" for s in sims[i]) + f"   {t}")
