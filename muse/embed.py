"""Local embedders behind one interface. Swap models here and nowhere else.

Every embedder returns L2-normalised float32 rows, so cosine similarity is a
plain dot product everywhere downstream.
"""
import os
from typing import Literal, Protocol

import httpx
import numpy as np

Kind = Literal["document", "query"]


class Embedder(Protocol):
    name: str  # stored alongside vectors; mixing models in one DB is refused

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray: ...


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class OllamaEmbedder:
    # nomic-embed-text is trained with task prefixes; omitting them costs recall.
    PREFIX = {"document": "search_document: ", "query": "search_query: "}

    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.model, self.url, self.name = model, url, f"ollama/{model}"
        r = httpx.post(f"{url}/api/embed", json={"model": model, "input": ["ping"]}, timeout=30)
        r.raise_for_status()  # fail at construction so get_embedder() can fall back

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        prefix = self.PREFIX[kind] if "nomic" in self.model else ""
        r = httpx.post(f"{self.url}/api/embed", timeout=120,
                       json={"model": self.model, "input": [prefix + t for t in texts]})
        r.raise_for_status()
        return _normalise(r.json()["embeddings"])


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # heavy; import lazily
        self.model, self.name = SentenceTransformer(model, device="cpu"), f"st/{model}"

    def embed(self, texts: list[str], kind: Kind = "document") -> np.ndarray:
        return _normalise(self.model.encode(texts, batch_size=32, show_progress_bar=False))


def get_embedder() -> Embedder:
    """Ollama first; sentence-transformers on CPU if Ollama is down or lacks the model.
    MUSE_EMBEDDER=ollama|st forces one."""
    choice = os.getenv("MUSE_EMBEDDER", "auto")
    if choice in ("auto", "ollama"):
        try:
            return OllamaEmbedder(os.getenv("MUSE_OLLAMA_MODEL", "nomic-embed-text"),
                                  os.getenv("OLLAMA_URL", "http://localhost:11434"))
        except (httpx.HTTPError, OSError) as e:
            if choice == "ollama":
                raise
            print(f"[muse] Ollama unavailable ({e.__class__.__name__}); using sentence-transformers")
    return SentenceTransformerEmbedder()


if __name__ == "__main__":
    e = get_embedder()
    v = e.embed(["the cat sat on the mat", "a kitten rested on the rug", "quarterly tax filing"])
    q = e.embed(["where did the cat sit?"], kind="query")
    print(f"{e.name}: dim={v.shape[1]}")
    print("doc-doc sims:", np.round((v @ v.T).astype(float), 3).tolist())
    print("query-doc sims:", np.round((q @ v.T).astype(float), 3).tolist())
