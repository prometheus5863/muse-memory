"""The memory graph: nodes are chunks, edges are cosine >= threshold, clusters are Louvain communities.

    python -m muse.graph stats      counts, largest cluster, density, similarity distribution
    python -m muse.graph rebuild    recompute every edge + cluster (after changing the threshold)
"""
import os
import sys

import networkx as nx
import numpy as np

from muse.embed import Embedder, get_embedder
from muse.store import Store

# Similarity scales differ wildly between models (see DECISIONS.md), so the edge
# threshold is per model. Calibrated on the sample notes; override with MUSE_EDGE_THRESHOLD.
DEFAULT_THRESHOLDS = {"ollama/nomic-embed-text": 0.70, "st/sentence-transformers/all-MiniLM-L6-v2": 0.50}


class Memory:
    def __init__(self, store: Store | None = None, embedder: Embedder | None = None):
        self.store, self.embedder = store or Store(), embedder or get_embedder()
        self.store.check_embedder(self.embedder.name)
        self.threshold = float(os.getenv("MUSE_EDGE_THRESHOLD") or DEFAULT_THRESHOLDS.get(self.embedder.name, 0.6))
        self.reload()

    def reload(self) -> None:
        self.nodes, self.vecs = self.store.load_nodes()
        self.index = {n["id"]: i for i, n in enumerate(self.nodes)}  # node id -> row in self.vecs
        self.G = nx.Graph()
        self.G.add_nodes_from(self.index)
        self.G.add_weighted_edges_from(self.store.load_edges())

    def add(self, texts: list[str], sources: list[str]) -> list[int]:
        """Embed, persist, link to every existing node above threshold, recluster."""
        vecs = self.embedder.embed(texts, kind="document")
        new_ids = self.store.add_nodes(list(zip(texts, sources)), vecs)
        if new_ids:
            self.reload()
            rows = np.array([self.index[i] for i in new_ids])
            self.store.add_edges(self._edges_for(rows))
            self.recluster()
        return new_ids

    def _edges_for(self, rows: np.ndarray) -> list[tuple[int, int, float]]:
        sims = self.vecs[rows] @ self.vecs.T  # (new, all): one matmul, no index needed
        edges = []
        for r, row in zip(rows, sims):
            for j in np.nonzero(row >= self.threshold)[0]:
                if j != r:
                    edges.append((self.nodes[r]["id"], self.nodes[j]["id"], float(row[j])))
        return edges

    def rebuild(self) -> None:
        self.store.replace_edges(self._edges_for(np.arange(len(self.nodes))) if self.nodes else [])
        self.recluster()

    def recluster(self) -> None:
        self.reload()
        # Louvain with a fixed seed: deterministic, weight-aware, ships with NetworkX.
        comms = nx.community.louvain_communities(self.G, weight="weight", seed=42)
        comms = sorted(comms, key=lambda c: (-len(c), min(c)))  # cluster 0 = largest
        self.store.set_clusters({n: cid for cid, c in enumerate(comms) for n in c})
        self.reload()

    def cluster_members(self, node_id: int) -> list[int]:
        c = self.nodes[self.index[node_id]]["cluster"]
        return [n["id"] for n in self.nodes if n["cluster"] == c and n["id"] != node_id]

    def stats(self) -> dict:
        sizes = np.bincount([n["cluster"] for n in self.nodes]) if self.nodes else np.array([0])
        sims = self.vecs @ self.vecs.T if self.nodes else np.zeros((1, 1))
        off = sims[np.triu_indices(len(sims), 1)]
        return {
            "embedder": self.embedder.name, "threshold": self.threshold,
            "nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(),
            "clusters": int((sizes > 0).sum()), "singletons": int((sizes == 1).sum()),
            "largest_cluster": int(sizes.max()), "density": round(nx.density(self.G), 4),
            "isolated_nodes": nx.number_of_isolates(self.G),
            "pairwise_sim_p50_p90_p99": [round(float(np.percentile(off, p)), 3) for p in (50, 90, 99)] if off.size else [],
        }


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    m = Memory()
    if cmd == "rebuild":
        m.rebuild()
    elif cmd != "stats":
        sys.exit(__doc__)
    for k, v in m.stats().items():
        print(f"{k:>26}: {v}")
