"""Retrieval: flat top-k cosine, and graph-expanded (top-k + 1-hop neighbours + cluster-mates).

Every result carries `reason` (direct_hit | neighbour | cluster_mate) and `via` (the seed
node that pulled it in), so the assembled context is explainable.
"""
import numpy as np

from muse.graph import Memory


def _hit(m: Memory, i: int, score: float, reason: str, via: int | None = None) -> dict:
    n = m.nodes[i]
    return {"id": n["id"], "source": n["source"], "cluster": n["cluster"], "score": round(score, 4),
            "reason": reason, "via": via, "text": n["text"]}


def flat(m: Memory, query: str, k: int = 5) -> list[dict]:
    if not m.nodes:
        return []
    sims = m.vecs @ m.embedder.embed([query], kind="query")[0]
    return [_hit(m, i, float(sims[i]), "direct_hit") for i in np.argsort(-sims)[:k]]


def graph(m: Memory, query: str, k: int = 5, budget: int | None = None) -> list[dict]:
    """Seed with the top-k cosine hits, then add one-hop neighbours, then cluster-mates.

    `budget` caps the total returned (default 3k). Expansion candidates are ranked by
    neighbour-before-cluster-mate, then by their own cosine to the query, so a huge
    cluster cannot flood the context with off-topic members.
    """
    if not m.nodes:
        return []
    budget = budget or 3 * k
    sims = m.vecs @ m.embedder.embed([query], kind="query")[0]
    seeds = [int(i) for i in np.argsort(-sims)[:min(k, budget)]]
    out = [_hit(m, i, float(sims[i]), "direct_hit") for i in seeds]
    taken = {m.nodes[i]["id"] for i in seeds}

    candidates: dict[int, tuple[int, int]] = {}  # node id -> (tier, via); tier 0 beats tier 1
    for i in seeds:
        sid = m.nodes[i]["id"]
        for tier, members in ((0, m.G.neighbors(sid)), (1, m.cluster_members(sid))):
            for nid in members:
                if nid not in taken and (nid not in candidates or tier < candidates[nid][0]):
                    candidates[nid] = (tier, sid)

    ranked = sorted(candidates, key=lambda nid: (candidates[nid][0], -sims[m.index[nid]]))
    for nid in ranked[:budget - len(out)]:
        tier, via = candidates[nid]
        out.append(_hit(m, m.index[nid], float(sims[m.index[nid]]), ("neighbour", "cluster_mate")[tier], via))
    return out


if __name__ == "__main__":
    import sys
    mem = Memory()
    q = " ".join(sys.argv[1:]) or "how do I keep a ferment safe?"
    for name, res in (("flat", flat(mem, q)), ("graph", graph(mem, q))):
        print(f"\n== {name}: {q}")
        for r in res:
            print(f"  {r['score']:.3f}  {r['reason']:<12} via={str(r['via']):<4} c{r['cluster']:<3} {r['source']}")
