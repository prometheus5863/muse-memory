"""Flat top-k cosine vs graph-expanded retrieval, recall@B on hand-labelled questions.

    python eval/retrieval_eval.py [questions.jsonl]

One JSON object per line: {"q": "...", "relevant": ["notes/foo.md#Some heading", ...]}.
A retrieved node satisfies a label if its source ends with the label, or (fallback) its
text contains the label. recall@B = labels satisfied within the first B results / labels.

The comparison that matters is at EQUAL budget B: graph mode spends half the budget on
seeds and the rest on expansion. Comparing flat@5 against graph-with-15-nodes would just
reward the method that reads more text, so that row is reported against flat@same-size.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse import retrieve  # noqa: E402
from muse.graph import Memory  # noqa: E402


def recall(results: list[dict], labels: list[str]) -> float:
    hit = lambda lab: any(r["source"].endswith(lab) or lab in r["text"] for r in results)  # noqa: E731
    return sum(map(hit, labels)) / len(labels)


def main() -> None:
    default = Path(__file__).with_name("questions.jsonl")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default if default.exists() else default.with_suffix(".example.jsonl")
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    m = Memory()
    chats = sum(n["source"].startswith("chat:") for n in m.nodes)
    print(f"{len(qs)} questions from {path.name}; {len(m.nodes)} nodes ({chats} from chats), "
          f"{m.G.number_of_edges()} edges, threshold {m.threshold}, {m.embedder.name}\n")

    print(f"{'B':>3} {'flat@B':>8} {'graph@B':>8} {'delta':>7}   wins/ties/losses (graph vs flat)")
    for B in (3, 5, 10):
        f = [recall(retrieve.flat(m, q["q"], B), q["relevant"]) for q in qs]
        g = [recall(retrieve.graph(m, q["q"], k=math.ceil(B / 2), budget=B), q["relevant"]) for q in qs]
        w, lo = sum(a > b for a, b in zip(g, f)), sum(a < b for a, b in zip(g, f))
        fm, gm = sum(f) / len(f), sum(g) / len(g)
        print(f"{B:>3} {fm:>8.3f} {gm:>8.3f} {gm - fm:>+7.3f}   {w}/{len(qs) - w - lo}/{lo}")

    # What /query does by default (k=5 seeds, budget 15), against flat given the same node count.
    g_res = [retrieve.graph(m, q["q"], k=5) for q in qs]
    size = round(sum(map(len, g_res)) / len(qs))
    g = sum(recall(r, q["relevant"]) for r, q in zip(g_res, qs)) / len(qs)
    f = sum(recall(retrieve.flat(m, q["q"], size), q["relevant"]) for q in qs) / len(qs)
    print(f"\ndefault /query (k=5, budget=15): recall {g:.3f} with ~{size} nodes; flat@{size}: {f:.3f}; delta {g - f:+.3f}")
    print("\nper question @5 (flat -> graph):")
    for q in qs:
        a = recall(retrieve.flat(m, q["q"], 5), q["relevant"])
        b = recall(retrieve.graph(m, q["q"], k=3, budget=5), q["relevant"])
        print(f"  {a:.2f} -> {b:.2f} {'+' if b > a else '-' if b < a else ' '} {q['q'][:70]}")


if __name__ == "__main__":
    main()
