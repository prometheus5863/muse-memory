# Decisions

Each entry: the choice, the alternative, and why.

## Embedder: one Protocol, two implementations, query/document `kind`

**Choice.** `Embedder.embed(texts, kind)` returns L2-normalised float32 rows. Ollama
`nomic-embed-text` is primary; `all-MiniLM-L6-v2` on CPU is the fallback. The model's
name is stored with the vectors.

**Alternative.** A bare `embed(texts)` with no `kind` argument.

**Why.** nomic-embed-text is trained with asymmetric prefixes (`search_query:` vs.
`search_document:`), and dropping them measurably hurts retrieval. MiniLM ignores `kind`.
Normalising inside the embedder means every later step can use a dot product as cosine.

**Consequence: the models' similarity scales are not comparable.** On the smoke test,
unrelated sentences score about 0.44 under nomic and about 0.04 under MiniLM. Any similarity
threshold (graph edges!) must therefore be set per model, and vectors from two models can
never share one DB. Switching embedders means re-ingesting.

## Storage: SQLite + NumPy, not a vector database

**Choice.** Vectors are float32 BLOBs in SQLite. At startup they are loaded into one `(n, d)`
NumPy matrix, and search is a single `matrix @ query`, i.e. exact brute-force cosine.

**Alternative.** A vector DB (Chroma, Qdrant, LanceDB) or an ANN index (FAISS, hnswlib).

**Why.** Exact search costs O(n·d). At 10k notes × 768 dims that is about 7.7M multiply-adds:
a few milliseconds and about 30 MB of RAM. ANN indexes trade *recall* for speed, and they only
win once n reaches hundreds of thousands. Below that they add a dependency, a second source of
truth that has to be kept in sync with the graph, and approximate results that make the eval
harder to interpret. SQLite already holds the nodes, edges, clusters and call log, so one file
is the whole state. Revisit this at roughly 200k nodes, or if the brute-force matmul becomes
visible in latency.

## Edges: similarity threshold, not k-NN

**Choice.** Connect i–j whenever cos(i, j) ≥ τ (0.70 for nomic, 0.50 for MiniLM; override
with `MUSE_EDGE_THRESHOLD`, then run `python -m muse.graph rebuild`).

**Alternative.** Connect each node to its k nearest neighbours.

**Why.** k-NN forces every node to have k edges, even an outlier note with no real relatives.
That invents links, and in a memory graph it drags unrelated memories into retrieval as
"neighbours" or "cluster-mates". With a threshold, degree reflects how much related material
actually exists: isolated notes stay isolated (they become singleton clusters), and dense topics
become dense. The threshold also has a meaning that holds as the graph grows, whereas k-NN
edges change for old nodes every time new ones arrive.

**Cost.** τ is model-specific and corpus-sensitive. Too low and one giant component forms;
too high and everything is isolated. `graph stats` prints the p50/p90/p99 of pairwise similarity
for this reason. On the sample notes (nomic) p90 = 0.68, so τ = 0.70 keeps about the top 7% of pairs.
A known weakness: dense topics can produce hub nodes. If your notes do that, a per-node edge cap
is the fix.

## Community detection: Louvain (NetworkX built-in, weighted, seed=42)

**Choice.** `nx.community.louvain_communities(G, weight="weight", seed=42)`.

**Alternatives.** Label propagation (faster, but non-deterministic and unstable on small graphs);
greedy modularity (deterministic, but O(n² log n) and tends to merge small communities);
Leiden (better guarantees against badly connected communities, but needs `leidenalg`/igraph, a
new native dependency).

**Why.** Louvain uses edge weights, which matters because cos 0.95 and 0.71 should not count
equally. It runs in near-linear time, so reclustering after every `/chat` costs milliseconds.
It ships with NetworkX, and a fixed seed makes cluster IDs reproducible, which the eval needs.
Its known flaw (it can produce internally disconnected communities) is what Leiden fixes; that
upgrade becomes worth it if inspection shows clusters that don't make sense.

## Retrieval expansion is capped and ranked by query similarity

**Choice.** `graph(query, k, budget)` returns k direct hits, then fills up to `budget` (default
3k) with 1-hop neighbours first and cluster-mates second, each ranked by cosine to the *query*.

**Alternative.** Return every neighbour and every cluster-mate, or rank them by edge weight to
the seed (spreading activation).

**Why.** Uncapped, one 200-node cluster would flood the prompt. Ranking by query cosine means the
graph decides *which candidates are eligible* and the query decides *which ones win*.
**Consequence for the eval:** because candidates are ranked by query cosine, graph mode at an
equal budget can only beat flat top-B when a graph-connected node outranks an unconnected one
that flat would have included. On a small, cleanly separated corpus that rarely happens (see README).

## Fallback chain: what counts as failure, including a 200 with garbage

| Outcome | Example | Retry same provider? | Then |
|---|---|---|---|
| `timeout`, `connect_error` | network blip, cold start | yes, up to `MUSE_MAX_RETRIES` (2) | next provider |
| `rate_limited` | 429 | yes, waiting `max(backoff, Retry-After)` | next provider |
| `server_error` | 5xx, 529 overloaded | yes | next provider |
| `auth_error`, `client_error` | 401/403, 400/404 | **no**: the same request will fail again | next provider |
| `bad_response` | 200 with HTML, non-JSON, or JSON missing `choices`/`content` | **no** | next provider |
| `empty` | 200 with empty or whitespace text | **no** | next provider |
| `filtered` | `finish_reason=content_filter`, `stop_reason=refusal` | **no** | next provider |

Backoff is `min(cap, base·2^attempt) · U(0.5, 1.5)`. The jitter stops many clients from retrying in sync.

**Why a garbage 200 skips retries.** A malformed or empty body usually means the provider is
misbehaving in a systematic way (a proxy returning an HTML error page, a model that returns empty
output for this prompt, a safety filter). Retrying the same prompt mostly reproduces it and burns
the latency budget, while the next provider fails independently. It is logged under its own
outcome, so a provider that "succeeds" with junk shows up in `calls` instead of being counted as served.

**What it cannot catch.** A well-formed, non-empty, *wrong or off-topic* answer. Detecting that
needs semantic validation (an LLM judge, or schema checks for structured output), which is out of
scope for today. The chain validates shape, not truth.

## Chat write-back stores both sides of the exchange as normal nodes

Each `/chat` adds `User asked: …` and `Assistant answered: …` as two nodes (`source = chat:<request_id>#user|#assistant`),
linked and clustered like any other chunk. **Risk:** model answers become retrievable "memories",
so a hallucination can be recalled later as if it were a note. The `chat:` source prefix makes these
nodes easy to filter, and `remember: false` turns write-back off per request. The eval reports how
many chat nodes are in the DB, because they compete for retrieval budget.
