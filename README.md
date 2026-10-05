# muse

A local-first sandbox for agent memory and multi-model orchestration.

- **Memory.** A small local embedding model turns markdown chunks into vectors, which are stored in SQLite.
  Edges connect chunks whose cosine similarity is at or above a threshold, Louvain community detection
  groups them into clusters, and retrieval returns top-k hits plus 1-hop neighbours and cluster-mates,
  with the **reason** each node was included.
- **Orchestration.** An env-configured chain of hosted LLM providers, with retries (exponential backoff
  plus jitter) and fall-through on rate limits, timeouts, 5xx errors and garbage 200 responses. Every
  attempt is logged to SQLite.
- **`/chat`** retrieves context, routes the prompt through the chain, and writes the exchange back into memory.

No vector DB, no ORM, no framework beyond FastAPI. About 570 lines of Python, excluding blanks and comments.
Every real design choice is written up in [DECISIONS.md](DECISIONS.md).

```
muse/embed.py      Embedder protocol: Ollama nomic-embed-text, fallback MiniLM on CPU
muse/store.py      SQLite schema: nodes, edges, meta, calls
muse/graph.py      Memory: threshold edges, Louvain clusters, `stats` / `rebuild` CLI
muse/ingest.py     markdown → heading-aware chunks → memory
muse/retrieve.py   flat top-k vs graph-expanded retrieval, with reasons
muse/providers.py  Provider protocol (OpenAI-compatible, Anthropic) + Router fallback chain
muse/api.py        FastAPI: /query /chat /calls /stats
eval/retrieval_eval.py  recall@B, flat vs graph, at equal context budget
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull nomic-embed-text       # optional; without Ollama it falls back to MiniLM on CPU
```

Configuration is environment variables only; see [.env.example](.env.example). Nothing loads `.env`
automatically. Export the variables, or on bash run `set -a; source .env; set +a`.

## Usage (the acceptance path)

```bash
# 1. ingest markdown → nodes + edges in SQLite (idempotent)
python -m muse.ingest examples/notes/*.md

# 2. graph stats: nodes, edges, clusters, largest cluster, density, similarity percentiles
python -m muse.graph stats

# 3. serve, then query
uvicorn muse.api:app
curl -s localhost:8000/query -H 'content-type: application/json' -d '{"query": "why is my bread sour?", "k": 3}'

# 4. chat: the answer, plus the `provider` field and an X-Muse-Provider header
export ANTHROPIC_API_KEY=...  GROQ_API_KEY=...
curl -si localhost:8000/chat -H 'content-type: application/json' -d '{"message": "When do I stop bulk fermentation?"}'

# 5. kill the primary key, restart, repeat: the call log shows the fall-through
python -m muse.providers log
```

Example log after a revoked primary key (local Ollama as the last link):

```
048ec16d8782  anthropic#1 auth_error      483ms served_by=ollama 401 {"type":"error",...
048ec16d8782     ollama#1 ok             6739ms served_by=ollama
```

`/query` returns per node: `score` (cosine to the query), `reason` (`direct_hit` | `neighbour` |
`cluster_mate`), `via` (the seed node that pulled it in), `cluster`, `source` (`file.md#Heading`), and `text`.

Other handy commands: `python -m muse.providers` (show the active chain),
`python -m muse.providers ask "..."`, `python -m muse.retrieve <query>` (flat vs graph side by side),
`python -m muse.graph rebuild` (after changing `MUSE_EDGE_THRESHOLD`).

**Ollama as a provider.** `MUSE_PROVIDERS=...,ollama` adds a local model (default `gemma4:e4b`,
override with `MUSE_OLLAMA_MODEL`) as a keyless last resort. The fallback path above was tested this way.

## Evaluation

```bash
python eval/retrieval_eval.py                    # uses eval/questions.jsonl, else the example set
```

Write `eval/questions.jsonl`, one object per line, with labels in the form `file.md#Heading`
(see [eval/questions.example.jsonl](eval/questions.example.jsonl)):

```json
{"q": "Which of my ferments rely on lactic acid bacteria?", "relevant": ["sourdough.md#Lactic acid bacteria", "ferments.md#Kimchi"]}
```

The comparison is **at equal context budget B**. Graph mode spends ⌈B/2⌉ slots on seeds and the rest on
expansion. Comparing flat top-5 against graph-with-15-nodes would only reward reading more text. The
default `/query` configuration is also reported against flat at the same node count.

### Result on the bundled sample notes (27 chunks, 20 questions, nomic-embed-text, τ=0.70)

| B | flat@B | graph@B | Δ | W/T/L |
|---|---|---|---|---|
| 3 | 0.867 | 0.892 | +0.025 | 1/19/0 |
| 5 | 0.933 | 0.933 | +0.000 | 0/20/0 |
| 10 | 1.000 | 0.975 | −0.025 | 0/19/1 |

**Honest reading: no difference.** One question flips each way, which is noise at n=20. The sample corpus
is too small and too cleanly separated: six topics, and flat top-5 already lands in the right cluster
93% of the time. That leaves no headroom. Graph expansion can only help when relevant material is
*graph-connected to a hit but ranked low by query cosine*. Typical cases are notes that use different
vocabulary for the same thing, and multi-hop questions. Run the eval on your own notes and questions
before quoting a number.
