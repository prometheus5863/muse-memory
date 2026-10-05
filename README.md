# muse

A local-first sandbox for agent memory plus multi-model orchestration.

- **Memory**: local embeddings → SQLite → NetworkX similarity graph → community clusters →
  retrieval as top-k + 1-hop expansion + cluster-mates, with a reason for every node.
- **Orchestration**: an env-configured fallback chain over hosted LLM providers, with retries,
  backoff and a persisted call log.

Status: prototype under construction, built stage by stage. See [DECISIONS.md](DECISIONS.md).

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text                       # optional; falls back to MiniLM on CPU
python -m muse.embed                               # smoke test
```
