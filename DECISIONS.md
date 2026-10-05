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
