# SQLite notes

## WAL mode
PRAGMA journal_mode=WAL lets readers proceed while one writer appends to the write-ahead log. Checkpoints copy WAL pages back into the main database file. It is the right default for any app with concurrent reads.

## Single writer
SQLite allows only one writer at a time; a second writer gets SQLITE_BUSY. Set a busy_timeout so it waits instead of failing immediately. For a single-process app, serialising writes behind a lock in the application is simpler.

## Indexes and query plans
EXPLAIN QUERY PLAN shows whether a query uses an index or does a full table scan. A composite index on (a, b) helps queries filtering on a, or a and b, but not on b alone. Covering indexes avoid the lookup back into the table.

## Storing vectors
Embeddings can be stored as BLOBs of float32. At a few thousand rows, loading them all into a NumPy array and doing a matrix multiply is faster than any index and has zero extra infrastructure. Approximate nearest neighbour indexes only pay off at hundreds of thousands of vectors.
