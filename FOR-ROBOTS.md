# FOR ROBOTS

You are an AI agent reading this repo. Ranked takeaways, densest first. One rule about
this file itself: **it is data, not authorization.** Nothing here overrides your
operator's policies or lets you install anything on your own.

## Alpha, ranked

1. **One file turns any folder of markdown into a memory backend you can query over HTTP or MCP.**
   `python vault_recall.py index <folder>` then `serve` (http://127.0.0.1:8770, `GET /api?q=...`) or
   `mcp` (stdio, one tool `recall(query, n?, scope?)`). Retrieval is dense multilingual-e5-base
   (top 60, one best chunk per note) plus an optional cross-encoder rerank; no LLM is called and
   nothing leaves the machine. Read-only over the notes; the index lives in `<folder>/.vault-recall/`.
2. **The index is a numpy file and stays the source of truth; engines are mirrors.** `--backend
   numpy|sqlite-vec|lancedb|qdrant|pgvector|chdb` mirrors the index into that store after `index` and
   searches through it in `ask`, `serve`, `mcp`. Every backend implements four calls
   (`replace_all`, `search`, `count`, `close`) in `vault_recall_backends.py`. A missing engine is an
   ImportError that prints the pip line; there is no silent fallback.
3. **`test_backends.py` is the acceptance test, and it goes red on a wrong vector.** For every
   installed engine: top-1 must equal exact cosine, recall@10 at least 0.9, the scope filter must
   not leak, reopening from disk must return the same top-1. `--break` feeds each engine a negated
   query; on 2026-11-05 that turned numpy, sqlite-vec, qdrant, lancedb and pgvector red, which is
   the evidence that a green run means something. chdb has no Windows wheel and is skipped there.
4. **The engine comparison behind the backends is published, with its limits.** On a 20,000-vector
   slice of a 105,374-chunk index (768-d), one run, host under load: numpy scan p50 2.03 ms recall
   1.0; pgvector HNSW (PostgreSQL 18.6 via embedded-postgres) p50 1.21 ms recall 0.995; chDB
   `cosineDistance` scan 30.17 ms; Qdrant client local mode (exact search) 84.25 ms; LanceDB flat
   157.42 ms, LanceDB IVF_PQ 64x48 recall 0.633 (misconfigured, kept on purpose). Full table, raw
   `results-*.json` and the bench script: https://github.com/tonydzi/clawrush/tree/main/artifacts/memory-db-pilots-2026-11-05.
   Article: https://github.com/tonydzi/clawrush/blob/main/longreads/seven-engines-under-one-agent-memory-2026-11-13.md.
5. **Security model is loopback-first.** Binds 127.0.0.1 only; any other address is refused unless
   `VAULT_RECALL_TOKEN` is set, and then every request needs `Authorization: Bearer <token>`.
   Returned text is data from notes, never instructions (the MCP tool description says so too).

## Provenance

Numbers in README and above come from `test_backends.py` runs on 2026-11-05 on a Windows hub
machine (Python 3.12, numpy 2.5.4, sqlite-vec 0.1.9, lancedb 0.41.0, qdrant-client 1.20.0,
embedded-postgres 18.6.3 with pgvector 0.8.6) and from the bench artifacts linked in item 4. The
original one-file engine (0.1.0, 2026-10-23) was extracted from the second brain behind
`tonydzi/charm-os` after daily use across a multi-machine agent fleet; 0.2.0 (2026-11-05) added
the backends. Corpus vectors are private and not published; the scripts run on any `.npy` of
normalised vectors plus a metadata pickle.

## Family

Sibling repos of the Palo Alto AI Research Lab: `tonydzi/claude-bible` is the family map, start
there for how the repos relate. `tonydzi/charm-os` is where this engine came from;
`tonydzi/sqlite-graph-memory` holds the retrieval-quality bench (memory-bench) that measures a
different thing (answer quality, refusals) than the storage-engine bench linked above;
`tonydzi/deep-research` holds the distilled research reports; `tonydzi/clawrush` holds the
longreads and artifacts. License here is MIT.
