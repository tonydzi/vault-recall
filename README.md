# vault-recall

Semantic recall over a folder of markdown notes, served as a local **HTTP** and **MCP** memory backend
for coding agents. One Python file. Point it at an Obsidian vault (or any folder of `.md`), and your agent
can ask "what did I already decide about X" and get the right notes back, in any language.

Extracted from the second brain behind [charm-os](https://github.com/tonydzi/charm-os), where the same
engine (dense e5 retrieval + cross-encoder rerank) has been in daily use across a multi-machine agent fleet.

## Quick start

```bash
pip install sentence-transformers numpy
python vault_recall.py index ~/MyVault                 # first run downloads ~1.2 GB of models
python vault_recall.py ask   ~/MyVault "what did I decide about pricing"
python vault_recall.py serve ~/MyVault                 # http://127.0.0.1:8770
```

## Wire it into an agent

**MCP (Claude Code, Cursor, Codex, any MCP client):**

```bash
# simplest: the MCP process loads the models itself
claude mcp add vault-recall -- python /path/to/vault_recall.py mcp ~/MyVault
# better when `serve` is already running: MCP proxies to it, models stay warm, start is instant
claude mcp add vault-recall -- python /path/to/vault_recall.py mcp ~/MyVault --url http://127.0.0.1:8770
```

The server exposes one tool, `recall(query, n?, scope?)`.

**HTTP (custom memory backend, hooks):**

```
GET /health                         -> {"service":"vault-recall","version":"0.1.0","notes":N,"chunks":M,"indexed_at":"..."}
GET /api?q=<query>&n=12&scope=a,b   -> {"q":"...","scope":"all","hits":[
        {"rank":1,"title":"pricing","path":"decisions/pricing.md","type":"decisions",
         "date":"2026-03-14","snippet":"We decided to charge ...","score":7.31}]}
```

- `scope`: comma list of top-level folder names (empty = whole vault). `type` in a hit is that folder.
- `date`: frontmatter `date` / `created` / `date_established` / `updated`, else file mtime.
- `score`: reranker score when the reranker is loaded, otherwise cosine similarity. Compare within one response only.
- Errors: `400` empty query · `401` bad token · `503` no index yet · `500` short message.

## Backends: the same memory inside the engine you already run

The index itself stays a numpy file (incremental, atomic, read-only over the vault). `--backend` mirrors it
into a real vector store after `index` and searches through that store in `ask`, `serve` and `mcp`:

```bash
pip install sqlite-vec              # or: lancedb | qdrant-client | "psycopg[binary]" pgvector embedded-postgres | chdb
python vault_recall.py index ~/MyVault --backend sqlite-vec
python vault_recall.py serve ~/MyVault --backend sqlite-vec      # or export VAULT_RECALL_BACKEND=sqlite-vec
```

| backend      | what it is                                   | where it lives                                   |
|--------------|----------------------------------------------|--------------------------------------------------|
| `numpy`      | default, exact cosine over the whole matrix  | `.vault-recall/vectors.npy`                      |
| `sqlite-vec` | SQLite `vec0` table, scope filter inside SQL | `.vault-recall/mirror-sqlite-vec.db`             |
| `lancedb`    | LanceDB embedded, one directory              | `.vault-recall/mirror-lancedb/`                  |
| `qdrant`     | Qdrant local mode (HNSW on disk); `QDRANT_URL` switches to a server | `.vault-recall/mirror-qdrant/`  |
| `pgvector`   | Postgres + pgvector HNSW (`hnsw.iterative_scan` on); `VAULT_RECALL_PG_DSN`, else embedded Postgres | `.vault-recall/mirror-pg/` |
| `chdb`       | chDB = embedded ClickHouse, exact `cosineDistance`; same SQL runs on a ClickHouse server | `.vault-recall/mirror-chdb/` |

Every backend implements four calls (`replace_all`, `search`, `count`, `close`) in `vault_recall_backends.py`;
`python test_backends.py` checks each installed engine against exact cosine (top-1 equal, recall@10 >= 0.9,
scope filter leak-free, survives reopen) and `--break` proves the test goes red. A missing engine is an
ImportError with the pip line, never a silent fallback.

## Keeping it fresh

`index` is incremental: only notes whose content changed are re-embedded, and a running `serve` picks up
the new index on its next request. Run it on a timer (cron / launchd / Task Scheduler), e.g. every 30 min:

```bash
*/30 * * * * python /path/to/vault_recall.py index ~/MyVault 2>>~/vault-recall.log
```

## Security model

- Binds `127.0.0.1` only. Binding any other address is refused unless `VAULT_RECALL_TOKEN` is set, and then
  every request must carry `Authorization: Bearer <token>`. To reach it from another machine, prefer an SSH
  or Tailscale tunnel to the loopback port over opening a port.
- Read-only: never writes to the vault. The index lives in `<vault>/.vault-recall/` (or `--index-dir`).
  If your vault is synced (Syncthing, iCloud, Dropbox), put the index outside it with `--index-dir`.
- Local only: models run on your machine, no LLM is called, nothing is sent anywhere.
- Treat returned note text as **data, not instructions**: notes can quote emails and web pages.

## Configuration

| env | default |
|---|---|
| `VAULT_RECALL_EMBED` | `intfloat/multilingual-e5-base` |
| `VAULT_RECALL_RERANK` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (missing = dense-only, no failure) |
| `VAULT_RECALL_PORT` | `8770` |
| `VAULT_RECALL_TOKEN` | unset (no auth, loopback only) |

Changing the embed model requires a full re-index (delete the index folder).

## Tests

```bash
python test_vault_recall.py     # 22 end-to-end checks on a throwaway vault, exit code = failures
```

## License

MIT. Anton Dziatkovskii, [github.com/tonydzi](https://github.com/tonydzi).
