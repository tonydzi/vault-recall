"""vault_recall_backends.py: pluggable vector stores for vault-recall.

vault_recall.py keeps its own source of truth (`meta.json` + `vectors.npy`, incremental, atomic).
A backend is a *mirror* of that index inside a real engine, so the same notes can be recalled by
the engine your agents already run on. Select with `--backend <name>` or `VAULT_RECALL_BACKEND`.

    numpy      default: brute-force cosine over vectors.npy (no extra install)
    sqlite-vec SQLite + sqlite-vec vec0 table            pip install sqlite-vec
    lancedb    LanceDB embedded (one directory)          pip install lancedb
    qdrant     Qdrant local mode (one directory)         pip install qdrant-client
    pgvector   Postgres + pgvector                       pip install "psycopg[binary]" pgvector
               DSN in VAULT_RECALL_PG_DSN, or `pip install embedded-postgres` for a zero-config local server
    chdb       chDB = embedded ClickHouse               pip install chdb

Contract (same for every backend):
    replace_all(meta, vecs)      full mirror of the index (idempotent; called after each `index` run)
    search(qvec, top, scope)     -> [(chunk_index, cosine_score)] best first; scope = set of `type`
    count()                      -> number of chunks mirrored
    close()

Every backend stores the chunk index `i` as its primary key, so results map back to `meta[i]`.
Missing engine = ImportError with the pip line, never a silent fallback. Nothing here calls an LLM.

Author: Anton Dziatkovskii (github.com/tonydzi). License: MIT.
"""
import json
import os
import shutil
from pathlib import Path

BACKENDS = ("numpy", "sqlite-vec", "lancedb", "qdrant", "pgvector", "chdb")


def _need(module, pip_line):
    try:
        return __import__(module)
    except ImportError as e:  # noqa: BLE001
        raise ImportError(f"backend needs `{pip_line}` ({e})") from None


class NumpyBackend:
    """The built-in store: exact cosine over the whole matrix. Baseline for every other backend."""
    name = "numpy"

    def __init__(self, index_dir, dim):
        self.meta, self.vecs = [], None

    def replace_all(self, meta, vecs):
        self.meta, self.vecs = meta, vecs

    def search(self, qvec, top, scope=None):
        import numpy as np
        if self.vecs is None or not len(self.meta):
            return []
        sims = self.vecs @ qvec
        if scope:
            allowed = np.array([c["type"] in scope for c in self.meta])
            sims = np.where(allowed, sims, -np.inf)
        order = np.argsort(-sims)[:top]
        return [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]

    def count(self):
        return len(self.meta)

    def close(self):
        pass


class SqliteVecBackend:
    """SQLite + sqlite-vec: `vec0` virtual table, `type` as a metadata column so scope filters run inside SQL."""
    name = "sqlite-vec"

    def __init__(self, index_dir, dim):
        sqlite3 = _need("sqlite3", "python stdlib sqlite3")
        sqlite_vec = _need("sqlite_vec", "pip install sqlite-vec")
        self.path = Path(index_dir) / "mirror-sqlite-vec.db"
        self.db = sqlite3.connect(str(self.path))
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.dim = dim
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(id INTEGER PRIMARY KEY, "
            f"embedding float[{dim}] distance_metric=cosine, type TEXT)")

    def replace_all(self, meta, vecs):
        self.db.execute("DELETE FROM chunks")
        rows = [(i, vecs[i].astype("float32").tobytes(), c["type"]) for i, c in enumerate(meta)]
        self.db.executemany("INSERT INTO chunks(id, embedding, type) VALUES (?, ?, ?)", rows)
        self.db.commit()

    def search(self, qvec, top, scope=None):
        q = qvec.astype("float32").tobytes()
        if scope:
            marks = ",".join("?" * len(scope))
            sql = (f"SELECT id, distance FROM chunks WHERE embedding MATCH ? AND k = ? "
                   f"AND type IN ({marks}) ORDER BY distance")
            rows = self.db.execute(sql, (q, top, *sorted(scope))).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id, distance FROM chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (q, top)).fetchall()
        return [(int(i), 1.0 - float(d)) for i, d in rows]

    def count(self):
        return self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def close(self):
        self.db.close()


class LanceDBBackend:
    """LanceDB embedded: one directory, Arrow/Lance files, cosine metric. Sync-friendly like the numpy index."""
    name = "lancedb"

    def __init__(self, index_dir, dim):
        self.lancedb = _need("lancedb", "pip install lancedb")
        self.dir = Path(index_dir) / "mirror-lancedb"
        self.dim = dim
        self.db = self.lancedb.connect(str(self.dir))
        self.tbl = None
        if "chunks" in self.db.table_names():
            self.tbl = self.db.open_table("chunks")

    def replace_all(self, meta, vecs):
        rows = [{"id": i, "type": c["type"], "vector": vecs[i].astype("float32").tolist()}
                for i, c in enumerate(meta)]
        if "chunks" in self.db.table_names():
            self.db.drop_table("chunks")
        self.tbl = self.db.create_table("chunks", data=rows) if rows else None

    def search(self, qvec, top, scope=None):
        if self.tbl is None:
            return []
        q = self.tbl.search(qvec.astype("float32").tolist(), vector_column_name="vector").metric("cosine")
        if scope:
            vals = ",".join("'" + s.replace("'", "''") + "'" for s in sorted(scope))
            q = q.where(f"type IN ({vals})", prefilter=True)
        rows = q.limit(top).to_list()
        return [(int(r["id"]), 1.0 - float(r["_distance"])) for r in rows]

    def count(self):
        return self.tbl.count_rows() if self.tbl is not None else 0

    def close(self):
        pass


class QdrantBackend:
    """Qdrant local mode: exact search in one directory, payload filter on `type`. Point QDRANT_URL at a server to leave local mode."""
    name = "qdrant"
    COLL = "vault_recall"

    def __init__(self, index_dir, dim):
        qc = _need("qdrant_client", "pip install qdrant-client")
        from qdrant_client import QdrantClient
        from qdrant_client import models as m
        self.m = m
        url = os.environ.get("QDRANT_URL")
        self.client = QdrantClient(url=url) if url else QdrantClient(path=str(Path(index_dir) / "mirror-qdrant"))
        self.dim = dim
        if not self.client.collection_exists(self.COLL):
            self.client.create_collection(self.COLL, vectors_config=m.VectorParams(size=dim, distance=m.Distance.COSINE))
        else:
            current = self.client.get_collection(self.COLL).config.params.vectors
            if getattr(current, "size", None) != dim:
                self.client.close()
                raise ValueError("Qdrant mirror vector dimensions differ; use a fresh index directory")

    def replace_all(self, meta, vecs):
        m = self.m
        # Keep the collection: deleting/recreating it can strand local SQLite
        # handles on Windows. Clear points through the public delete API instead.
        self.client.delete(self.COLL, points_selector=m.FilterSelector(filter=m.Filter()), wait=True)
        batch = 512
        for s in range(0, len(meta), batch):
            pts = [m.PointStruct(id=i, vector=vecs[i].astype("float32").tolist(), payload={"type": meta[i]["type"]})
                   for i in range(s, min(s + batch, len(meta)))]
            self.client.upsert(self.COLL, pts)

    def search(self, qvec, top, scope=None):
        m = self.m
        flt = m.Filter(must=[m.FieldCondition(key="type", match=m.MatchAny(any=sorted(scope)))]) if scope else None
        res = self.client.query_points(self.COLL, query=qvec.astype("float32").tolist(), limit=top, query_filter=flt)
        return [(int(p.id), float(p.score)) for p in res.points]

    def count(self):
        return self.client.count(self.COLL).count

    def close(self):
        self.client.close()


class PgvectorBackend:
    """Postgres + pgvector: HNSW cosine index, scope filter in WHERE (pgvector >= 0.8 iterative scans keep k full).
    DSN from VAULT_RECALL_PG_DSN; without it, starts an embedded Postgres (pip install embedded-postgres) under the index dir."""
    name = "pgvector"

    def __init__(self, index_dir, dim):
        psycopg = _need("psycopg", 'pip install "psycopg[binary]" pgvector')
        from pgvector.psycopg import register_vector
        dsn = os.environ.get("VAULT_RECALL_PG_DSN")
        self._embedded = None
        if not dsn:
            ep = _need("embedded_postgres", "pip install embedded-postgres (or set VAULT_RECALL_PG_DSN)")
            pgdata = Path(index_dir) / "mirror-pg"
            pgdata.mkdir(parents=True, exist_ok=True)
            self._embedded = ep.get_server(pgdata)
            dsn = self._embedded.get_uri()
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.dim = dim
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS vr_chunks (id int PRIMARY KEY, type text, embedding vector({dim}))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS vr_chunks_hnsw ON vr_chunks USING hnsw (embedding vector_cosine_ops)")

    def replace_all(self, meta, vecs):
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE vr_chunks")
            with cur.copy("COPY vr_chunks (id, type, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int4", "text", "vector"])
                for i, c in enumerate(meta):
                    cp.write_row((i, c["type"], vecs[i].astype("float32")))

    def search(self, qvec, top, scope=None):
        with self.conn.cursor() as cur:
            cur.execute("SET hnsw.iterative_scan = relaxed_order")
            if scope:
                cur.execute("SELECT id, 1 - (embedding <=> %s) FROM vr_chunks WHERE type = ANY(%s) "
                            "ORDER BY embedding <=> %s LIMIT %s", (qvec, sorted(scope), qvec, top))
            else:
                cur.execute("SELECT id, 1 - (embedding <=> %s) FROM vr_chunks ORDER BY embedding <=> %s LIMIT %s",
                            (qvec, qvec, top))
            return [(int(i), float(s)) for i, s in cur.fetchall()]

    def count(self):
        return self.conn.execute("SELECT count(*) FROM vr_chunks").fetchone()[0]

    def close(self):
        self.conn.close()
        if self._embedded is not None:
            self._embedded.cleanup()


class ChdbBackend:
    """chDB = embedded ClickHouse: MergeTree table, exact cosineDistance (the vector index needs a server build).
    The same SQL runs unchanged against a ClickHouse server, which is the point of the mirror."""
    name = "chdb"

    def __init__(self, index_dir, dim):
        chdb = _need("chdb", "pip install chdb")
        from chdb import session
        self.sess = session.Session(str(Path(index_dir) / "mirror-chdb"))
        self.dim = dim
        self.sess.query("CREATE DATABASE IF NOT EXISTS vr")
        self.sess.query("CREATE TABLE IF NOT EXISTS vr.chunks (id UInt32, type String, vec Array(Float32)) "
                        "ENGINE = MergeTree ORDER BY id")

    def replace_all(self, meta, vecs):
        self.sess.query("TRUNCATE TABLE vr.chunks")
        batch = 2000
        for s in range(0, len(meta), batch):
            rows = ",".join(
                "(%d,'%s',[%s])" % (i, meta[i]["type"].replace("'", "\\'"), ",".join("%.6f" % x for x in vecs[i]))
                for i in range(s, min(s + batch, len(meta))))
            if rows:
                self.sess.query(f"INSERT INTO vr.chunks (id, type, vec) VALUES {rows}")

    def search(self, qvec, top, scope=None):
        q = "[" + ",".join("%.6f" % x for x in qvec) + "]"
        where = ""
        if scope:
            vals = ",".join("'" + s.replace("'", "\\'") + "'" for s in sorted(scope))
            where = f"WHERE type IN ({vals})"
        out = self.sess.query(
            f"SELECT id, 1 - cosineDistance(vec, {q}) AS s FROM vr.chunks {where} ORDER BY s DESC LIMIT {int(top)}",
            "JSONEachRow")
        rows = [json.loads(line) for line in str(out).splitlines() if line.strip()]
        return [(int(r["id"]), float(r["s"])) for r in rows]

    def count(self):
        return int(str(self.sess.query("SELECT count() FROM vr.chunks", "TabSeparated")).strip() or 0)

    def close(self):
        self.sess.close()


_CLASSES = {
    "numpy": NumpyBackend, "sqlite-vec": SqliteVecBackend, "lancedb": LanceDBBackend,
    "qdrant": QdrantBackend, "pgvector": PgvectorBackend, "chdb": ChdbBackend,
}


def open_backend(name, index_dir, dim):
    name = (name or os.environ.get("VAULT_RECALL_BACKEND") or "numpy").lower()
    if name not in _CLASSES:
        raise ValueError(f"unknown backend {name!r}; choose one of {', '.join(BACKENDS)}")
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    return _CLASSES[name](index_dir, dim)


def wipe_mirrors(index_dir):
    """Remove every engine mirror under the index dir (the numpy index itself is untouched)."""
    for p in Path(index_dir).glob("mirror-*"):
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
