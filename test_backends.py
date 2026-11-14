"""test_backends.py: every installed backend must agree with exact numpy cosine on a synthetic index.

Run: python test_backends.py            (engines that are not installed are reported as SKIP, never as PASS)
     python test_backends.py --break    (red proof: feeds each engine a wrong vector; the test must FAIL)

For each engine: mirror 600 random unit vectors over 3 `type` folders, then 25 queries:
  * top-1 id equals exact top-1 (ANN engines are run small enough to be exact)
  * recall@10 against exact >= 0.9
  * scope filter returns only the requested type
  * restart: reopen the mirror from disk and get the same top-1
No LLM, no network. Exit 0 = all installed engines PASS; 1 = any FAIL; 2 = nothing but numpy installed.
Machine stamp lives in the repo's git trailers, not here.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vault_recall_backends import BACKENDS, open_backend  # noqa: E402

N, DIM, Q, K = 600, 64, 25, 10
TYPES = ["decisions", "people", "insights"]


def exact(vecs, q, scope, meta):
    sims = vecs @ q
    if scope:
        allowed = np.array([c["type"] in scope for c in meta])
        sims = np.where(allowed, sims, -np.inf)
    order = np.argsort(-sims)
    return [int(i) for i in order if np.isfinite(sims[i])]


def run(name, index_dir, meta, vecs, queries, broken=False):
    be = open_backend(name, index_dir, DIM)
    t0 = time.time()
    be.replace_all(meta, vecs)
    mirror_s = time.time() - t0
    assert be.count() == N, f"{name}: count {be.count()} != {N}"
    hits_at_10, lat = 0, []
    for qi, q in enumerate(queries):
        scope = {TYPES[qi % 3]} if qi % 2 else None
        gold = exact(vecs, q, scope, meta)
        qq = -q if broken else q
        t = time.time()
        got = be.search(qq, K, scope)
        lat.append(time.time() - t)
        ids = [i for i, _ in got]
        assert ids, f"{name}: empty result"
        assert ids[0] == gold[0], f"{name}: top1 {ids[0]} != exact {gold[0]} (scope={scope})"
        if scope:
            bad = [i for i in ids if meta[i]["type"] not in scope]
            assert not bad, f"{name}: scope leak {bad[:3]}"
        hits_at_10 += len(set(ids[:K]) & set(gold[:K])) / K
    be.close()
    r10 = hits_at_10 / len(queries)
    assert r10 >= 0.9, f"{name}: recall@10 {r10:.2f} < 0.9"
    # restart: reopen from disk, same answer
    if name != "numpy":
        be2 = open_backend(name, index_dir, DIM)
        assert be2.count() == N, f"{name}: after reopen count {be2.count()}"
        g = exact(vecs, queries[0], None, meta)[0]
        assert be2.search(queries[0], 1, None)[0][0] == g, f"{name}: restart changed top1"
        be2.close()
    p95 = sorted(lat)[int(0.95 * (len(lat) - 1))] * 1000
    return f"mirror {mirror_s:.2f}s  recall@10 {r10:.2f}  p95 {p95:.1f} ms"


def main():
    broken = "--break" in sys.argv
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((N, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    meta = [{"path": f"n{i}.md", "type": TYPES[i % 3]} for i in range(N)]
    queries = vecs[rng.choice(N, Q, replace=False)] + rng.standard_normal((Q, DIM)).astype("float32") * 0.05
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    tmp = tempfile.mkdtemp(prefix="vr-backends-")
    results, failed, installed = {}, False, 0
    for name in BACKENDS:
        d = os.path.join(tmp, name)
        try:
            results[name] = "PASS  " + run(name, d, meta, vecs, queries, broken)
            installed += 1
        except ImportError as e:
            results[name] = f"SKIP  {e}"
        except Exception as e:  # noqa: BLE001
            results[name] = f"FAIL  {type(e).__name__}: {str(e)[:160]}"
            failed = True
    shutil.rmtree(tmp, ignore_errors=True)
    for k, v in results.items():
        print(f"{k:10s} {v}")
    if broken:
        print("red proof:", "test went RED as it must" if failed else "test stayed GREEN on broken input: TEST IS BLIND")
        sys.exit(0 if failed else 1)
    sys.exit(1 if failed else (2 if installed < 2 else 0))


if __name__ == "__main__":
    main()
