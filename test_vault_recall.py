"""End-to-end tests for vault_recall.py on a tiny throwaway vault. Run: python test_vault_recall.py

Uses the real models (downloaded on first run). Each check prints PASS/FAIL; exit code = number of failures.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vault_recall as vr  # noqa: E402

NOTES = {
    "decisions/pricing.md": "---\ndate: 2026-03-14\n---\n# Pricing\nWe decided to charge 49 dollars per seat per month "
                            "and to drop the free tier after the beta.",
    "decisions/hiring.md": "# Hiring\nWe hire senior Python engineers only, remote from Europe.",
    "people/maria.md": "# Maria Lopez\nMaria runs the Lisbon AI meetup and introduced us to two investors.",
    "notes/garden.md": "# Garden\nПомидоры надо поливать утром, а не вечером, иначе грибок.",
    ".obsidian/workspace.md": "# should be skipped\nsecret config",
}
fails = 0


def check(name, cond, detail=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail}"))
    fails += 0 if cond else 1


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


with tempfile.TemporaryDirectory() as tmp:
    vault = Path(tmp) / "vault"
    for rel, text in NOTES.items():
        p = vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    # 1. index: hidden folders skipped, every visible note embedded
    st = vr.build_index(vault, quiet=True)
    check("index embeds 4 visible notes, skips .obsidian", st["notes_embedded"] == 4, st)
    idx = vr.Index(vault)
    check("stats count notes", idx.stats()["notes"] == 4, idx.stats())

    # 2. semantic recall (no shared keywords with the note)
    hits = idx.search("how much do we charge customers", n=3)
    check("meaning search finds pricing note first", hits and hits[0]["path"] == "decisions/pricing.md",
          [h["path"] for h in hits])
    check("frontmatter date wins over mtime", hits and hits[0]["date"] == "2026-03-14", hits[:1])
    hits_ru = idx.search("когда поливать томаты", n=2)
    check("Russian query finds Russian note", hits_ru and hits_ru[0]["path"] == "notes/garden.md",
          [h["path"] for h in hits_ru])
    check("no note returned twice", len({h["path"] for h in idx.search("we", n=10)}) == len(idx.search("we", n=10)))

    # 3. scope filter
    sc = idx.search("who introduced us to investors", n=5, scope={"decisions"})
    check("scope keeps only that folder", sc and all(h["type"] == "decisions" for h in sc), sc)

    # 4. incremental re-index
    st2 = vr.build_index(vault, quiet=True)
    check("unchanged vault re-embeds nothing", st2["notes_embedded"] == 0 and st2["notes_reused"] == 4, st2)
    (vault / "decisions/hiring.md").write_text("# Hiring\nWe now also hire designers in Brazil.", encoding="utf-8")
    st3 = vr.build_index(vault, quiet=True)
    check("one edited note re-embeds exactly one", st3["notes_embedded"] == 1, st3)
    check("running Index picks up new index", any("designers" in h["snippet"]
                                                   for h in idx.search("designers Brazil", n=2)))

    # 5. HTTP server
    port = free_port()
    handler = vr.make_handler(vr.Index(vault), None)
    srv = vr.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    code, h = get(base + "/health")
    check("/health 200 with service name", code == 200 and h.get("service") == "vault-recall", (code, h))
    code, r = get(base + "/api?q=Lisbon%20meetup%20organiser&n=2")
    check("/api returns hits", code == 200 and r["hits"] and r["hits"][0]["path"] == "people/maria.md", r)
    code, r = get(base + "/api?q=")
    check("/api empty query -> 400", code == 400, code)
    srv.shutdown()

    port2 = free_port()
    srv2 = vr.ThreadingHTTPServer(("127.0.0.1", port2), vr.make_handler(vr.Index(vault), "s3cret"))
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    code, _ = get(f"http://127.0.0.1:{port2}/health")
    check("token set: no header -> 401", code == 401, code)
    code, _ = get(f"http://127.0.0.1:{port2}/health", token="s3cret")
    check("token set: right header -> 200", code == 200, code)
    srv2.shutdown()

    # 6. missing index -> 503, not a crash
    empty = Path(tmp) / "empty"
    empty.mkdir()
    port3 = free_port()
    srv3 = vr.ThreadingHTTPServer(("127.0.0.1", port3), vr.make_handler(vr.Index(empty), None))
    threading.Thread(target=srv3.serve_forever, daemon=True).start()
    code, r = get(f"http://127.0.0.1:{port3}/api?q=x")
    check("no index -> 503 index missing", code == 503 and r.get("error") == "index missing", (code, r))
    srv3.shutdown()

    # 7. refuses public bind without token
    env = {k: v for k, v in os.environ.items() if k != "VAULT_RECALL_TOKEN"}
    p = subprocess.run([sys.executable, str(HERE / "vault_recall.py"), "serve", str(vault), "--host", "0.0.0.0"],
                       capture_output=True, text=True, env=env, timeout=60)
    check("non-loopback bind without token refused", p.returncode != 0 and "refusing" in p.stderr, p.stderr[-200:])

    # 8. MCP stdio
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "recall", "arguments": {"query": "price per seat", "n": 1}}},
        {"jsonrpc": "2.0", "id": 4, "method": "nope"},
    ]
    p = subprocess.run([sys.executable, str(HERE / "vault_recall.py"), "mcp", str(vault)],
                       input="\n".join(json.dumps(r) for r in reqs) + "\n",
                       capture_output=True, text=True, encoding="utf-8", timeout=300)
    out = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    by_id = {m["id"]: m for m in out}
    check("MCP: notification gets no reply, 4 replies total", len(out) == 4, p.stdout[-300:] + p.stderr[-300:])
    check("MCP initialize echoes protocol", by_id.get(1, {}).get("result", {}).get("protocolVersion") == "2025-06-18")
    check("MCP tools/list has recall", [t["name"] for t in by_id.get(2, {}).get("result", {}).get("tools", [])]
          == ["recall"])
    text = by_id.get(3, {}).get("result", {}).get("content", [{}])[0].get("text", "")
    check("MCP recall finds pricing", "pricing" in text, text[:200])
    check("MCP unknown method -> -32601", by_id.get(4, {}).get("error", {}).get("code") == -32601)

print(f"\n{fails} failure(s)")
sys.exit(fails)
