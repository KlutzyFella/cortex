"""Idempotent corpus seeding — the harness owns its fixture data.

--via db (default, the metric path): chunks + embeds locally and upserts with
the same SQL semantics as the ingestion worker (upsert document, delete stale
high-index chunks, upsert batch). Deterministic, no Kafka timing involved.

--via gateway (smoke only): POSTs each doc to /ingest and polls until every
doc has chunks. Exercises the real async path; never the metric path.

Never imports a service module (flat-name rule): the few SQL statements are
duplicated here and documented as mirrors.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

from evalkit.chunks import chunk_document
from evalkit.db import connect


def _bootstrap(orig_cwd: str, args, *attrs: str) -> None:
    """Fix working directory and resolve relative paths (see run.py)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    for attr in attrs:
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            base = orig_cwd if os.path.dirname(val) else here
            setattr(args, attr, os.path.join(base, val))
    os.chdir(here)


def ensure_schema(conn) -> None:
    """Mirror of services/ingestion/db.initialize_schema — same tables, same
    384-d column, same HNSW index. The worker owns the real schema; this only
    guarantees the eval seed works on a fresh database."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT        PRIMARY KEY,
                content     TEXT        NOT NULL,
                created_at  TIMESTAMP   NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id           SERIAL      PRIMARY KEY,
                doc_id       TEXT        NOT NULL
                                REFERENCES documents(id) ON DELETE CASCADE,
                chunk_text   TEXT        NOT NULL,
                chunk_index  INTEGER     NOT NULL,
                embedding    vector(384) NOT NULL,
                UNIQUE (doc_id, chunk_index)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunk_embedding_hnsw
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            """
        )


def seed_db(conn, corpus, chunk_size, chunk_overlap, model) -> int:
    """Upsert every corpus doc; return total chunk count."""
    total = 0
    for doc in corpus:
        doc_id, content = doc["doc_id"], doc["content"]
        texts = chunk_document(content, chunk_size, chunk_overlap)
        embeddings = model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).tolist()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, content, created_at)
                VALUES (%s, %s, now())
                ON CONFLICT (id)
                DO UPDATE SET content = EXCLUDED.content, created_at = now()
                """,
                (doc_id, content),
            )
            max_idx = len(texts) - 1
            cur.execute(
                "DELETE FROM document_chunks WHERE doc_id = %s AND chunk_index > %s",
                (doc_id, max_idx),
            )
            cur.executemany(
                """
                INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id, chunk_index)
                DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    embedding  = EXCLUDED.embedding
                """,
                [(doc_id, i, t, e) for i, (t, e) in enumerate(zip(texts, embeddings))],
            )
        total += len(texts)
        print(f"  {doc_id}: {len(texts)} chunks")
    return total


def seed_gateway(corpus, gateway_url, timeout_s=180) -> None:
    """POST each doc, then poll until every doc has at least one chunk."""
    for doc in corpus:
        body = json.dumps(
            {"doc_id": doc["doc_id"], "content": doc["content"]}
        ).encode()
        req = urllib.request.Request(
            gateway_url.rstrip("/") + "/ingest",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 202, f"ingest rejected {doc['doc_id']}"
        print(f"  accepted {doc['doc_id']}")
    conn = connect()
    try:
        deadline = time.time() + timeout_s
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc_id, COUNT(*) FROM document_chunks GROUP BY doc_id"
                )
                counts = dict(cur.fetchall())
            missing = [d["doc_id"] for d in corpus if not counts.get(d["doc_id"])]
            if not missing:
                print(f"  all {len(corpus)} docs chunked")
                return
            if time.time() > deadline:
                raise TimeoutError(f"still missing chunks for: {missing}")
            time.sleep(2)
    finally:
        conn.close()


def load_corpus(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    orig_cwd = os.getcwd()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--via", choices=("db", "gateway"), default="db")
    ap.add_argument("--gateway-url", default="http://localhost:8080")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()
    _bootstrap(orig_cwd, args, "corpus")

    corpus = load_corpus(args.corpus)
    if args.via == "gateway":
        seed_gateway(corpus, args.gateway_url)
        return
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    conn = connect()
    try:
        ensure_schema(conn)
        total = seed_db(conn, corpus, args.chunk_size, args.chunk_overlap, model)
    finally:
        conn.close()
    print(f"seeded {len(corpus)} docs, {total} chunks via db")


if __name__ == "__main__":
    main()
