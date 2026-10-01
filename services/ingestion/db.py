"""Database layer: connection, schema initialisation, document and chunk persistence."""

import logging
from typing import TYPE_CHECKING

import psycopg
from pgvector.psycopg import register_vector

if TYPE_CHECKING:
    from config import IngestionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_connection(config: "IngestionConfig") -> psycopg.Connection:
    """Return a new psycopg 3 connection with the pgvector type registered.

    The ``vector`` extension is created first, in its own auto-committed
    transaction.  ``register_vector`` queries ``pg_type`` for the ``vector``
    type and raises ``ProgrammingError: vector type not found in the database``
    when the extension is absent, so on a fresh database the extension has to
    exist before registration is attempted.  Doing it here is what makes the
    documented "schema is created on first connect" behaviour true.
    """
    conn = psycopg.connect(config.db_dsn, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def initialize_schema(conn: psycopg.Connection) -> None:
    """Create the pgvector extension, tables, and HNSW index if absent.

    Safe to call on every startup — all statements use IF NOT EXISTS.
    """
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

    # Schema DDL is auto-committed separately from application data so that
    # callers can manage their own transaction boundaries cleanly.
    conn.commit()
    logger.info("Database schema initialised")


# ---------------------------------------------------------------------------
# Document persistence
# ---------------------------------------------------------------------------


def insert_document(conn: psycopg.Connection, doc_id: str, content: str) -> None:
    """Upsert a raw document record.

    If a document with the same *doc_id* already exists, its content and
    *created_at* timestamp are updated so re-ingestion is idempotent.
    """
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


# ---------------------------------------------------------------------------
# Chunk persistence
# ---------------------------------------------------------------------------


def upsert_chunks(
    conn: psycopg.Connection,
    doc_id: str,
    chunks: list[tuple[int, str, list[float]]],
) -> None:
    """Upsert a batch of (chunk_index, chunk_text, embedding) tuples.

    Stale chunks (those with a higher index than the current batch) are
    deleted first so that re-ingesting a shorter document does not leave
    orphaned rows behind.

    Args:
        conn:   Active psycopg connection (caller manages the transaction).
        doc_id: Parent document identifier.
        chunks: List of (chunk_index, chunk_text, embedding) tuples.
    """
    if not chunks:
        logger.warning("upsert_chunks called with empty chunk list for doc '%s'", doc_id)
        return

    max_new_index = max(idx for idx, _, _ in chunks)

    with conn.cursor() as cur:
        # Remove stale high-index chunks from a previous, longer version of the doc.
        cur.execute(
            "DELETE FROM document_chunks WHERE doc_id = %s AND chunk_index > %s",
            (doc_id, max_new_index),
        )

        # Batch upsert all new chunks.
        cur.executemany(
            """
            INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id, chunk_index)
            DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding  = EXCLUDED.embedding
            """,
            [
                (doc_id, chunk_index, chunk_text, embedding)
                for chunk_index, chunk_text, embedding in chunks
            ],
        )
