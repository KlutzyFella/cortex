"""Database search layer: cosine similarity search via pgvector."""

import logging
from typing import TYPE_CHECKING

import psycopg
from pgvector.psycopg import register_vector

if TYPE_CHECKING:
    from config import RetrieverConfig

logger = logging.getLogger(__name__)


def get_connection(config: "RetrieverConfig") -> psycopg.Connection:
    """Return a new psycopg 3 connection with the pgvector type registered."""
    conn = psycopg.connect(config.db_dsn, autocommit=True)
    register_vector(conn)
    return conn


def search_similar_chunks(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int,
) -> list[dict]:
    """Return the *top_k* chunks most similar to *query_embedding*.

    Uses the pgvector cosine distance operator ``<=>`` (lower = more similar).
    The returned score is converted to cosine *similarity* (1 - distance) so
    callers receive a value in [0, 1] where 1 means identical.

    Args:
        conn:            Active psycopg connection (autocommit is fine).
        query_embedding: 384-dimensional query vector.
        top_k:           Maximum number of results to return.

    Returns:
        List of dicts with keys: ``chunk_id``, ``doc_id``, ``chunk_text``, ``score``.
    """
    if top_k <= 0:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id::text                          AS chunk_id,
                doc_id,
                chunk_text,
                1 - (embedding <=> %s::vector)    AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )
        rows = cur.fetchall()

    return [
        {
            "chunk_id": row[0],
            "doc_id": row[1],
            "chunk_text": row[2],
            "score": float(row[3]),
        }
        for row in rows
    ]
