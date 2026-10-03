"""Minimal evals database access.

Own connection helper (never imports a service `db` module — flat-name rule).
Mirrors the retriever's autocommit style: eval statements are single-shot, so
there is nothing to transact.
"""

import os

import psycopg
from pgvector.psycopg import register_vector


def connect() -> psycopg.Connection:
    conn = psycopg.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME", "cortex"),
        user=os.environ.get("DB_USER", "cortex"),
        password=os.environ["DB_PASSWORD"],
        autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn
