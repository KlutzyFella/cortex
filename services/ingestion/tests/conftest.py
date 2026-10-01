"""Shared pytest fixtures for ingestion service tests."""

import pytest
from config import IngestionConfig


@pytest.fixture(scope="session")
def test_config() -> IngestionConfig:
    """Configuration pointing at the local dev database.

    DB_PASSWORD must be set in the environment (or a .env file) when running
    integration tests.  For local development this is typically ``cortex_dev``.
    """
    import os

    return IngestionConfig(
        db_host=os.environ.get("DB_HOST", "localhost"),
        db_port=int(os.environ.get("DB_PORT", "5432")),
        db_user=os.environ.get("DB_USER", "cortex"),
        db_password=os.environ["DB_PASSWORD"],
        db_name=os.environ.get("DB_NAME", "cortex"),
        kafka_bootstrap_servers="localhost:9092",
        kafka_topic="document.uploaded",
        kafka_group_id="ingestion-worker-test",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
    )


@pytest.fixture(scope="session")
def embedding_model():
    """Load the sentence-transformer model once per test session."""
    from embedder import load_model

    return load_model("all-MiniLM-L6-v2")


@pytest.fixture()
def db_conn(test_config: IngestionConfig):
    """
    Yield a real psycopg connection for integration tests.

    Uses ``db.get_connection`` rather than ``psycopg.connect`` directly so the
    tests exercise the same path the ingestion worker does, including
    ``register_vector``. Connecting directly would skip that call and leave the
    production connection setup untested.

    The schema is initialised before the test and the entire connection is
    closed afterwards.  Each test should manage its own transaction.
    """
    pytest.importorskip("psycopg", reason="psycopg 3 not installed")

    from db import get_connection, initialize_schema

    conn = get_connection(test_config)
    initialize_schema(conn)
    yield conn
    conn.rollback()
    conn.close()
