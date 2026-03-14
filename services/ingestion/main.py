"""
Cortex Ingestion Worker

Consumes `document.uploaded` events from Kafka and runs each document through
the full RAG ingestion pipeline:

    Kafka → parse → chunk → embed → pgvector upsert

Each message is processed inside its own try/except block so that a single
bad payload never crashes the worker.  Kafka offsets are committed manually
only after a successful DB commit, preventing silent data loss on crash.
"""

import json
import logging
import os
import re
import sys
import time
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import psycopg
from confluent_kafka import Consumer, KafkaError, KafkaException

from config import IngestionConfig
from db import get_connection, initialize_schema, insert_document, upsert_chunks
from embedder import embed_chunks, load_model
from processor import chunk_document

# Control characters that must not appear in logged values from untrusted input.
_UNSAFE_LOG_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_MAX_LOG_STR_LEN = 200


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


logger = logging.getLogger(__name__)


def _sanitize_for_log(value: str) -> str:
    """Strip control characters and truncate untrusted strings before logging."""
    cleaned = _UNSAFE_LOG_CHARS.sub("?", value)
    if len(cleaned) > _MAX_LOG_STR_LEN:
        cleaned = cleaned[:_MAX_LOG_STR_LEN] + "…"
    return cleaned


def _parse_message(raw: bytes) -> dict[str, Any] | None:
    """Decode and validate a Kafka message payload.

    Returns the parsed dict on success, or None if the message is malformed.
    Validates that doc_id and content are both non-empty strings.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("Failed to decode message as JSON: %s | raw=%r", exc, raw[:200])
        return None

    if not isinstance(payload, dict):
        logger.error("Message payload is not a JSON object: %r", str(payload)[:200])
        return None

    for key in ("doc_id", "content"):
        val = payload.get(key)
        if not isinstance(val, str):
            logger.error(
                "Field '%s' must be a non-null string, got %s",
                key,
                type(val).__name__,
            )
            return None

    return payload


def _get_connection_with_retry(
    config: IngestionConfig,
    max_attempts: int = 5,
    backoff_seconds: float = 2.0,
) -> psycopg.Connection:
    """Connect to PostgreSQL, retrying with exponential backoff on failure."""
    for attempt in range(1, max_attempts + 1):
        try:
            conn = get_connection(config)
            logger.info("Database connection established (attempt %d)", attempt)
            return conn
        except psycopg.OperationalError as exc:
            if attempt == max_attempts:
                raise
            wait = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "DB connection failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt,
                max_attempts,
                exc,
                wait,
            )
            time.sleep(wait)

    # Unreachable, but satisfies the type checker.
    raise RuntimeError("Failed to connect to database")  # pragma: no cover


def _ensure_connection(
    conn: psycopg.Connection | None,
    config: IngestionConfig,
) -> psycopg.Connection:
    """Return a healthy connection, reconnecting if the existing one is closed."""
    if conn is None or conn.closed:
        logger.warning("DB connection lost — reconnecting …")
        return _get_connection_with_retry(config)
    return conn


def _process_message(
    payload: dict[str, Any],
    model: Any,
    conn: psycopg.Connection,
    config: IngestionConfig,
) -> None:
    """Run a single document through the full ingestion pipeline.

    The caller must commit or rollback the connection after this returns.
    """
    doc_id: str = payload["doc_id"]
    content: str = payload["content"]
    safe_doc_id = _sanitize_for_log(doc_id)

    # 1. Persist raw document
    insert_document(conn, doc_id, content)

    # 2. Chunk
    chunks = chunk_document(content, config.chunk_size, config.chunk_overlap)
    if not chunks:
        logger.warning("No chunks produced for doc '%s' — skipping embed/upsert", safe_doc_id)
        conn.commit()
        return

    # 3. Embed
    embeddings = embed_chunks(model, chunks)

    # 4. Upsert chunks + embeddings in the same transaction as insert_document
    chunk_tuples = [
        (idx, chunk_text, embedding)
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
    ]
    upsert_chunks(conn, doc_id, chunk_tuples)

    conn.commit()

    logger.info("Successfully indexed %d chunks for %s", len(chunks), safe_doc_id)


def main() -> None:
    setup_logging()

    config = IngestionConfig.from_env()
    logger.info("Starting ingestion worker (topic=%s)", config.kafka_topic)

    model = load_model(config.embedding_model)
    conn: psycopg.Connection | None = _get_connection_with_retry(config)
    initialize_schema(conn)

    consumer = Consumer(
        {
            "bootstrap.servers": config.kafka_bootstrap_servers,
            "group.id": config.kafka_group_id,
            "auto.offset.reset": "earliest",
            # Manual offset commits: only advance the offset after a confirmed
            # DB commit so that a crash mid-processing causes re-delivery.
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([config.kafka_topic])
    logger.info("Subscribed to Kafka topic '%s'", config.kafka_topic)

    try:
        while True:
            try:
                msg = consumer.poll(timeout=1.0)
            except KafkaException as exc:
                logger.critical("Fatal Kafka error during poll: %s — shutting down", exc)
                break

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(
                        "Reached end of partition %s/%d offset %d",
                        msg.topic(),
                        msg.partition(),
                        msg.offset(),
                    )
                else:
                    logger.error("Kafka consumer error: %s", msg.error())
                continue

            payload = _parse_message(msg.value())
            if payload is None:
                # Malformed message: commit offset to skip it permanently.
                consumer.commit(message=msg)
                continue

            conn = _ensure_connection(conn, config)

            try:
                _process_message(payload, model, conn, config)
                # Only commit the Kafka offset after the DB transaction succeeds.
                consumer.commit(message=msg)
            except Exception:
                safe_id = _sanitize_for_log(str(payload.get("doc_id", "<unknown>")))
                logger.exception(
                    "Unhandled error processing doc '%s' — rolling back and continuing",
                    safe_id,
                )
                conn.rollback()
                # Do NOT commit the Kafka offset: the message will be re-delivered.

    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        logger.info("Closing consumer and database connection")
        consumer.close()
        if conn and not conn.closed:
            conn.close()


if __name__ == "__main__":
    main()
