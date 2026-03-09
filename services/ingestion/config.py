"""
Centralised configuration for the ingestion worker.

All settings are read from environment variables with sane defaults so that
the service can be started locally without any extra setup.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionConfig:
    # PostgreSQL
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str

    # Kafka
    kafka_bootstrap_servers: str
    kafka_topic: str
    kafka_group_id: str

    # Embedding model
    embedding_model: str

    # Chunking
    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

    @classmethod
    def from_env(cls) -> "IngestionConfig":
        # DB_PASSWORD has no default — fail fast if the secret is absent
        # rather than connecting with a known credential.
        return cls(
            db_host=os.environ.get("DB_HOST", "localhost"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_user=os.environ.get("DB_USER", "cortex"),
            db_password=os.environ["DB_PASSWORD"],
            db_name=os.environ.get("DB_NAME", "cortex"),
            kafka_bootstrap_servers=os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
            ),
            kafka_topic=os.environ.get("KAFKA_TOPIC", "document.uploaded"),
            kafka_group_id=os.environ.get("KAFKA_GROUP_ID", "ingestion-worker"),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "50")),
        )

    @property
    def db_dsn(self) -> str:
        # Never log this property — it contains the database password.
        return (
            f"host={self.db_host} port={self.db_port} "
            f"dbname={self.db_name} user={self.db_user} "
            f"password={self.db_password}"
        )
