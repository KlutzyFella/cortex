"""Integration tests for the database layer.

These tests require a running PostgreSQL 16 instance with the pgvector
extension installed and the credentials in conftest.py.

Run with:
    pytest -m integration services/ingestion/tests/test_db.py
"""

import pytest

pytestmark = pytest.mark.integration

DUMMY_EMBEDDING = [0.1] * 384


class TestInitializeSchema:
    def test_creates_documents_table(self, db_conn) -> None:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.documents')::text"
            )
            assert cur.fetchone()[0] == "documents"

    def test_creates_document_chunks_table(self, db_conn) -> None:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.document_chunks')::text"
            )
            assert cur.fetchone()[0] == "document_chunks"

    def test_creates_hnsw_index(self, db_conn) -> None:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'document_chunks'
                  AND indexname = 'idx_chunk_embedding_hnsw'
                """
            )
            row = cur.fetchone()
            assert row is not None, "HNSW index not found"

    def test_idempotent_on_repeated_calls(self, db_conn) -> None:
        from db import initialize_schema

        # Should not raise even when called twice
        initialize_schema(db_conn)
        initialize_schema(db_conn)


class TestInsertDocument:
    def test_inserts_document(self, db_conn) -> None:
        from db import insert_document

        insert_document(db_conn, "doc-insert-1", "Hello content")
        db_conn.commit()

        with db_conn.cursor() as cur:
            cur.execute("SELECT content FROM documents WHERE id = %s", ("doc-insert-1",))
            row = cur.fetchone()

        assert row is not None
        assert row[0] == "Hello content"

        # Cleanup
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", ("doc-insert-1",))
        db_conn.commit()

    def test_upserts_on_duplicate_doc_id(self, db_conn) -> None:
        from db import insert_document

        insert_document(db_conn, "doc-upsert-1", "original")
        db_conn.commit()
        insert_document(db_conn, "doc-upsert-1", "updated")
        db_conn.commit()

        with db_conn.cursor() as cur:
            cur.execute("SELECT content FROM documents WHERE id = %s", ("doc-upsert-1",))
            row = cur.fetchone()

        assert row[0] == "updated"

        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", ("doc-upsert-1",))
        db_conn.commit()


class TestUpsertChunks:
    def _insert_parent(self, conn, doc_id: str) -> None:
        from db import insert_document

        insert_document(conn, doc_id, "parent doc")
        conn.commit()

    def _cleanup(self, conn, doc_id: str) -> None:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE doc_id = %s", (doc_id,))
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()

    def test_inserts_chunks(self, db_conn) -> None:
        from db import upsert_chunks

        doc_id = "doc-chunk-insert"
        self._insert_parent(db_conn, doc_id)

        chunks = [(0, "first chunk", DUMMY_EMBEDDING), (1, "second chunk", DUMMY_EMBEDDING)]
        upsert_chunks(db_conn, doc_id, chunks)
        db_conn.commit()

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_index, chunk_text FROM document_chunks "
                "WHERE doc_id = %s ORDER BY chunk_index",
                (doc_id,),
            )
            rows = cur.fetchall()

        assert len(rows) == 2
        assert rows[0] == (0, "first chunk")
        assert rows[1] == (1, "second chunk")

        self._cleanup(db_conn, doc_id)

    def test_upsert_is_idempotent(self, db_conn) -> None:
        from db import upsert_chunks

        doc_id = "doc-chunk-idempotent"
        self._insert_parent(db_conn, doc_id)

        chunks = [(0, "chunk text", DUMMY_EMBEDDING)]
        upsert_chunks(db_conn, doc_id, chunks)
        db_conn.commit()
        upsert_chunks(db_conn, doc_id, chunks)
        db_conn.commit()

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE doc_id = %s", (doc_id,)
            )
            count = cur.fetchone()[0]

        assert count == 1

        self._cleanup(db_conn, doc_id)

    def test_stale_chunks_are_removed_on_re_ingestion(self, db_conn) -> None:
        from db import upsert_chunks

        doc_id = "doc-chunk-stale"
        self._insert_parent(db_conn, doc_id)

        # First ingest: 3 chunks
        chunks_v1 = [
            (0, "chunk 0", DUMMY_EMBEDDING),
            (1, "chunk 1", DUMMY_EMBEDDING),
            (2, "chunk 2", DUMMY_EMBEDDING),
        ]
        upsert_chunks(db_conn, doc_id, chunks_v1)
        db_conn.commit()

        # Re-ingest with only 1 chunk (document got shorter)
        chunks_v2 = [(0, "only chunk", DUMMY_EMBEDDING)]
        upsert_chunks(db_conn, doc_id, chunks_v2)
        db_conn.commit()

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_index FROM document_chunks WHERE doc_id = %s ORDER BY chunk_index",
                (doc_id,),
            )
            rows = cur.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == 0

        self._cleanup(db_conn, doc_id)

    def test_embedding_dimension_stored_correctly(self, db_conn) -> None:
        from db import upsert_chunks

        doc_id = "doc-chunk-dim"
        self._insert_parent(db_conn, doc_id)

        chunks = [(0, "embedding test", DUMMY_EMBEDDING)]
        upsert_chunks(db_conn, doc_id, chunks)
        db_conn.commit()

        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT vector_dims(embedding) FROM document_chunks WHERE doc_id = %s",
                (doc_id,),
            )
            row = cur.fetchone()

        assert row[0] == 384

        self._cleanup(db_conn, doc_id)

    def test_foreign_key_constraint_enforced(self, db_conn) -> None:
        import psycopg
        from db import upsert_chunks

        chunks = [(0, "orphan chunk", DUMMY_EMBEDDING)]
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            upsert_chunks(db_conn, "nonexistent-doc", chunks)
            db_conn.commit()

        db_conn.rollback()
