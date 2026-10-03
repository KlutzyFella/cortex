"""Unit tests for db.search_similar_chunks.

A fake connection/cursor stands in for psycopg: the SQL string, the parameter
tuple, and the row mapping are what is under test — not Postgres itself.
(No network, no database.)
"""

import db


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, rows):
        self.cursor_obj = FakeCursor(rows)
        self.cursor_calls = 0

    def cursor(self):
        self.cursor_calls += 1
        return self.cursor_obj


def _conn(rows=None):
    rows = rows if rows is not None else [
        ("7", "doc-a", "alpha text", 0.91),
        ("9", "doc-b", "beta text", 0.42),
    ]
    return FakeConn(rows)


def test_returns_mapped_rows_with_float_score():
    rows = db.search_similar_chunks(_conn(), [0.1] * 384, 5)

    assert rows == [
        {"chunk_id": "7", "doc_id": "doc-a", "chunk_text": "alpha text", "score": 0.91},
        {"chunk_id": "9", "doc_id": "doc-b", "chunk_text": "beta text", "score": 0.42},
    ]
    assert all(isinstance(r["score"], float) for r in rows)


def test_empty_result_set_maps_to_empty_list():
    assert db.search_similar_chunks(_conn(rows=[]), [0.1] * 384, 5) == []


def test_nonpositive_top_k_returns_empty_without_touching_db():
    for bad in (0, -1, -100):
        conn = _conn()
        assert db.search_similar_chunks(conn, [0.1] * 384, bad) == []
        assert conn.cursor_calls == 0


def test_sql_orders_by_cosine_distance_and_returns_similarity():
    conn = _conn()
    db.search_similar_chunks(conn, [0.1] * 384, 5)

    (sql, _), *_ = conn.cursor_obj.executed
    assert "embedding <=> %s::vector" in sql
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert "1 - (embedding <=> %s::vector)" in sql


def test_embedding_is_bound_twice_and_top_k_last():
    emb = [0.5] * 384
    conn = _conn()
    db.search_similar_chunks(conn, emb, 7)

    (_, params), *_ = conn.cursor_obj.executed
    assert params == (emb, emb, 7)
