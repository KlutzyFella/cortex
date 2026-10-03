"""Unit tests for the RetrieverServicer gRPC contract.

embed_query and get_connection are monkeypatched: no model download, no
database. The real search_similar_chunks runs against the fake connection,
so the SQL-building behaviour in db.py is exercised through the server path.
"""

from types import SimpleNamespace

import grpc
import pytest
import server
from config import RetrieverConfig


def _config():
    return RetrieverConfig(
        db_host="localhost",
        db_port=5432,
        db_user="cortex",
        db_password="test-only",
        db_name="cortex",
        grpc_port=50051,
        embedding_model="all-MiniLM-L6-v2",
    )


class FakeContext:
    def __init__(self):
        self.code = grpc.StatusCode.OK
        self.details = ""

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class FakeCursor:
    def __init__(self):
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.params = params

    def fetchall(self):
        return [
            ("11", "doc-x", "x marks the spot", 0.77),
            ("12", "doc-y", "y why", 0.31),
        ]


class FakeConn:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def close(self):
        self.closed = True


class Harness:
    """Wires a servicer with stubbed embed + DB layers and records calls."""

    def __init__(self, monkeypatch, embedding=None, db_error=None):
        self.embedding = embedding if embedding is not None else [0.2] * 384
        self.embed_calls = []
        self.conn = FakeConn()
        self.db_error = db_error

        def fake_embed(model, query):
            self.embed_calls.append(query)
            if isinstance(self.embedding, Exception):
                raise self.embedding
            return self.embedding

        def fake_connect(config):
            if isinstance(self.db_error, Exception):
                raise self.db_error
            return self.conn

        monkeypatch.setattr(server, "embed_query", fake_embed)
        monkeypatch.setattr(server, "get_connection", fake_connect)
        self.servicer = server.RetrieverServicer(_config(), object())

    def search(self, query, top_k=0):
        ctx = FakeContext()
        resp = self.servicer.SearchDocuments(
            SimpleNamespace(query=query, top_k=top_k), ctx
        )
        return resp, ctx


def test_blank_query_is_invalid_and_never_embeds(monkeypatch):
    for bad in ("", "   "):
        h = Harness(monkeypatch)
        resp, ctx = h.search(bad)

        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "non-empty" in ctx.details
        assert h.embed_calls == []
        assert len(resp.chunks) == 0


def test_omitted_top_k_falls_back_to_five(monkeypatch):
    h = Harness(monkeypatch)
    resp, ctx = h.search("hello", top_k=0)

    assert ctx.code == grpc.StatusCode.OK
    assert h.conn.cursor_obj.params == (h.embedding, h.embedding, 5)
    assert [c.chunk_id for c in resp.chunks] == ["11", "12"]


def test_negative_top_k_falls_back_to_five(monkeypatch):
    h = Harness(monkeypatch)
    _, ctx = h.search("hello", top_k=-3)

    assert ctx.code == grpc.StatusCode.OK
    assert h.conn.cursor_obj.params == (h.embedding, h.embedding, 5)


def test_explicit_top_k_passes_through(monkeypatch):
    h = Harness(monkeypatch)
    _, ctx = h.search("hello", top_k=3)

    assert ctx.code == grpc.StatusCode.OK
    assert h.conn.cursor_obj.params == (h.embedding, h.embedding, 3)


def test_chunks_map_to_proto_fields(monkeypatch):
    h = Harness(monkeypatch)
    resp, _ = h.search("hello")

    first = resp.chunks[0]
    assert (first.chunk_id, first.document_id) == ("11", "doc-x")
    assert first.content == "x marks the spot"
    # Proto `float` is 32-bit: 0.77 arrives as 0.76999998. approx documents
    # the precision contract rather than fighting it.
    assert first.score == pytest.approx(0.77)
    assert h.conn.closed is True


def test_embedding_failure_is_internal(monkeypatch):
    h = Harness(monkeypatch, embedding=RuntimeError("model blew up"))
    resp, ctx = h.search("hello")

    assert ctx.code == grpc.StatusCode.INTERNAL
    assert "embedding" in ctx.details
    assert len(resp.chunks) == 0


def test_database_failure_is_internal(monkeypatch):
    h = Harness(monkeypatch, db_error=RuntimeError("pgvector down"))
    resp, ctx = h.search("hello")

    assert ctx.code == grpc.StatusCode.INTERNAL
    assert "database" in ctx.details
    assert len(resp.chunks) == 0
