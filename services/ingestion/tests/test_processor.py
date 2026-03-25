"""Unit tests for the document processor (text chunking)."""

import pytest

from processor import chunk_document


class TestChunkDocument:
    def test_short_text_returns_single_chunk(self) -> None:
        text = "This is a short sentence."
        chunks = chunk_document(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_returns_multiple_chunks(self) -> None:
        # Build a text clearly longer than chunk_size=100
        text = " ".join(["word"] * 200)
        chunks = chunk_document(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_each_chunk_within_size_limit(self) -> None:
        text = " ".join(["word"] * 500)
        chunk_size = 100
        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=10)
        for chunk in chunks:
            # LangChain may slightly exceed limit on word boundaries; allow 20% slack
            assert len(chunk) <= chunk_size * 1.2

    def test_adjacent_chunks_share_overlap(self) -> None:
        # With overlap > 0, the end of chunk[n] should appear at the start of chunk[n+1]
        text = "alpha " * 200
        chunks = chunk_document(text, chunk_size=100, chunk_overlap=20)
        if len(chunks) >= 2:
            # The last ~20 chars of chunk[0] should appear somewhere in chunk[1]
            tail = chunks[0][-15:]
            assert tail in chunks[1], "Expected overlap between adjacent chunks"

    def test_empty_string_returns_empty_list(self) -> None:
        assert chunk_document("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert chunk_document("   \n\t  ") == []

    def test_returns_list_of_strings(self) -> None:
        chunks = chunk_document("Hello world. " * 100, chunk_size=50, chunk_overlap=5)
        assert all(isinstance(c, str) for c in chunks)

    def test_default_chunk_size_and_overlap(self) -> None:
        # Should use 500/50 defaults without error
        text = "sentence. " * 300
        chunks = chunk_document(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
