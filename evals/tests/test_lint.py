"""Unit tests for lint_golden.py on synthetic fixtures (no real corpus)."""

from lint_golden import lint

DOC_A = (
    "Postgres B-tree indexes accelerate equality and range scans on sortable "
    "columns. The planner consults pg_stats histograms before choosing a scan."
)
DOC_B = (
    "Postgres pg_stats histograms inform the query planner about sortable "
    "column distributions. Sequential scans surface when statistics mislead."
)


def _corpus():
    return [
        {"doc_id": "doc-a", "content": DOC_A},
        {"doc_id": "doc-b", "content": DOC_B},
    ]


def _positive(query, gold):
    return {"query": query, "gold": gold, "answer": "x", "negative": False}


Q = "How do B-tree structures speed up planner lookups on sortable columns?"


def test_clean_positive_passes():
    item = _positive(
        Q,
        [{"doc_id": "doc-a", "chunk_index": 0}],
    )
    assert lint(_corpus(), [item]) == []


def test_verbatim_copy_fails():
    item = _positive(DOC_A, [{"doc_id": "doc-a", "chunk_index": 0}])
    errors = lint(_corpus(), [item])
    assert any("verbatim" in e for e in errors)


def test_missing_distractor_fails():
    lonely = [
        {"doc_id": "only", "content": "Zebra xylophone quantum juxtapose."}
    ]
    item = _positive(
        "What does zebra xylophone quantum state?",
        [{"doc_id": "only", "chunk_index": 0}],
    )
    errors = lint(lonely, [item])
    assert any("distractor" in e for e in errors)


def test_bad_chunk_index_fails():
    item = _positive(
        Q,
        [{"doc_id": "doc-a", "chunk_index": 9}],
    )
    assert any("out of range" in e for e in lint(_corpus(), [item]))


def test_unknown_doc_fails():
    item = _positive(
        Q,
        [{"doc_id": "doc-zzz", "chunk_index": 0}],
    )
    assert any("unknown doc" in e for e in lint(_corpus(), [item]))


def test_negative_with_gold_fails():
    item = {
        "query": "What color is the Postgres elephant?",
        "gold": [{"doc_id": "doc-a", "chunk_index": 0}],
        "answer": "",
        "negative": True,
    }
    assert any("empty gold" in e for e in lint(_corpus(), [item]))


def test_clean_negative_passes():
    item = {
        "query": "What color is the Postgres elephant?",
        "gold": [],
        "answer": "",
        "negative": True,
    }
    assert lint(_corpus(), [item]) == []


def test_empty_query_fails():
    assert any(
        "empty query" in e for e in lint(_corpus(), [_positive("", [])])
    )
