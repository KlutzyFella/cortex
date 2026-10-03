"""Chunking for evals — a deliberate mirror of the ingestion worker.

Uses the identical splitter configuration as services/ingestion/processor.py
(RecursiveCharacterTextSplitter, length_function=len, no start index) so the
(chunk_index) labels in golden.jsonl mean the same thing the database stores.
If ingestion's chunking changes, golden.meta.json pins the old parameters and
the harness refuses to run until the golden set is regenerated — that coupling
is intentional and loud, not silent.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_document(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split *content* exactly the way the ingestion worker does."""
    if not content or not content.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )
    return splitter.split_text(content)


def chunk_corpus(
    corpus: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, int, str]]:
    """Return [(doc_id, chunk_index, chunk_text)] for every doc in *corpus*."""
    out = []
    for doc in corpus:
        for i, text in enumerate(chunk_document(doc["content"], chunk_size, chunk_overlap)):
            out.append((doc["doc_id"], i, text))
    return out
