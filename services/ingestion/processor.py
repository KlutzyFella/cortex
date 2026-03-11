"""Document processor: splits raw text into overlapping chunks."""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_document(
    content: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split *content* into a list of text chunks.

    Returns an empty list for blank or whitespace-only input rather than
    raising, so the caller can decide how to handle empty documents.
    """
    if not content or not content.strip():
        logger.warning("chunk_document received empty or whitespace-only content")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )
    return splitter.split_text(content)
