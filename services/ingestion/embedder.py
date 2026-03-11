"""Vector embedder: wraps sentence-transformers for chunk embedding."""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


EXPECTED_EMBEDDING_DIM = 384


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and return a SentenceTransformer model.

    Validates that the model's output dimension matches the pgvector schema
    (384) and raises ``ValueError`` at startup rather than letting every
    ``upsert_chunks`` call fail with a dimension mismatch error.

    On first call the model weights (~80 MB) will be downloaded from
    HuggingFace Hub and cached locally.  Subsequent calls are fast.
    Call this once at startup and reuse the returned object.
    """
    logger.info("Loading embedding model '%s' …", model_name)
    model = SentenceTransformer(model_name)
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Model '{model_name}' produces {actual_dim}-dimensional vectors "
            f"but the schema expects {EXPECTED_EMBEDDING_DIM}. "
            "Update the schema or choose a compatible model."
        )
    logger.info("Embedding model loaded (dim=%d)", actual_dim)
    return model


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[str],
) -> list[list[float]]:
    """Return a 384-dimensional float vector for each chunk.

    Args:
        model: A pre-loaded SentenceTransformer instance.
        chunks: List of text strings to embed.

    Returns:
        A list of float lists, one per input chunk.  Empty input → empty output.
    """
    if not chunks:
        return []

    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    # Convert numpy array rows to plain Python lists of floats so callers have
    # no numpy dependency and the values are JSON-serialisable.
    return [embedding.tolist() for embedding in embeddings]
