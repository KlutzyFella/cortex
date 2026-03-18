"""Embedding layer: converts query text to a 384-dimensional vector."""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EXPECTED_DIM = 384


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and return a SentenceTransformer model.

    Validates the output dimension against the pgvector schema at startup so
    a misconfigured model name fails immediately rather than at query time.
    """
    logger.info("Loading embedding model '%s' …", model_name)
    model = SentenceTransformer(model_name)
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != _EXPECTED_DIM:
        raise ValueError(
            f"Model '{model_name}' produces {actual_dim}-dim vectors; "
            f"schema expects {_EXPECTED_DIM}."
        )
    logger.info("Embedding model loaded (dim=%d)", actual_dim)
    return model


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """Embed a single query string and return a flat float list.

    Args:
        model: Pre-loaded SentenceTransformer instance.
        query: The search query text.

    Returns:
        384-dimensional list of floats.
    """
    embedding = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    return embedding.tolist()
