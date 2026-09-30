"""gRPC service implementation for the Retriever."""

import logging
import os
import sys

# Ensure the gen/ directory is on sys.path so the protobuf stubs resolve.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))

import grpc
from config import RetrieverConfig
from cortex.v1 import cortex_pb2, cortex_pb2_grpc
from db import get_connection, search_similar_chunks
from embedder import embed_query

logger = logging.getLogger(__name__)


class RetrieverServicer(cortex_pb2_grpc.RetrieverServiceServicer):
    """Implements the RetrieverService gRPC contract."""

    def __init__(self, config: RetrieverConfig, model) -> None:
        self._config = config
        self._model = model

    def SearchDocuments(self, request, context):
        query = request.query
        top_k = request.top_k if request.top_k > 0 else 5

        if not query or not query.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query must be a non-empty string")
            return cortex_pb2.SearchDocumentsResponse()

        logger.info("SearchDocuments query=%r top_k=%d", query[:100], top_k)

        # 1. Embed the query
        try:
            query_embedding = embed_query(self._model, query)
        except Exception as exc:
            logger.exception("Embedding failed: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("embedding service error")
            return cortex_pb2.SearchDocumentsResponse()

        # 2. Search pgvector
        try:
            conn = get_connection(self._config)
            try:
                rows = search_similar_chunks(conn, query_embedding, top_k)
            finally:
                conn.close()
        except Exception as exc:
            logger.exception("Database search failed: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("database error")
            return cortex_pb2.SearchDocumentsResponse()

        # 3. Map to protobuf response
        chunks = [
            cortex_pb2.DocumentChunk(
                chunk_id=row["chunk_id"],
                document_id=row["doc_id"],
                content=row["chunk_text"],
                score=row["score"],
            )
            for row in rows
        ]

        logger.info("Returning %d chunks for query=%r", len(chunks), query[:100])
        return cortex_pb2.SearchDocumentsResponse(chunks=chunks)
