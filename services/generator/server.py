"""gRPC service implementation for the Generator."""

import logging
import os
import sys

# Ensure gen/ is on sys.path so protobuf stubs resolve.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))

import grpc
from config import GeneratorConfig
from cortex.v1 import cortex_pb2, cortex_pb2_grpc
from llm import build_chain, generate_response

logger = logging.getLogger(__name__)


class GeneratorServicer(cortex_pb2_grpc.GeneratorServiceServicer):
    """Implements the GeneratorService gRPC contract."""

    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config
        self._chain = build_chain(
            config.provider, config.api_key, config.default_model
        )
        logger.info("LLM chain initialised with model '%s'", config.default_model)

    def GenerateAnswer(self, request, context):
        query = request.query
        context_chunks = list(request.context_chunks)  # repeated DocumentChunk

        if not query or not query.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query must be a non-empty string")
            return cortex_pb2.GenerateAnswerResponse()

        if not context_chunks:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("context_chunks must not be empty")
            return cortex_pb2.GenerateAnswerResponse()

        # Use the model override from the request if provided, otherwise keep the default.
        # The override stays within the configured provider: a model ID alone
        # does not say which API serves it.
        model_name = request.model if request.model else self._config.default_model
        if model_name != self._config.default_model:
            chain = build_chain(
                self._config.provider, self._config.api_key, model_name
            )
        else:
            chain = self._chain

        logger.info(
            "GenerateAnswer query=%r chunks=%d model=%s",
            query[:100],
            len(context_chunks),
            model_name,
        )

        # Convert proto DocumentChunk messages to plain dicts for llm.py.
        chunks = [
            {"chunk_id": c.chunk_id, "content": c.content}
            for c in context_chunks
        ]
        valid_chunk_ids = {c["chunk_id"] for c in chunks}

        try:
            result = generate_response(chain, query, chunks, valid_chunk_ids)
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("LLM generation error")
            return cortex_pb2.GenerateAnswerResponse()

        # Map citations back to protobuf, looking up document_id from the input chunks.
        chunk_doc_map = {c.chunk_id: c.document_id for c in context_chunks}
        proto_citations = [
            cortex_pb2.Citation(
                chunk_id=cit.chunk_id,
                document_id=chunk_doc_map.get(cit.chunk_id, ""),
                excerpt=cit.excerpt,
            )
            for cit in result.citations
        ]

        logger.info(
            "GenerateAnswer complete: %d chars, %d citations, grounded=%s",
            len(result.answer),
            len(proto_citations),
            result.grounded,
        )

        return cortex_pb2.GenerateAnswerResponse(
            answer=result.answer,
            citations=proto_citations,
            grounded=result.grounded,
        )
