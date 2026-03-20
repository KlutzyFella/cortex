"""
Cortex Retriever Service

Starts a gRPC server on port 50051 implementing RetrieverService.SearchDocuments:
  query text → embed → pgvector cosine search → DocumentChunks
"""

import logging
import os
import signal
import sys
from concurrent import futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))

import grpc
from cortex.v1 import cortex_pb2_grpc

from config import RetrieverConfig
from embedder import load_model
from server import RetrieverServicer


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    config = RetrieverConfig.from_env()
    logger.info("Starting retriever service on port %d", config.grpc_port)

    model = load_model(config.embedding_model)

    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),  # 4 MB
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
        ],
    )
    cortex_pb2_grpc.add_RetrieverServiceServicer_to_server(
        RetrieverServicer(config, model),
        grpc_server,
    )
    grpc_server.add_insecure_port(f"[::]:{config.grpc_port}")
    grpc_server.start()
    logger.info("Retriever gRPC server listening on port %d", config.grpc_port)

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping gRPC server …")
        grpc_server.stop(grace=5)  # 5-second grace period for in-flight RPCs
        logger.info("Retriever stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    grpc_server.wait_for_termination()


if __name__ == "__main__":
    main()
