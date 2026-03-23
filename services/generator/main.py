"""
Cortex Generator Service

Starts a gRPC server on port 50052 implementing GeneratorService.GenerateAnswer:
  query + context_chunks → LLM (Claude) → grounded answer + citations
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

from config import GeneratorConfig
from server import GeneratorServicer


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

    config = GeneratorConfig.from_env()
    logger.info("Starting generator service on port %d", config.grpc_port)

    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),  # 8 MB
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
        ],
    )
    cortex_pb2_grpc.add_GeneratorServiceServicer_to_server(
        GeneratorServicer(config),
        grpc_server,
    )
    grpc_server.add_insecure_port(f"[::]:{config.grpc_port}")
    grpc_server.start()
    logger.info("Generator gRPC server listening on port %d", config.grpc_port)

    def _shutdown(signum, frame):
        logger.info("Shutdown signal received — stopping gRPC server …")
        grpc_server.stop(grace=10)  # 10-second grace for in-flight LLM calls
        logger.info("Generator stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    grpc_server.wait_for_termination()


if __name__ == "__main__":
    main()
