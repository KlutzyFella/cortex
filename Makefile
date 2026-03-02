## Cortex — RAG-powered internal knowledge base
## ============================================================
## Usage: make <target>

PROTO_DIR   := proto
GATEWAY_GEN := gateway/gen
SERVICES    := services/ingestion services/retriever services/generator
COMPOSE     := infra/local/docker-compose.yml

.PHONY: proto up down clean help

## help       Show available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = "   "}; {printf "  \033[36m%-12s\033[0m%s\n", $$2, $$3}'

## proto       Lint protos and generate Go + Python gRPC stubs
proto: $(GATEWAY_GEN)
	@echo "==> Linting proto files..."
	buf lint $(PROTO_DIR)
	@echo "==> Generating Go stubs (via buf)..."
	buf generate
	@echo "==> Generating Python stubs (via grpcio-tools and uv)..."
	@for svc in $(SERVICES); do \
		mkdir -p $$svc/gen; \
		uv run python -m grpc_tools.protoc \
			-I$(PROTO_DIR) \
			--python_out=$$svc/gen \
			--grpc_python_out=$$svc/gen \
			$(PROTO_DIR)/cortex/v1/cortex.proto; \
		echo "  ✓ $$svc/gen"; \
	done
	@echo "==> Proto generation complete."

## up         Start local infrastructure (Postgres, Redis, Kafka)
up:
	docker compose -f $(COMPOSE) up -d

## down       Stop local infrastructure
down:
	docker compose -f $(COMPOSE) down

## clean      Remove all generated gRPC stubs
clean:
	rm -rf $(GATEWAY_GEN)
	@for svc in $(SERVICES); do rm -rf $$svc/gen; done

$(GATEWAY_GEN):
	mkdir -p $(GATEWAY_GEN)
