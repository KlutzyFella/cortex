# Cortex

> **A production-grade, polyglot RAG engine.** Ingest unstructured documents, embed them mathematically, and answer natural-language queries with strictly grounded, hallucination-free responses — every answer backed by citations to the exact source passages.

![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Kafka](https://img.shields.io/badge/Kafka-Redpanda-FF6B35?style=flat-square&logo=apachekafka&logoColor=white)
![gRPC](https://img.shields.io/badge/gRPC-Protobuf-244c5a?style=flat-square&logo=google&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What is Cortex?

Cortex is a **Retrieval-Augmented Generation (RAG)** system built as a polyglot monorepo. It combines a high-concurrency Go API gateway with three specialised Python microservices communicating over gRPC, backed by a vector-native PostgreSQL database and an event-driven Kafka pipeline.

The system is designed around one guarantee: **every answer is grounded.** The Generator service performs citation validation at the structural level — any claim the LLM makes that cannot be traced back to a chunk actually provided in the request is stripped from the response before it reaches the caller.

![Architecture Diagram](docs/architecture.png)

---

## System Architecture

### Ingestion Pipeline

```
Client
  │
  └─▶  POST /ingest  ──▶  Go Gateway
                              │
                              └─▶  Kafka  (topic: document.uploaded)
                                      │
                                      └─▶  Ingestion Worker (Python)
                                                │
                                                ├─▶  LangChain RecursiveCharacterTextSplitter
                                                │       chunk_size=500, overlap=50
                                                │
                                                ├─▶  SentenceTransformers  all-MiniLM-L6-v2
                                                │       → 384-dim dense vectors
                                                │
                                                └─▶  PostgreSQL + pgvector
                                                        tables: documents, document_chunks
                                                        index:  HNSW cosine
```

### Query Pipeline

```
Client
  │
  └─▶  POST /api/v1/query  ──▶  Go Gateway
                                    │
                                    ├─▶  gRPC: RetrieverService.SearchDocuments  (:50051)
                                    │           │
                                    │           └─▶  pgvector  <=>  cosine distance
                                    │                   → top-k DocumentChunks + scores
                                    │
                                    └─▶  gRPC: GeneratorService.GenerateAnswer   (:50052)
                                                │
                                                ├─▶  RAG prompt: context chunks + query
                                                ├─▶  Google Gemini 2.5 Flash  (via LangChain)
                                                ├─▶  Structured output  (Pydantic schema)
                                                └─▶  Citation validation
                                                        strip hallucinated chunk_ids
                                                        set grounded=true if ≥1 valid citation
                                                    ↓
                                               { answer, citations, grounded, chunks }
```

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Polyglot microservices** | Go gateway for throughput-critical routing; Python services for ML-heavy workloads |
| **Event-driven ingestion** | Kafka decouples document upload from indexing — the gateway never blocks on embedding |
| **HNSW vector index** | pgvector's Hierarchical Navigable Small World index gives sub-millisecond approximate nearest-neighbour search at scale |
| **Grounded generation** | Structured LLM output parsed via Pydantic; every citation cross-referenced against the actual chunks in the request — hallucinated references are stripped automatically |
| **Contract-first API** | All inter-service communication defined in a single `proto/cortex/v1/cortex.proto`; stubs generated for both Go and Python via `buf` |
| **Monorepo workspace** | `uv` workspaces unify three Python services under one lockfile; a root `pyproject.toml` captures shared ML dependencies |
| **Zero-trust defaults** | No hardcoded secrets; every service fails fast at startup with a descriptive error if a required env var is absent |
| **Manual Kafka commits** | The ingestion worker commits offsets only after a successful database transaction — a crash mid-processing causes re-delivery, not silent data loss |

---

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Docker + Compose | 24+ | Runs Postgres, Redpanda, Redis |
| Go | 1.22+ | Builds the API gateway |
| Python | 3.12+ | All three Python services |
| [`uv`](https://docs.astral.sh/uv/) | 0.4+ | Python workspace & dependency management |
| Buf CLI | 1.x | Only needed to regenerate protobuf stubs |
| Google Gemini API Key | — | `gemini-2.5-flash` model access required |

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/KlutzyFella/cortex.git
cd cortex

# Install all Python workspace dependencies into a shared .venv
uv sync
```

### 2. Start infrastructure

```bash
docker compose -f infra/local/docker-compose.yml up -d
```

This starts:
- **PostgreSQL 16** with the `pgvector` extension on `localhost:5432`
- **Redpanda** (Kafka-compatible) on `localhost:9092`
- **Redis** on `localhost:6379`

Wait ~10 seconds for Redpanda to finish its health check before proceeding.

### 3. Start the Go gateway

```bash
cd gateway
GATEWAY_PORT=8080 \
KAFKA_ADDR=localhost:9092 \
RETRIEVER_ADDR=localhost:50051 \
GENERATOR_ADDR=localhost:50052 \
go run .
```

```bash
# Verify
curl http://localhost:8080/healthz
# {"status":"ok"}
```

### 4. Start the Retriever service

```bash
DB_PASSWORD=cortex_dev \
uv run --package retriever python services/retriever/main.py
```

### 5. Start the Ingestion worker

```bash
DB_PASSWORD=cortex_dev \
uv run --package ingestion python services/ingestion/main.py
```

### 6. Start the Generator service

```bash
GOOGLE_API_KEY=<your_key> \
uv run --package generator python services/generator/main.py
```

All four processes are now live. The system is ready to ingest documents and answer queries.

---

## API Reference

### `POST /ingest` — Ingest a document

Publishes the document to Kafka. The ingestion worker picks it up asynchronously, chunks it, embeds it, and writes it to pgvector. Returns immediately with `202 Accepted`.

```bash
curl -s -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "go-concurrency-guide",
    "content": "Go achieves concurrency through goroutines — lightweight threads managed by the Go runtime. Unlike OS threads, goroutines are multiplexed across a small number of OS threads by the scheduler, making it practical to spawn thousands simultaneously. Communication between goroutines is encouraged via channels rather than shared memory, following the principle: do not communicate by sharing memory; instead, share memory by communicating."
  }'
```

**Response `202 Accepted`:**
```json
{
  "status": "accepted",
  "doc_id": "go-concurrency-guide"
}
```

---

### `POST /api/v1/query` — Query the knowledge base

Runs the full RAG pipeline synchronously: vector search → LLM generation → citation validation.

```bash
curl -s -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Go handle concurrency?",
    "top_k": 3
  }' | jq
```

**Response `200 OK`:**
```json
{
  "answer": "Go handles concurrency through goroutines, which are lightweight threads managed by the Go runtime rather than the operating system. The scheduler multiplexes goroutines across a small number of OS threads, making it practical to run thousands concurrently. The idiomatic approach to coordination is via channels, following the principle of sharing memory by communicating rather than communicating by sharing memory.",
  "grounded": true,
  "citations": [
    {
      "chunk_id": "42",
      "document_id": "go-concurrency-guide",
      "excerpt": "goroutines — lightweight threads managed by the Go runtime"
    }
  ],
  "chunks": [
    {
      "chunk_id": "42",
      "document_id": "go-concurrency-guide",
      "content": "Go achieves concurrency through goroutines — lightweight threads managed by the Go runtime...",
      "score": 0.9741
    }
  ]
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | LLM-generated answer grounded in the retrieved context |
| `grounded` | `bool` | `true` if at least one citation was validated against the provided chunks |
| `citations` | `array` | Each entry maps a claim to a `chunk_id`, `document_id`, and verbatim `excerpt` |
| `chunks` | `array` | Raw retrieval results with cosine similarity `score` ∈ [0, 1] |

---

## Project Structure

```
cortex/
├── gateway/                       # Go API gateway
│   ├── main.go                    # HTTP server, Kafka producer, gRPC clients
│   ├── gen/cortex/v1/             # Generated Go protobuf + gRPC stubs
│   ├── go.mod
│   └── go.sum
│
├── services/
│   ├── ingestion/                 # Python — Kafka consumer → chunk → embed → pgvector
│   │   ├── main.py                # Consumer loop & pipeline orchestration
│   │   ├── config.py              # Env-var configuration (frozen dataclass)
│   │   ├── db.py                  # Schema init, document & chunk persistence
│   │   ├── processor.py           # LangChain RecursiveCharacterTextSplitter
│   │   ├── embedder.py            # SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
│   │   ├── gen/cortex/v1/         # Generated Python protobuf stubs
│   │   └── pyproject.toml
│   │
│   ├── retriever/                 # Python — gRPC server, pgvector cosine search
│   │   ├── main.py                # gRPC server entrypoint (port 50051)
│   │   ├── config.py
│   │   ├── db.py                  # search_similar_chunks (<=> cosine operator)
│   │   ├── embedder.py            # Query embedding
│   │   ├── server.py              # RetrieverServiceServicer implementation
│   │   ├── gen/cortex/v1/         # Generated Python protobuf stubs
│   │   └── pyproject.toml
│   │
│   └── generator/                 # Python — gRPC server, Gemini, citation validation
│       ├── main.py                # gRPC server entrypoint (port 50052)
│       ├── config.py
│       ├── llm.py                 # RAG prompt, structured output, hallucination guard
│       ├── server.py              # GeneratorServiceServicer implementation
│       ├── gen/cortex/v1/         # Generated Python protobuf stubs
│       └── pyproject.toml
│
├── proto/
│   └── cortex/v1/
│       └── cortex.proto           # Single source of truth for all inter-service contracts
│
├── infra/
│   ├── local/
│   │   └── docker-compose.yml     # Postgres+pgvector, Redpanda, Redis
│   └── terraform/                 # Cloud deployment (WIP)
│       ├── main.tf
│       └── variables.tf
│
├── buf.gen.yaml                   # Buf code generation config
├── pyproject.toml                 # Root uv workspace
└── Makefile
```

---

## Environment Variables

| Variable | Service | Required | Default | Description |
|----------|---------|:--------:|---------|-------------|
| `DB_PASSWORD` | ingestion, retriever | ✅ | — | PostgreSQL password |
| `DB_HOST` | ingestion, retriever | | `localhost` | PostgreSQL host |
| `DB_PORT` | ingestion, retriever | | `5432` | PostgreSQL port |
| `DB_USER` | ingestion, retriever | | `cortex` | PostgreSQL user |
| `DB_NAME` | ingestion, retriever | | `cortex` | PostgreSQL database |
| `GOOGLE_API_KEY` | generator | ✅ | — | Google AI Studio API key |
| `GENERATOR_MODEL` | generator | | `gemini-2.5-flash` | Gemini model override |
| `KAFKA_ADDR` | gateway, ingestion | | `localhost:9092` | Kafka bootstrap server |
| `KAFKA_TOPIC` | gateway, ingestion | | `document.uploaded` | Ingestion topic name |
| `RETRIEVER_ADDR` | gateway | | `localhost:50051` | Retriever gRPC address |
| `GENERATOR_ADDR` | gateway | | `localhost:50052` | Generator gRPC address |
| `GATEWAY_PORT` | gateway | | `8080` | HTTP listen port |
| `REDIS_ADDR` | gateway | | `localhost:6379` | Redis address |

---

## Regenerating Protobuf Stubs

The generated stubs in each `gen/` directory are committed to the repository. To regenerate them after modifying `proto/cortex/v1/cortex.proto`:

```bash
# Install buf: https://buf.build/docs/installation
buf generate
```

`buf.gen.yaml` emits Go stubs into `gateway/gen/` and Python stubs into each `services/*/gen/`.

---

## gRPC Contracts

All services are defined in `proto/cortex/v1/cortex.proto`.

### `RetrieverService.SearchDocuments`

Takes a natural-language `query` and `top_k`. Embeds the query with the same `all-MiniLM-L6-v2` model used at index time and performs an HNSW cosine similarity search. Returns a ranked list of `DocumentChunk` objects with chunk content, source document ID, and relevance score.

### `GeneratorService.GenerateAnswer`

Takes a `query` and the list of `DocumentChunk` objects from the retriever. Constructs a RAG prompt, calls Gemini via LangChain with `with_structured_output`, and validates every citation against the `chunk_id`s actually provided. Returns `answer`, `citations[]`, and `grounded` (true if ≥1 citation survived validation).

---

## Service Ports

| Service | Port | Protocol |
|---------|------|----------|
| API Gateway | 8080 | HTTP |
| Retriever | 50051 | gRPC |
| Generator | 50052 | gRPC |
| PostgreSQL | 5432 | TCP |
| Redis | 6379 | TCP |
| Kafka / Redpanda | 9092 | TCP |

---

## Author

Built by **Ronnit Chopra** — Computer Science, Michigan State University, Class of 2026.

[![GitHub](https://img.shields.io/badge/GitHub-KlutzyFella-181717?style=flat-square&logo=github)](https://github.com/KlutzyFella)

---

## License

MIT
