#!/usr/bin/env bash
#
# Start the whole stack in the correct order and wait until each piece is
# actually serving before starting the piece that depends on it.
#
#   make run                 # everything
#   SKIP_GENERATOR=1 make run   # omit the generator (no GOOGLE_API_KEY needed)
#
# Logs go to logs/<service>.log; Ctrl-C stops everything.
#
# The ordering matters and is the reason this exists rather than four terminals:
#   1. Postgres and Kafka must be accepting connections
#   2. the Kafka topic must exist before the consumer subscribes, otherwise
#      confluent-kafka logs UNKNOWN_TOPIC_OR_PART on the first poll
#   3. retriever and generator must be listening before the gateway is useful
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
COMPOSE_FILE="infra/local/docker-compose.yml"
TOPIC="${KAFKA_TOPIC:-document.uploaded}"

PIDS=()

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'
  YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; GREEN=""; RED=""; YELLOW=""; RESET=""
fi

step() { printf '%s==>%s %s%s%s\n' "$GREEN" "$RESET" "$BOLD" "$1" "$RESET"; }
warn() { printf '%s warn:%s %s\n' "$YELLOW" "$RESET" "$1"; }
die()  { printf '%serror:%s %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

preflight() {
  step "Checking prerequisites"

  command -v docker >/dev/null || die "docker not found. Install Docker and re-run."
  docker info >/dev/null 2>&1 || die "cannot reach the Docker daemon. Is Docker running?"
  command -v uv >/dev/null || die "uv not found. See https://docs.astral.sh/uv/"

  # DB_PASSWORD is read with os.environ[...] by two services, so a missing value
  # is a crash at startup rather than a clear message.
  if [ -z "${DB_PASSWORD:-}" ]; then
    die "DB_PASSWORD is not set. The quickstart uses: export DB_PASSWORD=cortex_dev"
  fi

  if [ "${SKIP_GENERATOR:-0}" = "1" ]; then
    warn "SKIP_GENERATOR=1 — starting without the generator; queries will return 503"
  elif [ -z "${GOOGLE_API_KEY:-}" ]; then
    die "GOOGLE_API_KEY is not set. Either export it, or use SKIP_GENERATOR=1 to
       start without the generator (ingest and retrieval still work)."
  fi

  echo "  docker, uv and DB_PASSWORD present"
}

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

wait_for() {
  # wait_for <label> <timeout_seconds> <command...>
  local label="$1" timeout="$2"; shift 2
  local waited=0
  while ! "$@" >/dev/null 2>&1; do
    if [ "$waited" -ge "$timeout" ]; then
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
    printf '%s   … waiting for %s (%ss)%s\n' "$DIM" "$label" "$waited" "$RESET"
  done
  return 0
}

start_infra() {
  step "Starting Postgres and Kafka"
  docker compose -f "$COMPOSE_FILE" up -d || die "docker compose up failed"

  wait_for "Postgres" 90 \
    docker exec cortex-postgres pg_isready -U cortex \
    || die "Postgres did not become ready in 90s. Check: docker logs cortex-postgres"

  # The broker answers API versions well before it finishes its own startup,
  # so this is the earliest point a client can safely talk to it.
  wait_for "Kafka" 120 \
    docker exec cortex-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 \
    || die "Kafka did not become ready in 120s. Check: docker logs cortex-kafka"

  echo "  Postgres ready on :5432, Kafka ready on :9092"
}

create_topic() {
  # Auto-creation works, but only after a client subscribes or produces, and it
  # makes the ingestion worker log UNKNOWN_TOPIC_OR_PART on its first poll.
  # Creating the topic up front means the consumer subscribes to something real.
  step "Ensuring Kafka topic '$TOPIC' exists"
  docker exec cortex-kafka kafka-topics \
    --bootstrap-server localhost:9092 \
    --create --if-not-exists \
    --topic "$TOPIC" --partitions 1 --replication-factor 1 >/dev/null 2>&1 \
    && echo "  topic ready" \
    || warn "could not create topic; relying on broker auto-creation"
}

# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------

# start_service <name> <command...>
start_service() {
  local name="$1"; shift
  : >"$LOG_DIR/$name.log"
  ( "$@" >>"$LOG_DIR/$name.log" 2>&1 ) &
  local pid=$!
  PIDS+=("$pid")
  echo "$pid" >"$LOG_DIR/$name.pid"
  echo "  $name started (pid $pid) -> logs/$name.log"
}

wait_for_port() {
  # wait_for_port <port> <timeout_seconds>
  local port="$1" timeout="$2" waited=0
  while ! (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    if [ "$waited" -ge "$timeout" ]; then
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 0
}

start_services() {
  mkdir -p "$LOG_DIR"
  step "Starting services"

  # `uv run` resolves the workspace virtualenv, so each service gets its own
  # sys.path and the shared flat module names do not collide.
  start_service ingestion uv run --frozen python services/ingestion/main.py

  # The retriever loads the embedding model at startup, which is a download on a
  # cold HuggingFace cache -- hence the 180s wait below.
  start_service retriever uv run --frozen python services/retriever/main.py

  if [ "${SKIP_GENERATOR:-0}" != "1" ]; then
    start_service generator uv run --frozen python services/generator/main.py
  fi

  step "Waiting for services to listen"
  wait_for_port 50051 180 \
    || die "retriever did not listen on 50051. Check logs/retriever.log"
  echo "  retriever listening on :50051"

  if [ "${SKIP_GENERATOR:-0}" != "1" ]; then
    wait_for_port 50052 120 \
      || die "generator did not listen on 50052. Check logs/generator.log"
    echo "  generator listening on :50052"
  fi

  # `go run .` compiles to a temporary binary and runs it as a *child* process,
  # so the pid we track would be the go tool rather than the server -- killing it
  # on Ctrl-C would leave the gateway still holding :8080. Building first means
  # the pid we track is the server itself.
  step "Building the gateway"
  ( cd gateway && go build -o "$LOG_DIR/gateway-bin" . ) \
    || die "gateway build failed"
  echo "  gateway built"

  start_service gateway "$LOG_DIR/gateway-bin"

  wait_for_port 8080 120 \
    || die "gateway did not listen on 8080. Check logs/gateway.log"
  echo "  gateway listening on :8080"
}

# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

shutdown() {
  step "Stopping services"
  # Reverse order so the gateway stops before the services it calls.
  for (( i=${#PIDS[@]}-1 ; i>=0 ; i-- )); do
    local pid="${PIDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  # Give them a moment to close connections, then insist.
  sleep 3
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done

  rm -f "$LOG_DIR"/*.pid
  echo "  stopped. Postgres and Kafka are still up; stop them with: make down"
}

trap 'shutdown; exit 0' INT TERM

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

preflight
start_infra
create_topic
start_services

step "Stack is up"
cat <<EOF
  Gateway    http://localhost:8080
  Ingest     curl -X POST localhost:8080/ingest \\
               -d '{"doc_id":"doc-1","content":"Cortex stores vectors in pgvector."}'
  Query      curl -X POST localhost:8080/api/v1/query \\
               -d '{"query":"How does Cortex search?","top_k":3}'

  Logs       tail -f logs/*.log
  Stop       Ctrl-C
EOF

if [ "${SKIP_GENERATOR:-0}" = "1" ]; then
  printf '%s  Note: the generator is not running, so queries will return 503.%s\n' "$YELLOW" "$RESET"
fi

# Exit as soon as any service dies, so a crashed process is not mistaken for a
# healthy stack.
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      warn "a service exited unexpectedly; check the logs in $LOG_DIR"
      shutdown
      exit 1
    fi
  done
  sleep 2
done