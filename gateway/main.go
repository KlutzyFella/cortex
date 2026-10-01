package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/confluentinc/confluent-kafka-go/v2/kafka"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	cortexv1 "github.com/KlutzyFella/cortex/gateway/gen/cortex/v1"
)

// server holds all long-lived dependencies so they can be cleanly closed on shutdown.
type server struct {
	producer      *kafka.Producer
	retriever     cortexv1.RetrieverServiceClient
	generator     cortexv1.GeneratorServiceClient
	retrieverConn *grpc.ClientConn
	generatorConn *grpc.ClientConn
	kafkaTopic    string
}

func main() {
	port := getenv("GATEWAY_PORT", "8080")
	kafkaAddr := getenv("KAFKA_ADDR", "localhost:9092")
	kafkaTopic := getenv("KAFKA_TOPIC", "document.uploaded")
	retrieverAddr := getenv("RETRIEVER_ADDR", "localhost:50051")
	generatorAddr := getenv("GENERATOR_ADDR", "localhost:50052")

	// --- Kafka producer ---
	producer, err := kafka.NewProducer(&kafka.ConfigMap{
		"bootstrap.servers": kafkaAddr,
	})
	if err != nil {
		log.Fatalf("failed to create kafka producer: %v", err)
	}
	log.Printf("kafka producer connected to %s", kafkaAddr)

	// Drain producer delivery reports in the background so internal queues don't block.
	go func() {
		for e := range producer.Events() {
			if m, ok := e.(*kafka.Message); ok && m.TopicPartition.Error != nil {
				log.Printf("kafka delivery error: %v", m.TopicPartition.Error)
			}
		}
	}()

	// --- gRPC → Retriever ---
	retrieverConn, err := grpc.NewClient(
		retrieverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("failed to dial retriever at %s: %v", retrieverAddr, err)
	}
	log.Printf("gRPC client connected to retriever at %s", retrieverAddr)

	// --- gRPC → Generator ---
	generatorConn, err := grpc.NewClient(
		generatorAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatalf("failed to dial generator at %s: %v", generatorAddr, err)
	}
	log.Printf("gRPC client connected to generator at %s", generatorAddr)

	svc := &server{
		producer:      producer,
		retriever:     cortexv1.NewRetrieverServiceClient(retrieverConn),
		generator:     cortexv1.NewGeneratorServiceClient(generatorConn),
		retrieverConn: retrieverConn,
		generatorConn: generatorConn,
		kafkaTopic:    kafkaTopic,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", healthzHandler)
	mux.HandleFunc("POST /ingest", svc.ingestHandler)
	mux.HandleFunc("POST /api/v1/query", svc.queryHandler)

	httpSrv := &http.Server{
		Addr:    ":" + port,
		Handler: mux,
	}

	// Graceful shutdown: wait for SIGINT / SIGTERM.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Printf("gateway listening on :%s", port)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-quit
	log.Println("shutting down gateway…")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Printf("HTTP shutdown error: %v", err)
	}

	producer.Flush(5000) // wait up to 5 s for in-flight messages
	producer.Close()
	retrieverConn.Close()
	generatorConn.Close()

	log.Println("gateway stopped")
}

// healthzHandler returns a simple liveness probe response.
func healthzHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// ingestHandler publishes a document to the Kafka ingestion topic.
//
//	POST /ingest
//	Body: {"doc_id": "...", "content": "..."}
func (s *server) ingestHandler(w http.ResponseWriter, r *http.Request) {
	var payload struct {
		DocID   string `json:"doc_id"`
		Content string `json:"content"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	if payload.DocID == "" || payload.Content == "" {
		http.Error(w, "doc_id and content are required", http.StatusBadRequest)
		return
	}

	msg, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, "failed to encode message", http.StatusInternalServerError)
		return
	}

	err = s.producer.Produce(&kafka.Message{
		TopicPartition: kafka.TopicPartition{
			Topic:     &s.kafkaTopic,
			Partition: kafka.PartitionAny,
		},
		Value: msg,
	}, nil)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to enqueue message: %v", err), http.StatusInternalServerError)
		return
	}

	log.Printf("enqueued doc '%s' to topic '%s'", payload.DocID, s.kafkaTopic)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{"status": "accepted", "doc_id": payload.DocID})
}

// queryHandler runs the full RAG pipeline: it retrieves the top-k relevant
// chunks with RetrieverService.SearchDocuments, then generates a grounded
// answer with GeneratorService.GenerateAnswer. Both calls share one deadline.
//
//	POST /api/v1/query
//	Body: {"query": "...", "top_k": 3}
func (s *server) queryHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query string `json:"query"`
		TopK  int32  `json:"top_k"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}

	// Use a single deadline shared across both downstream RPCs.
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	// --- Step 1: Retrieve relevant chunks ---
	searchResp, err := s.retriever.SearchDocuments(ctx, &cortexv1.SearchDocumentsRequest{
		Query: req.Query,
		TopK:  req.TopK,
	})
	if err != nil {
		log.Printf("retriever error: %v", err)
		writeGRPCError(w, "retriever", err)
		return
	}

	retrievedChunks := searchResp.GetChunks()
	if len(retrievedChunks) == 0 {
		// Nothing in the index matched. This is not an upstream failure, so it
		// must not be reported as one: the caller needs to know the request was
		// valid but there was no corpus to answer from.
		log.Printf("retriever returned no chunks for query=%q", truncate(req.Query, 100))
		http.Error(w,
			"no indexed content matched this query; ingest documents before querying",
			http.StatusNotFound)
		return
	}

	// --- Step 2: Generate grounded answer ---
	genResp, err := s.generator.GenerateAnswer(ctx, &cortexv1.GenerateAnswerRequest{
		Query:         req.Query,
		ContextChunks: retrievedChunks,
	})
	if err != nil {
		log.Printf("generator error: %v", err)
		writeGRPCError(w, "generator", err)
		return
	}

	// --- Build response ---
	type chunkJSON struct {
		ChunkID    string  `json:"chunk_id"`
		DocumentID string  `json:"document_id"`
		Content    string  `json:"content"`
		Score      float32 `json:"score"`
	}
	type citationJSON struct {
		ChunkID    string `json:"chunk_id"`
		DocumentID string `json:"document_id"`
		Excerpt    string `json:"excerpt"`
	}

	chunks := make([]chunkJSON, 0, len(retrievedChunks))
	for _, c := range retrievedChunks {
		chunks = append(chunks, chunkJSON{
			ChunkID:    c.GetChunkId(),
			DocumentID: c.GetDocumentId(),
			Content:    c.GetContent(),
			Score:      c.GetScore(),
		})
	}

	citations := make([]citationJSON, 0, len(genResp.GetCitations()))
	for _, c := range genResp.GetCitations() {
		citations = append(citations, citationJSON{
			ChunkID:    c.GetChunkId(),
			DocumentID: c.GetDocumentId(),
			Excerpt:    c.GetExcerpt(),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"answer":    genResp.GetAnswer(),
		"grounded":  genResp.GetGrounded(),
		"citations": citations,
		"chunks":    chunks,
	})
}

// writeGRPCError translates a downstream gRPC status into the closest HTTP
// status and forwards the upstream's own detail message.
//
// Collapsing every failure to 502 mislabels client errors as upstream outages:
// an INVALID_ARGUMENT from the generator means this gateway sent a request the
// generator rejected, which is a 400, not a bad gateway.
func writeGRPCError(w http.ResponseWriter, upstream string, err error) {
	st, ok := status.FromError(err)
	if !ok {
		// Not a gRPC status (e.g. a transport-level failure), so there is no
		// code to map and nothing to report but the fact of the outage.
		http.Error(w, upstream+" unreachable", http.StatusBadGateway)
		return
	}

	var code int
	switch st.Code() {
	case codes.InvalidArgument:
		code = http.StatusBadRequest
	case codes.NotFound:
		code = http.StatusNotFound
	case codes.AlreadyExists:
		code = http.StatusConflict
	case codes.ResourceExhausted:
		code = http.StatusTooManyRequests
	case codes.DeadlineExceeded:
		code = http.StatusGatewayTimeout
	case codes.Unavailable:
		code = http.StatusServiceUnavailable
	default:
		code = http.StatusBadGateway
	}

	http.Error(w, fmt.Sprintf("%s: %s", upstream, st.Message()), code)
}

// truncate shortens s to at most n runes so untrusted input can be logged
// without unbounded growth. Rune-based so a multi-byte character is never split.
func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "..."
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
