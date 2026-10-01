package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	cortexv1 "github.com/KlutzyFella/cortex/gateway/gen/cortex/v1"
)

// stubRetriever is a fake cortexv1.RetrieverServiceClient that returns a canned
// response or error and records the request it was handed.
type stubRetriever struct {
	resp *cortexv1.SearchDocumentsResponse
	err  error

	got *cortexv1.SearchDocumentsRequest
}

func (s *stubRetriever) SearchDocuments(
	_ context.Context, in *cortexv1.SearchDocumentsRequest, _ ...grpc.CallOption,
) (*cortexv1.SearchDocumentsResponse, error) {
	s.got = in
	return s.resp, s.err
}

// stubGenerator is a fake cortexv1.GeneratorServiceClient.
type stubGenerator struct {
	resp *cortexv1.GenerateAnswerResponse
	err  error

	got *cortexv1.GenerateAnswerRequest
}

func (s *stubGenerator) GenerateAnswer(
	_ context.Context, in *cortexv1.GenerateAnswerRequest, _ ...grpc.CallOption,
) (*cortexv1.GenerateAnswerResponse, error) {
	s.got = in
	return s.resp, s.err
}

// callQuery drives queryHandler with the given JSON body and returns the recorder.
func callQuery(t *testing.T, s *server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.queryHandler(rec, req)
	return rec
}

func chunk(id string) *cortexv1.DocumentChunk {
	return &cortexv1.DocumentChunk{
		ChunkId:    id,
		DocumentId: "doc-1",
		Content:    "content of " + id,
		Score:      0.87,
	}
}

// TestQueryEmptyRetrievalIsNotAnUpstreamFailure is the regression test for the
// bug where an empty index surfaced as "502 generator unavailable": the
// generator rejects empty context, and the gateway blamed the generator for it.
// An empty result set is a client-visible condition, not a service outage.
func TestQueryEmptyRetrievalIsNotAnUpstreamFailure(t *testing.T) {
	s := &server{
		retriever: &stubRetriever{resp: &cortexv1.SearchDocumentsResponse{}},
		generator: &stubGenerator{}, // would be called, and would fail, if reached
	}

	rec := callQuery(t, s, `{"query":"anything"}`)

	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want %d (empty result must not be reported as an outage)",
			rec.Code, http.StatusNotFound)
	}
	if s.generator.(*stubGenerator).got != nil {
		t.Error("generator was called with zero chunks; it should be skipped entirely")
	}
	if body := rec.Body.String(); !strings.Contains(body, "ingest") {
		t.Errorf("body = %q, want a message telling the caller to ingest documents", body)
	}
}

func TestQueryGeneratorInvalidArgumentBecomes400(t *testing.T) {
	s := &server{
		retriever: &stubRetriever{resp: &cortexv1.SearchDocumentsResponse{
			Chunks: []*cortexv1.DocumentChunk{chunk("c1")},
		}},
		generator: &stubGenerator{err: status.Error(codes.InvalidArgument, "model not supported")},
	}

	rec := callQuery(t, s, `{"query":"hello"}`)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
	if body := rec.Body.String(); !strings.Contains(body, "model not supported") {
		t.Errorf("body = %q, want the upstream detail message forwarded", body)
	}
	if strings.Contains(rec.Body.String(), "unavailable") {
		t.Error("body blamed an unavailable upstream for a client error")
	}
}

func TestQueryUpstreamUnavailabilityBecomes503(t *testing.T) {
	s := &server{
		retriever: &stubRetriever{err: status.Error(codes.Unavailable, "connection refused")},
		generator: &stubGenerator{},
	}

	rec := callQuery(t, s, `{"query":"hello"}`)

	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
	}
}

func TestQueryDeadlineExceededBecomes504(t *testing.T) {
	s := &server{
		retriever: &stubRetriever{err: status.Error(codes.DeadlineExceeded, "context deadline exceeded")},
		generator: &stubGenerator{},
	}

	rec := callQuery(t, s, `{"query":"hello"}`)

	if rec.Code != http.StatusGatewayTimeout {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusGatewayTimeout)
	}
}

// TestQueryNonGRPCErrorFallsBackTo502 covers a transport-level failure, which
// carries no gRPC status to map.
func TestQueryNonGRPCErrorFallsBackTo502(t *testing.T) {
	s := &server{
		retriever: &stubRetriever{err: errors.New("dial tcp: connection reset")},
		generator: &stubGenerator{},
	}

	rec := callQuery(t, s, `{"query":"hello"}`)

	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusBadGateway)
	}
}

func TestQueryHappyPath(t *testing.T) {
	s := &server{
		retriever: &stubRetriever{resp: &cortexv1.SearchDocumentsResponse{
			Chunks: []*cortexv1.DocumentChunk{chunk("c1"), chunk("c2")},
		}},
		generator: &stubGenerator{resp: &cortexv1.GenerateAnswerResponse{
			Answer:   "Paris is the capital.",
			Grounded: true,
			Citations: []*cortexv1.Citation{
				{ChunkId: "c1", DocumentId: "doc-1", Excerpt: "Paris is the capital of France"},
			},
		}},
	}

	rec := callQuery(t, s, `{"query":"capital of France","top_k":2}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", rec.Code, rec.Body)
	}

	var body struct {
		Answer    string `json:"answer"`
		Grounded  bool   `json:"grounded"`
		Citations []struct {
			ChunkID string `json:"chunk_id"`
		} `json:"citations"`
		Chunks []struct {
			ChunkID string  `json:"chunk_id"`
			Score   float32 `json:"score"`
		} `json:"chunks"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}

	if body.Answer != "Paris is the capital." {
		t.Errorf("answer = %q", body.Answer)
	}
	if !body.Grounded {
		t.Error("grounded = false, want true")
	}
	if len(body.Citations) != 1 || body.Citations[0].ChunkID != "c1" {
		t.Errorf("citations = %+v, want one citation for c1", body.Citations)
	}
	// Empty slices must marshal as [] rather than null so clients can iterate safely.
	if len(body.Chunks) != 2 {
		t.Fatalf("chunks = %d, want 2", len(body.Chunks))
	}
	if !strings.Contains(rec.Body.String(), `"chunks":[{`) {
		t.Error("chunks did not marshal as an array; clients would have to handle null")
	}
}

func TestQueryRejectsMissingQuery(t *testing.T) {
	s := &server{retriever: &stubRetriever{}, generator: &stubGenerator{}}

	rec := callQuery(t, s, `{"query":""}`)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

func TestQueryRejectsMalformedJSON(t *testing.T) {
	s := &server{retriever: &stubRetriever{}, generator: &stubGenerator{}}

	rec := callQuery(t, s, `{"query":`)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}

// TestQueryDefaultsTopK confirms the default is applied before the RPC, since
// the proto treats an unset int32 as 0.
func TestQueryDefaultsTopK(t *testing.T) {
	r := &stubRetriever{resp: &cortexv1.SearchDocumentsResponse{
		Chunks: []*cortexv1.DocumentChunk{chunk("c1")},
	}}
	s := &server{retriever: r, generator: &stubGenerator{resp: &cortexv1.GenerateAnswerResponse{}}}

	callQuery(t, s, `{"query":"hello"}`)

	if r.got.GetTopK() != 5 {
		t.Errorf("top_k = %d, want the default 5", r.got.GetTopK())
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		name string
		in   string
		n    int
		want string
	}{
		{"shorter than limit is untouched", "hello", 10, "hello"},
		{"exactly at limit is untouched", "hello", 5, "hello"},
		{"longer than limit is cut", "hello world", 5, "hello..."},
		// A byte-based cut would slice the two-byte 'é' in half and emit
		// invalid UTF-8 into the log line.
		{"does not split a multi-byte rune", "héllo", 3, "hél..."},
		{"empty string", "", 5, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := truncate(tt.in, tt.n); got != tt.want {
				t.Errorf("truncate(%q, %d) = %q, want %q", tt.in, tt.n, got, tt.want)
			}
		})
	}
}
