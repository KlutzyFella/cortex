"""Unit tests for the generator's citation validation.

These exercise the invariant the project is built around: an answer is only
reported as grounded when at least one citation survives validation against the
chunk_ids that were actually supplied as context.

The LangChain chain is replaced by DummyChain, which returns a canned
LLMResponse, so these tests need no network access and no GOOGLE_API_KEY.
"""

from llm import CitationModel, LLMResponse, _format_context, generate_response


class DummyChain:
    """Stands in for the LangChain LCEL chain returned by build_chain."""

    def __init__(self, result):
        self._result = result
        self.invoked_with = None

    def invoke(self, kwargs):
        self.invoked_with = kwargs
        return self._result


def _response(*citations):
    """Build the canned LLMResponse a DummyChain will return."""
    return LLMResponse(answer="answer text", citations=list(citations))


def _cite(chunk_id, excerpt="excerpt"):
    return CitationModel(chunk_id=chunk_id, excerpt=excerpt)


CHUNKS = [{"chunk_id": "c1", "content": "..."}, {"chunk_id": "c2", "content": "..."}]


def test_all_valid_citations_are_kept():
    chain = DummyChain(_response(_cite("c1", "A"), _cite("c2", "B")))

    res = generate_response(chain, "q", CHUNKS, {"c1", "c2"})

    assert res.grounded is True
    assert [c.chunk_id for c in res.citations] == ["c1", "c2"]
    assert res.answer == "answer text"


def test_hallucinated_citations_are_stripped():
    chain = DummyChain(_response(_cite("c1", "A"), _cite("not-a-real-chunk", "H")))

    res = generate_response(chain, "q", CHUNKS, {"c1", "c2"})

    # One valid citation survives, so the answer is still grounded.
    assert res.grounded is True
    assert [c.chunk_id for c in res.citations] == ["c1"]


def test_answer_text_survives_validation():
    """Stripping a citation must never discard the answer itself."""
    chain = DummyChain(_response(_cite("not-a-real-chunk", "H")))

    res = generate_response(chain, "q", CHUNKS, {"c1", "c2"})

    assert res.grounded is False
    assert res.citations == []
    assert res.answer == "answer text"


def test_all_citations_hallucinated_reports_ungrounded():
    chain = DummyChain(_response(_cite("bad1", "x"), _cite("bad2", "y")))

    res = generate_response(chain, "q", CHUNKS, {"c1", "c2"})

    assert res.grounded is False
    assert res.citations == []


def test_no_citations_reports_ungrounded():
    chain = DummyChain(_response())

    res = generate_response(chain, "q", CHUNKS, {"c1", "c2"})

    assert res.grounded is False
    assert res.citations == []


def test_empty_valid_set_strips_everything():
    chain = DummyChain(_response(_cite("c1", "A")))

    res = generate_response(chain, "q", CHUNKS, set())

    assert res.grounded is False
    assert res.citations == []


def test_chain_receives_formatted_context_and_question():
    chain = DummyChain(_response(_cite("c1", "A")))

    generate_response(chain, "what is the capital?", CHUNKS, {"c1", "c2"})

    assert chain.invoked_with["question"] == "what is the capital?"
    assert "[c1]" in chain.invoked_with["context"]
    assert "[c2]" in chain.invoked_with["context"]


class TestFormatContext:
    def test_uses_chunk_id_and_content(self):
        ctx = _format_context(
            [
                {"chunk_id": "c1", "content": "Paris is the capital of France."},
                {"chunk_id": "c2", "content": "Berlin is the capital of Germany."},
            ]
        )

        assert "[c1]\nParis is the capital of France." in ctx
        assert "[c2]\nBerlin is the capital of Germany." in ctx

    def test_falls_back_to_one_based_index_when_id_missing(self):
        ctx = _format_context([{"content": "first"}, {"content": "second"}])

        # enumerate(..., 1) means the fallback labels are 1-based.
        assert "[chunk-1]" in ctx
        assert "[chunk-2]" in ctx

    def test_empty_chunks_yields_empty_string(self):
        assert _format_context([]) == ""

    def test_missing_content_yields_empty_body(self):
        assert _format_context([{"chunk_id": "c1"}]) == "[c1]\n"