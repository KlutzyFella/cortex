"""LLM logic: RAG prompt construction, generation, and citation extraction."""

import logging
from typing import NamedTuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class CitationModel(BaseModel):
    chunk_id: str = Field(description="The chunk_id of the source chunk")
    excerpt: str = Field(
        description="A short verbatim quote from that chunk that supports the answer"
    )


class LLMResponse(BaseModel):
    answer: str = Field(
        description="The answer to the user's question, grounded in the provided context"
    )
    citations: list[CitationModel] = Field(
        default_factory=list,
        description="List of citations referencing the chunk_ids used to form the answer",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain enough information to "
    "answer the question, say so clearly. Do not hallucinate facts.\n\n"
    "For each claim you make, include a citation referencing the chunk_id of "
    "the source passage."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GeneratedAnswer(NamedTuple):
    answer: str
    citations: list[CitationModel]
    grounded: bool


def build_chain(api_key: str, model_name: str):
    """Return a LangChain LCEL chain with structured output."""
    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model_name,
        temperature=0,
        max_output_tokens=2048,
    )
    structured_llm = llm.with_structured_output(LLMResponse)
    return _PROMPT | structured_llm


def _format_context(chunks: list[dict]) -> str:
    """Format chunks into a numbered context block with chunk IDs."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("chunk_id", f"chunk-{i}")
        content = chunk.get("content", "")
        lines.append(f"[{chunk_id}]\n{content}")
    return "\n\n---\n\n".join(lines)


def generate_response(
    chain,
    question: str,
    chunks: list[dict],
    valid_chunk_ids: set[str],
) -> GeneratedAnswer:
    """Call the LLM chain and return a validated GeneratedAnswer.

    Args:
        chain:           LangChain LCEL chain (from build_chain).
        question:        The user's query.
        chunks:          List of dicts with keys 'chunk_id' and 'content'.
        valid_chunk_ids: Set of chunk_id strings that were actually provided,
                         used to strip hallucinated citations.

    Returns:
        GeneratedAnswer with answer text, validated citations, and grounded flag.
    """
    context = _format_context(chunks)
    logger.debug("Calling LLM with %d chunks, question=%r", len(chunks), question[:100])

    result: LLMResponse = chain.invoke({"context": context, "question": question})

    # Validate: strip any citation whose chunk_id was not in the provided context.
    validated = [c for c in result.citations if c.chunk_id in valid_chunk_ids]
    hallucinated_count = len(result.citations) - len(validated)
    if hallucinated_count:
        logger.warning(
            "Stripped %d hallucinated citation(s) not present in provided chunks",
            hallucinated_count,
        )

    grounded = len(validated) > 0
    return GeneratedAnswer(
        answer=result.answer,
        citations=validated,
        grounded=grounded,
    )
