"""Unit tests for the provider factory in llm.py.

Constructing chat-model objects performs no I/O (the HTTP call happens on
invoke), so these run with no network access and no API keys. Anything that
invokes a chain is out of scope here — see the live smoke procedure instead.
"""

import pytest
from llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    LLMResponse,
    build_chain,
    build_chat_model,
)


def test_gemini_builds_google_chat_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = build_chat_model("gemini", "g-key", "gemini-2.5-flash")

    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-2.5-flash"


def test_openrouter_builds_openai_compatible_model_pointing_at_openrouter():
    from langchain_openai import ChatOpenAI

    llm = build_chat_model(
        "openrouter", "or-key", "nvidia/nemotron-3-super-120b-a12b:free"
    )

    assert isinstance(llm, ChatOpenAI)
    assert "openrouter.ai" in str(llm.openai_api_base)
    assert llm.model_name == "nvidia/nemotron-3-super-120b-a12b:free"


def test_openrouter_base_url_is_overridable(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://mirror.local/v1")

    llm = build_chat_model("openrouter", "or-key", "some-model")

    assert isinstance(llm, ChatOpenAI)
    assert "mirror.local" in str(llm.openai_api_base)


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="provider"):
        build_chat_model("anthropic", "key", "model")


def test_build_chain_constructs_without_network():
    # Construction binds prompt + structured output + retry. No invoke, so no
    # HTTP. If this ever starts doing I/O at build time, this test becomes
    # the canary (it would hang or fail offline).
    chain = build_chain("openrouter", "or-key", DEFAULT_OPENROUTER_MODEL)

    assert chain is not None


def test_build_chain_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider"):
        build_chain("anthropic", "key", "model")


def test_canonical_defaults_are_the_reviewed_models():
    # The reviewed decision: structured-output-capable Nemotron Super, not the
    # larger variants that lack response_format support.
    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"
    assert DEFAULT_OPENROUTER_MODEL == "nvidia/nemotron-3-super-120b-a12b:free"


def test_structured_output_schema_is_unchanged():
    # The provider swap must not alter the contract: the chain still parses
    # into LLMResponse with answer + citations.
    assert set(LLMResponse.model_fields) == {"answer", "citations"}
