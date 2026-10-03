"""Unit tests for provider-aware GeneratorConfig.

No network access: from_env only reads os.environ. Each test starts from a
clean slate with both keys removed so the developer's real environment can
never leak into an assertion.
"""

import os

import pytest
from config import GeneratorConfig


@pytest.fixture()
def clean_env(monkeypatch):
    for var in ("LLM_PROVIDER", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
                "GENERATOR_MODEL", "GRPC_PORT"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_default_provider_is_gemini(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    cfg = GeneratorConfig.from_env()

    assert cfg.provider == "gemini"
    assert cfg.api_key == "g-key"
    assert cfg.default_model == "gemini-2.5-flash"


def test_gemini_missing_key_raises_key_error(clean_env):
    with pytest.raises(KeyError, match="GOOGLE_API_KEY"):
        GeneratorConfig.from_env()


def test_gemini_model_override_respected(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    clean_env.setenv("GENERATOR_MODEL", "gemini-2.5-pro")

    assert GeneratorConfig.from_env().default_model == "gemini-2.5-pro"


def test_openrouter_defaults_to_nemotron_super(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openrouter")
    clean_env.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = GeneratorConfig.from_env()

    assert cfg.provider == "openrouter"
    assert cfg.api_key == "or-key"
    assert cfg.default_model == "nvidia/nemotron-3-super-120b-a12b:free"


def test_openrouter_missing_key_names_openrouter_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openrouter")

    with pytest.raises(KeyError, match="OPENROUTER_API_KEY"):
        GeneratorConfig.from_env()


def test_openrouter_ignores_google_key(clean_env):
    # A stale GOOGLE_API_KEY must not satisfy the openrouter requirement:
    # the wrong credential is the same as no credential.
    clean_env.setenv("LLM_PROVIDER", "openrouter")
    clean_env.setenv("GOOGLE_API_KEY", "g-key")

    with pytest.raises(KeyError, match="OPENROUTER_API_KEY"):
        GeneratorConfig.from_env()


def test_gemini_ignores_openrouter_key(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "or-key")

    with pytest.raises(KeyError, match="GOOGLE_API_KEY"):
        GeneratorConfig.from_env()


def test_unknown_provider_raises_value_error(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        GeneratorConfig.from_env()


def test_provider_is_case_insensitive(clean_env):
    clean_env.setenv("LLM_PROVIDER", "OpenRouter")
    clean_env.setenv("OPENROUTER_API_KEY", "or-key")

    assert GeneratorConfig.from_env().provider == "openrouter"


def test_grpc_port_default_and_override(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    assert GeneratorConfig.from_env().grpc_port == 50052

    clean_env.setenv("GRPC_PORT", "50099")
    assert GeneratorConfig.from_env().grpc_port == 50099


def test_api_key_never_defaults(clean_env):
    # from_env must not invent a credential: empty string and absent are
    # both missing.
    clean_env.setenv("GOOGLE_API_KEY", "")

    with pytest.raises(KeyError):
        GeneratorConfig.from_env()

    assert os.environ.get("GOOGLE_API_KEY") == ""
