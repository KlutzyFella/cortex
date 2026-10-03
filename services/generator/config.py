"""Centralised configuration for the generator service."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorConfig:
    provider: str
    api_key: str
    default_model: str
    grpc_port: int

    @classmethod
    def from_env(cls) -> "GeneratorConfig":
        provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        if provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise KeyError(
                    "GOOGLE_API_KEY is not set. "
                    "Export it before starting the generator service."
                )
            default_model = os.environ.get("GENERATOR_MODEL", "gemini-2.5-flash")
        elif provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise KeyError(
                    "OPENROUTER_API_KEY is not set (LLM_PROVIDER=openrouter). "
                    "Export it before starting the generator service."
                )
            default_model = os.environ.get(
                "GENERATOR_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
            )
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER {provider!r}: expected 'gemini' or 'openrouter'."
            )
        return cls(
            provider=provider,
            api_key=api_key,
            default_model=default_model,
            grpc_port=int(os.environ.get("GRPC_PORT", "50052")),
        )
