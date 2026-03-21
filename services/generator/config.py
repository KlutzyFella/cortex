"""Centralised configuration for the generator service."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorConfig:
    google_api_key: str
    default_model: str
    grpc_port: int

    @classmethod
    def from_env(cls) -> "GeneratorConfig":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise KeyError(
                "GOOGLE_API_KEY is not set. "
                "Export it before starting the generator service."
            )
        return cls(
            google_api_key=api_key,
            default_model=os.environ.get(
                "GENERATOR_MODEL", "gemini-2.5-flash"
            ),
            grpc_port=int(os.environ.get("GRPC_PORT", "50052")),
        )
