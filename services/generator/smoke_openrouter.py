"""Manual smoke: one structured-output call via OpenRouter.

This is NOT a test and never runs in CI (it needs a key AND network).
Run it once when an OPENROUTER_API_KEY becomes available:

    OPENROUTER_API_KEY=... uv run --project services/generator \
        python services/generator/smoke_openrouter.py

Exit 0: structured output parsed into LLMResponse with a validated citation.
Exit 2: no key configured (nothing attempted, no network touched).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import DEFAULT_OPENROUTER_MODEL, build_chain, build_chat_model, generate_response


def _load_dotenv():
    """Fill missing vars from the repo-root .env (stdlib only, no dependency).

    `make run` sources .env for the stack, but this script is run directly, so
    it does its own fallback. Explicitly exported variables win; this only
    fills gaps. Format: KEY=value lines, `#` comments, optional quotes.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(os.path.dirname(here), "..", ".env")
    try:
        with open(env_path) as f:
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    _load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("smoke skipped: OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    model_name = os.environ.get("GENERATOR_MODEL", DEFAULT_OPENROUTER_MODEL)

    chunks = [{"chunk_id": "c1", "content": "Cortex stores vectors in pgvector."}]
    chain = build_chain("openrouter", api_key, model_name)
    result = generate_response(
        chain, "Where does Cortex store vectors?", chunks, {"c1"}
    )
    print(f"answer: {result.answer}")
    print(f"citations: {[c.chunk_id for c in result.citations]}")
    print(f"grounded: {result.grounded}")
    if not result.grounded:
        print("SMOKE FAIL: answer carries no validated citation", file=sys.stderr)
        return 1

    # The structured chain returns parsed LLMResponse, which drops response
    # metadata — so ask the bare model once what actually served the request.
    # If the alias moved, this is where it shows up instead of silently
    # invalidating future baseline comparisons.
    raw = build_chat_model("openrouter", api_key, model_name).invoke("ping")
    print(f"resolved_model: {raw.response_metadata.get('model_name', 'unknown')}")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
