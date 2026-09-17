"""Turns .env settings into the model objects the agents use.

There are two, because no free provider we tested does everything well:

- The main model runs the tutors, tools and handoffs. Measured: Groq's
  gpt-oss-120b calls tools reliably and routes handoffs correctly, but cannot
  emit multi-field JSON (400 json_validate_failed, even non-strict).
- The structured model runs the quiz agents, which need `output_type` but use
  no tools and no handoffs. OpenRouter's Nemotron is the mirror image: solid
  structured output, unreliable tool calling.

Set the three STUDY_HELPER_STRUCTURED_* variables to split them. Leave them
unset and the structured model falls back to the main one, so a single
provider that does both needs no code change - just delete the variables.
"""

import os
from functools import lru_cache

from agents import AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from dotenv import load_dotenv

load_dotenv()

# Tracing ships run data to OpenAI's platform and needs a real OpenAI key.
# We are on a third-party endpoint, so turn it off.
set_tracing_disabled(True)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def _build(api_key: str, base_url: str, model: str) -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(
        model=model,
        openai_client=AsyncOpenAI(api_key=api_key, base_url=base_url),
    )


@lru_cache(maxsize=1)
def build_model() -> OpenAIChatCompletionsModel:
    """The main model: tutors, tools, handoffs. Cached so agents share a client."""
    api_key = os.environ.get("STUDY_HELPER_API_KEY")
    if not api_key:
        raise SystemExit(
            "STUDY_HELPER_API_KEY is not set.\n"
            "Copy .env.example to .env and fill in your key."
        )
    return _build(
        api_key,
        os.environ.get("STUDY_HELPER_BASE_URL", DEFAULT_BASE_URL),
        os.environ.get("STUDY_HELPER_MODEL", DEFAULT_MODEL),
    )


@lru_cache(maxsize=1)
def build_structured_model() -> OpenAIChatCompletionsModel:
    """The model for agents with an `output_type`.

    Falls back to the main model when STUDY_HELPER_STRUCTURED_* is unset, so
    the split is opt-in and disappears cleanly when it is no longer needed.
    """
    model = os.environ.get("STUDY_HELPER_STRUCTURED_MODEL")
    if not model:
        return build_model()

    return _build(
        # Falling back per-variable means you can point at a different model on
        # the same provider by setting only STUDY_HELPER_STRUCTURED_MODEL.
        os.environ.get("STUDY_HELPER_STRUCTURED_API_KEY")
        or os.environ.get("STUDY_HELPER_API_KEY", ""),
        os.environ.get("STUDY_HELPER_STRUCTURED_BASE_URL")
        or os.environ.get("STUDY_HELPER_BASE_URL", DEFAULT_BASE_URL),
        model,
    )
