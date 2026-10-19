import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_GEMINI_VERSION = re.compile(r"^gemini-(\d+)\.(\d+)")


def _accepts_minimal_thinking(model: str) -> bool:
    """Whether ``thinking_level="minimal"`` is accepted: numbered Flash models
    before 3.8. Pro, 3.8+ and version-less aliases (which move between
    generations) are treated as rejecting it."""
    model_lc = model.lower()
    match = _GEMINI_VERSION.match(model_lc)
    return bool(match) and "pro" not in model_lc and (
        (int(match.group(1)), int(match.group(2))) < (3, 8)
    )


import os
import logging

logger = logging.getLogger(__name__)


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output and OpenRouter fallback on 429."""

    def invoke(self, input, config=None, **kwargs):
        try:
            return normalize_content(super().invoke(input, config, **kwargs))
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()):
                clean_kwargs = {k: v for k, v in kwargs.items() if k not in ("response_mime_type", "thinking_level", "response_json_schema")}
                groq_key = os.getenv("GROQ_API_KEY", "").split(",")[0].strip()
                if groq_key:
                    try:
                        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
                        logger.warning(f"Gemini 429 quota reached. Gracefully delegating to Groq Cloud ({groq_model})...")
                        from .factory import create_llm_client
                        fb_client = create_llm_client("groq", groq_model, api_key=groq_key, llm_max_retries=0, timeout=30.0)
                        return fb_client.get_llm().invoke(input, config=config, **clean_kwargs)
                    except Exception as ge:
                        ge_str = str(ge)
                        if any(k in ge_str for k in ("429", "rate_limit", "RateLimitError", "413", "Payload Too Large")) and os.getenv("OPENROUTER_API_KEY"):
                            logger.warning("Groq rate limit/payload reached. Failing over to OpenRouter...")
                            from .factory import create_llm_client
                            fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning:free")
                            fb_client = create_llm_client("openrouter", fallback_model, api_key=os.getenv("OPENROUTER_API_KEY"), timeout=30.0)
                            return fb_client.get_llm().invoke(input, config=config, **clean_kwargs)
                        raise
                elif os.getenv("OPENROUTER_API_KEY"):
                    logger.warning("Gemini 429 quota reached. Gracefully delegating to OpenRouter fallback...")
                    from .factory import create_llm_client
                    fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning:free")
                    fb_client = create_llm_client("openrouter", fallback_model, api_key=os.getenv("OPENROUTER_API_KEY"), timeout=30.0)
                    return fb_client.get_llm().invoke(input, config=config, **clean_kwargs)
            raise


# The read timeout the OpenAI and Anthropic SDKs use by default.
REQUEST_TIMEOUT_SECONDS = 600


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "temperature", "max_output_tokens",
                    "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # Fast failover on rate limit rather than waiting minutes in exponential backoff
        llm_kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
        llm_kwargs.setdefault("max_retries", 1)

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Gemini 3.x takes the string ``thinking_level`` (the integer
        # ``thinking_budget`` was for the now-retired 2.5 line). Pro, Gemini
        # 3.8+ and the -latest aliases reject "minimal" with a 400; "low" is
        # accepted everywhere, so it is the fallback.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if thinking_level == "minimal" and not _accepts_minimal_thinking(self.model):
                thinking_level = "low"
            llm_kwargs["thinking_level"] = thinking_level

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
