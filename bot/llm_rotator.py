"""Multi-Provider and Multi-Key LLM Rotator with Automatic Failover.

Supports Round-Robin key rotation and transparent failover across multiple free AI platforms:
1. Google AI Studio (Gemini 3 Flash, Gemini Flash Latest)
2. OpenRouter (Nemotron, Llama 3.3, DeepSeek R1 free tiers)
3. Groq Cloud (Llama 3.3 70B, Llama 3.1 8B, Mixtral)
4. Mistral AI (Mistral Small, Codestral)
5. GitHub Models (GPT-4o-mini, Llama 3.3 70B)
6. Cerebras Cloud (Llama 3.1 70B, 8B)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ProviderCredential:
    provider_name: str
    api_key: str
    model: str
    base_url: str | None = None
    cooldown_until: float = 0.0
    total_calls: int = 0
    total_errors: int = 0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until


class RotatingChatProxy(BaseChatModel):
    """Proxy that routes invocations through the active rotating provider pool with instant failover."""

    rotator: Any = field(default=None)

    def __init__(self, rotator: LLMRotator, **kwargs):
        super().__init__(**kwargs)
        self.rotator = rotator

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self.rotator.invoke_with_failover(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "rotating-chat-proxy"


class LLMRotator:
    """Manages a pool of AI API credentials with round-robin rotation and cooldown protection."""

    def __init__(self):
        self.pool: list[ProviderCredential] = []
        self._index: int = 0
        self.load_credentials_from_env()

    def load_credentials_from_env(self) -> None:
        """Scan environment variables and populate provider credentials."""
        self.pool.clear()

        # 1. Google Gemini (Supports comma-separated keys)
        google_keys = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        for k in [x.strip() for x in google_keys.split(",") if x.strip()]:
            self.pool.append(
                ProviderCredential(
                    provider_name="Google Gemini",
                    api_key=k,
                    model=os.getenv("TRADINGAGENTS_QUICK_THINK_LLM", "gemini-3-flash-preview"),
                )
            )

        # 2. OpenRouter (Supports comma-separated keys)
        openrouter_keys = os.getenv("OPENROUTER_API_KEY", "")
        for k in [x.strip() for x in openrouter_keys.split(",") if x.strip()]:
            self.pool.append(
                ProviderCredential(
                    provider_name="OpenRouter",
                    api_key=k,
                    model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
                    base_url="https://openrouter.ai/api/v1",
                )
            )

        # 3. Groq Cloud
        groq_keys = os.getenv("GROQ_API_KEY", "")
        for k in [x.strip() for x in groq_keys.split(",") if x.strip()]:
            self.pool.append(
                ProviderCredential(
                    provider_name="Groq Cloud",
                    api_key=k,
                    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                    base_url="https://api.groq.com/openai/v1",
                )
            )

        # 4. Mistral AI
        mistral_keys = os.getenv("MISTRAL_API_KEY", "")
        for k in [x.strip() for x in mistral_keys.split(",") if x.strip()]:
            self.pool.append(
                ProviderCredential(
                    provider_name="Mistral AI",
                    api_key=k,
                    model="mistral-small-latest",
                    base_url="https://api.mistral.ai/v1",
                )
            )

        # 5. GitHub Models
        github_token = os.getenv("GITHUB_TOKEN", "") or os.getenv("GITHUB_MODELS_KEY", "")
        for k in [x.strip() for x in github_token.split(",") if x.strip()]:
            self.pool.append(
                ProviderCredential(
                    provider_name="GitHub Models",
                    api_key=k,
                    model="gpt-4o-mini",
                    base_url="https://models.inference.ai.azure.com",
                )
            )

        # 6. Cerebras Cloud
        cerebras_keys = os.getenv("CEREBRAS_API_KEY", "")
        for k in [x.strip() for x in cerebras_keys.split(",") if x.strip()]:
            self.pool.append(
                ProviderCredential(
                    provider_name="Cerebras Cloud",
                    api_key=k,
                    model="llama3.1-70b",
                    base_url="https://api.cerebras.ai/v1",
                )
            )

        logger.info("LLMRotator initialized with %d active credentials across %d providers", len(self.pool), len(set(c.provider_name for c in self.pool)))

    def _create_client_for_credential(self, cred: ProviderCredential) -> BaseChatModel:
        """Instantiate a ChatModel instance for the given provider credential."""
        if cred.provider_name == "Google Gemini":
            return ChatGoogleGenerativeAI(
                model=cred.model,
                google_api_key=cred.api_key,
                timeout=60.0,
            )
        else:
            return ChatOpenAI(
                model=cred.model,
                api_key=cred.api_key,
                base_url=cred.base_url,
                timeout=60.0,
            )

    def get_next_credential(self) -> ProviderCredential:
        """Select next available credential using Round-Robin with cooldown filtering."""
        if not self.pool:
            raise RuntimeError("No LLM API keys configured in pool. Please set GOOGLE_API_KEY, OPENROUTER_API_KEY, or GROQ_API_KEY in .env.")

        # Find healthy credentials
        now = time.time()
        available = [c for c in self.pool if c.is_available]

        if not available:
            # If all are in cooldown, pick the one that expires soonest
            logger.warning("All LLM providers currently in cooldown. Resetting earliest provider...")
            earliest = min(self.pool, key=lambda c: c.cooldown_until)
            earliest.cooldown_until = now
            return earliest

        # Round-robin selection among available
        self._index = (self._index + 1) % len(available)
        return available[self._index]

    def invoke_with_failover(self, messages: Any, **kwargs) -> Any:
        """Execute chat completion rotating between keys and failing over on rate limits (429)."""
        attempts = 0
        max_attempts = max(3, len(self.pool))

        while attempts < max_attempts:
            cred = self.get_next_credential()
            cred.total_calls += 1
            try:
                client = self._create_client_for_credential(cred)
                logger.debug("Dispatching LLM prompt to [%s / %s]", cred.provider_name, cred.model)
                res = client._generate(messages, **kwargs)
                return res
            except Exception as e:
                cred.total_errors += 1
                err_str = str(e)
                # Check for rate limits or quota exhaustion
                if any(k in err_str for k in ("429", "RESOURCE_EXHAUSTED", "quota", "RateLimitError", "rate_limit")):
                    logger.warning(
                        "Rate limit (429) on [%s]. Placing on 5m cooldown and rotating to next provider...",
                        cred.provider_name,
                    )
                    cred.cooldown_until = time.time() + 300.0  # 5 min cooldown
                else:
                    logger.warning("Error with provider [%s]: %s. Retrying next...", cred.provider_name, e)
                attempts += 1

        raise RuntimeError(f"All {len(self.pool)} LLM providers failed after {attempts} attempts.")

    def get_pool_status(self) -> list[dict[str, Any]]:
        """Return status and call metrics for all credentials in the pool."""
        now = time.time()
        return [
            {
                "provider": c.provider_name,
                "model": c.model,
                "key_preview": f"{c.api_key[:8]}...{c.api_key[-4:]}" if len(c.api_key) > 12 else "***",
                "status": "READY" if c.is_available else f"COOLDOWN ({int(c.cooldown_until - now)}s remaining)",
                "total_calls": c.total_calls,
                "total_errors": c.total_errors,
            }
            for c in self.pool
        ]


# Global singleton instance
LLM_ROTATOR = LLMRotator()
