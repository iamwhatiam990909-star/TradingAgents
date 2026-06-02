import asyncio
import logging
import random
import time
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)


# Optional integration with TradeGPT's process-wide throttle.
try:
    from app.pipeline.llm_throttle import acquire as _throttle_acquire
except Exception:  # pragma: no cover
    def _throttle_acquire(label: str = "", tokens: int = 1) -> None:
        return None


try:
    from app.config import settings as _app_settings
    _RETRY_MAX_ATTEMPTS = int(_app_settings.LLM_RETRY_MAX_ATTEMPTS)
    _RETRY_BASE_WAIT = int(_app_settings.LLM_RETRY_BASE_WAIT)
    _RETRY_MAX_WAIT = int(_app_settings.LLM_RETRY_MAX_WAIT)
except Exception:  # pragma: no cover
    _RETRY_MAX_ATTEMPTS = 5
    _RETRY_BASE_WAIT = 60
    _RETRY_MAX_WAIT = 240


# Permanent failures: never retry, never auto-fallback to other provider.
_PERMANENT_KEYWORDS = (
    "credit balance", "billing", "purchase credits",
    "insufficient_quota",
)

# Transient rate-limit failures: retry with backoff.
_TRANSIENT_KEYWORDS = (
    "rate_limit", "rate limit", "overloaded", "capacity",
    "too many requests", "429",
)


def _is_permanent(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in _PERMANENT_KEYWORDS)


def _is_transient_rate_limit(e: Exception) -> bool:
    if _is_permanent(e):
        return False
    err_type = type(e).__name__.lower()
    if "ratelimit" in err_type:
        return True
    msg = str(e).lower()
    return any(k in msg for k in _TRANSIENT_KEYWORDS)


def _backoff_seconds(attempt: int) -> float:
    raw = _RETRY_BASE_WAIT * (2 ** attempt)
    jitter = random.uniform(0.0, 5.0)
    return min(raw + jitter, _RETRY_MAX_WAIT)


def _estimate_tokens(input_obj: Any) -> int:
    """Rough input-token estimate (~4 chars/token; conservative for zh)."""
    try:
        return max(1, len(str(input_obj)) // 4)
    except Exception:
        return 1


class ThrottledChatAnthropic(ChatAnthropic):
    """ChatAnthropic with throttle + transient-429 retry on all paths."""

    def _retry_sync(self, fn, *args, **kwargs):
        est = _estimate_tokens(args[0]) if args else 1
        for attempt in range(_RETRY_MAX_ATTEMPTS + 1):
            _throttle_acquire(label="anthropic", tokens=est)
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not _is_transient_rate_limit(e) or attempt == _RETRY_MAX_ATTEMPTS:
                    raise
                wait = _backoff_seconds(attempt)
                logger.warning(
                    "Anthropic 429 (attempt %d/%d). Backing off %.1fs.",
                    attempt + 1, _RETRY_MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)

    async def _retry_async(self, fn, *args, **kwargs):
        est = _estimate_tokens(args[0]) if args else 1
        for attempt in range(_RETRY_MAX_ATTEMPTS + 1):
            await asyncio.to_thread(_throttle_acquire, "anthropic", est)
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not _is_transient_rate_limit(e) or attempt == _RETRY_MAX_ATTEMPTS:
                    raise
                wait = _backoff_seconds(attempt)
                logger.warning(
                    "Anthropic 429 async (attempt %d/%d). Backing off %.1fs.",
                    attempt + 1, _RETRY_MAX_ATTEMPTS, wait,
                )
                await asyncio.sleep(wait)

    def invoke(self, input, config=None, **kwargs):
        return self._retry_sync(super().invoke, input, config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        return await self._retry_async(super().ainvoke, input, config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        gen = self._retry_sync(super().stream, input, config, **kwargs)
        for chunk in gen:
            yield chunk

    async def astream(self, input, config=None, **kwargs):
        gen = await self._retry_async(super().astream, input, config, **kwargs)
        async for chunk in gen:
            yield chunk


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        llm_kwargs = {"model": self.model}

        for key in ("timeout", "max_retries", "api_key", "max_tokens", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return ThrottledChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
