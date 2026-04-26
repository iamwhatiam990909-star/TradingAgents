import asyncio
import logging
import random
import time
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)


# Optional integration with TradeGPT's process-wide throttle.
# Wrapped in try/except so this submodule remains usable standalone.
try:
    from app.pipeline.llm_throttle import acquire as _throttle_acquire
except Exception:  # pragma: no cover
    def _throttle_acquire(label: str = "") -> None:
        return None


# Tunables overridable via TradeGPT settings; fall back to safe defaults
# when imported standalone.
try:
    from app.config import settings as _app_settings
    _RETRY_MAX_ATTEMPTS = int(_app_settings.LLM_RETRY_MAX_ATTEMPTS)
    _RETRY_BASE_WAIT = int(_app_settings.LLM_RETRY_BASE_WAIT)
    _RETRY_MAX_WAIT = int(_app_settings.LLM_RETRY_MAX_WAIT)
except Exception:  # pragma: no cover
    _RETRY_MAX_ATTEMPTS = 5
    _RETRY_BASE_WAIT = 60
    _RETRY_MAX_WAIT = 240


_RATE_LIMIT_KEYWORDS = (
    "429", "resource_exhausted", "resourceexhausted",
    "quota", "rate_limit", "rate limit", "too many requests",
)


def _is_rate_limit_error(e: Exception) -> bool:
    err_type = type(e).__name__.lower()
    err_msg = str(e).lower()
    return any(k in err_type or k in err_msg for k in _RATE_LIMIT_KEYWORDS)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at _RETRY_MAX_WAIT."""
    raw = _RETRY_BASE_WAIT * (2 ** attempt)
    jitter = random.uniform(0.0, 5.0)
    return min(raw + jitter, _RETRY_MAX_WAIT)


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content + 429 retry on all paths.

    Gemini 3 models return content as list: [{'type': 'text', 'text': '...'}]
    This normalizes to string for consistent downstream handling.

    All call paths (sync invoke / async ainvoke / sync stream / async
    astream) acquire a throttle token first, then retry transient 429 /
    RESOURCE_EXHAUSTED errors with exponential backoff. Retries are
    bounded by ``LLM_RETRY_MAX_ATTEMPTS``; permanent quota / billing
    failures still surface to the caller after the cap.
    """

    def _normalize_content(self, response):
        content = getattr(response, "content", None)
        if isinstance(content, list):
            texts = [
                item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                else item if isinstance(item, str) else ""
                for item in content
            ]
            response.content = "\n".join(t for t in texts if t)
        return response

    def _retry_sync(self, fn, *args, **kwargs):
        for attempt in range(_RETRY_MAX_ATTEMPTS + 1):
            _throttle_acquire(label="gemini")
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not _is_rate_limit_error(e) or attempt == _RETRY_MAX_ATTEMPTS:
                    raise
                wait = _backoff_seconds(attempt)
                logger.warning(
                    "Gemini 429 (attempt %d/%d). Backing off %.1fs.",
                    attempt + 1, _RETRY_MAX_ATTEMPTS, wait,
                )
                time.sleep(wait)

    async def _retry_async(self, fn, *args, **kwargs):
        for attempt in range(_RETRY_MAX_ATTEMPTS + 1):
            # Token bucket acquire blocks the event loop briefly; offload.
            await asyncio.to_thread(_throttle_acquire, "gemini")
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not _is_rate_limit_error(e) or attempt == _RETRY_MAX_ATTEMPTS:
                    raise
                wait = _backoff_seconds(attempt)
                logger.warning(
                    "Gemini 429 async (attempt %d/%d). Backing off %.1fs.",
                    attempt + 1, _RETRY_MAX_ATTEMPTS, wait,
                )
                await asyncio.sleep(wait)

    def invoke(self, input, config=None, **kwargs):
        return self._normalize_content(
            self._retry_sync(super().invoke, input, config, **kwargs),
        )

    async def ainvoke(self, input, config=None, **kwargs):
        return self._normalize_content(
            await self._retry_async(super().ainvoke, input, config, **kwargs),
        )

    def stream(self, input, config=None, **kwargs):
        # Streaming returns a generator. Acquire + retry the *initial*
        # connection; mid-stream errors will still propagate, but the
        # 429 always fires on the first chunk request.
        gen = self._retry_sync(super().stream, input, config, **kwargs)
        for chunk in gen:
            yield chunk

    async def astream(self, input, config=None, **kwargs):
        gen = await self._retry_async(super().astream, input, config, **kwargs)
        async for chunk in gen:
            yield chunk


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        llm_kwargs = {"model": self.model}

        for key in ("timeout", "max_retries", "google_api_key", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Map thinking_level to appropriate API param based on model
        # Gemini 3 Pro: low, high
        # Gemini 3 Flash: minimal, low, medium, high
        # Gemini 2.5: thinking_budget (0=disable, -1=dynamic)
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = self.model.lower()
            if "gemini-3" in model_lower:
                # Gemini 3 Pro doesn't support "minimal", use "low" instead
                if "pro" in model_lower and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                # Gemini 2.5: map to thinking_budget
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
