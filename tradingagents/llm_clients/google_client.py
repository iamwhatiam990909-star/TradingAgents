import logging
import time
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list: [{'type': 'text', 'text': '...'}]
    This normalizes to string for consistent downstream handling.
    Also retries on rate-limit (429) errors with a 60-second wait.
    """

    _RATE_LIMIT_RETRY_MAX: int = 3
    _RATE_LIMIT_WAIT_SECONDS: int = 60

    @staticmethod
    def _is_rate_limit_error(e: Exception) -> bool:
        err_type = type(e).__name__.lower()
        err_msg = str(e).lower()
        keywords = (
            "429", "resource_exhausted", "resourceexhausted",
            "quota", "rate_limit", "rate limit", "too many requests",
        )
        return any(k in err_type or k in err_msg for k in keywords)

    def _normalize_content(self, response):
        content = response.content
        if isinstance(content, list):
            texts = [
                item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                else item if isinstance(item, str) else ""
                for item in content
            ]
            response.content = "\n".join(t for t in texts if t)
        return response

    def invoke(self, input, config=None, **kwargs):
        for attempt in range(self._RATE_LIMIT_RETRY_MAX + 1):
            try:
                return self._normalize_content(super().invoke(input, config, **kwargs))
            except Exception as e:
                if not self._is_rate_limit_error(e) or attempt == self._RATE_LIMIT_RETRY_MAX:
                    raise
                logger.warning(
                    "Gemini rate limit hit (attempt %d/%d). Waiting %ds before retry...",
                    attempt + 1, self._RATE_LIMIT_RETRY_MAX, self._RATE_LIMIT_WAIT_SECONDS,
                )
                time.sleep(self._RATE_LIMIT_WAIT_SECONDS)


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
