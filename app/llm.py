import os
import random
import time
import logging

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from app.config import load_dotenv_if_enabled
from app.logging_config import monotonic_ms
from app.metrics import record_llm_usage, record_error

load_dotenv_if_enabled()

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL")

if not _OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")
if not _OPENAI_MODEL:
    raise RuntimeError("Missing OPENAI_MODEL in environment.")

client = OpenAI(api_key=_OPENAI_API_KEY)
MODEL_NAME = _OPENAI_MODEL

logger = logging.getLogger("app.llm")


class LLMRateLimitError(RuntimeError):
    pass


class LLMTransientError(RuntimeError):
    pass


def ask_llm(prompt: str, system_prompt: str = "You are a helpful research assistant.") -> str:
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "5"))
    base_delay_s = float(os.getenv("LLM_RETRY_BASE_DELAY_S", "0.5"))
    max_delay_s = float(os.getenv("LLM_RETRY_MAX_DELAY_S", "8"))

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        start_ms = monotonic_ms()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            elapsed_ms = monotonic_ms() - start_ms

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None

            record_llm_usage(
                model=MODEL_NAME,
                duration=elapsed_ms / 1000,
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                total_tokens=total_tokens or 0,
            )

            logger.info(
                "llm_response",
                extra={
                    "event": "llm_response",
                    "model": MODEL_NAME,
                    "latency_ms": elapsed_ms,
                    "attempt": attempt,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )
            return response.choices[0].message.content or "No response returned."

        except RateLimitError as exc:
            last_exc = exc
            elapsed_ms = monotonic_ms() - start_ms

            record_error(component="llm", error_type=type(exc).__name__)

            logger.warning(
                "llm_rate_limited",
                extra={
                    "event": "llm_rate_limited",
                    "model": MODEL_NAME,
                    "attempt": attempt,
                    "latency_ms": elapsed_ms,
                },
            )
            if attempt >= max_retries:
                raise LLMRateLimitError("LLM rate limit exceeded") from exc

        except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
            last_exc = exc
            elapsed_ms = monotonic_ms() - start_ms

            record_error(component="llm", error_type=type(exc).__name__)

            logger.warning(
                "llm_transient_error",
                extra={
                    "event": "llm_transient_error",
                    "model": MODEL_NAME,
                    "attempt": attempt,
                    "latency_ms": elapsed_ms,
                    "error_type": type(exc).__name__,
                },
            )
            if attempt >= max_retries:
                raise LLMTransientError("LLM temporarily unavailable") from exc

        delay = min(max_delay_s, base_delay_s * (2**attempt))
        delay = delay * (0.5 + random.random())
        logger.info(
            "llm_retry_sleep",
            extra={
                "event": "llm_retry_sleep",
                "attempt": attempt,
                "sleep_s": delay,
            },
        )
        time.sleep(delay)

    raise LLMTransientError("LLM temporarily unavailable") from last_exc