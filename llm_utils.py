# =============================================
# LLM UTILS — Retry con Exponential Backoff
# =============================================
# File neutro — importato da agents.py e rag.py
# senza creare dipendenze circolari.

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import RateLimitError

logger = logging.getLogger(__name__)

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    before_sleep=lambda retry_state: logger.warning(
        f"Groq rate limit — attendo {retry_state.next_action.sleep:.0f}s "
        f"(tentativo {retry_state.attempt_number}/4)..."
    ),
)
def _invoke_with_backoff(chain, payload: dict) -> object:
    """
    Esegue chain.invoke() con retry automatico su RateLimitError.
    """
    return chain.invoke(payload)