"""Shared async client for llama-server chat completions."""

import logging
import re

import httpx

from ..config import LLAMA_CHAT_ENDPOINT, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"</?think>\s*")
_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def strip_think_tags(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


async def chat_completion(
    base_url: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = LLM_TEMPERATURE,
    timeout: float | None = None,
) -> str:
    """POST a chat completion to llama-server and return the cleaned content."""
    url = f"{base_url}{LLAMA_CHAT_ENDPOINT}"
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = await get_client().post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        logger.error("Malformed LLM response: %s", str(data)[:300])
        raise RuntimeError("Respuesta inesperada del modelo de lenguaje") from e
    return strip_think_tags(content)
