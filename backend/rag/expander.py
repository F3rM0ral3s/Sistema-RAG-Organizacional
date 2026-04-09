"""Query expansion via the LLM (llama-server)."""

import json
import logging
import re

import httpx

from ..config import (
    EXPANSION_COUNT,
    EXPANSION_MAX_TOKENS,
    LLAMA_CHAT_ENDPOINT,
    LLAMA_LIGHT_URL,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

EXPANSION_SYSTEM_PROMPT = (
    "Eres un asistente especializado en reformular consultas de búsqueda. "
    "Tu tarea es generar variaciones de la consulta original para mejorar la recuperación de información "
    "en una base de datos de documentos universitarios de la UNAM (Gaceta UNAM). "
    "Las variaciones deben mantener la intención original pero usar sinónimos, "
    "diferentes estructuras gramaticales o enfoques alternativos.\n\n"
    "REGLAS:\n"
    "1. Responde ÚNICAMENTE con un JSON array de strings.\n"
    "2. No incluyas la consulta original.\n"
    "3. Mantén el idioma español.\n"
    "4. No agregues explicaciones."
)

EXPANSION_USER_TEMPLATE = (
    "Genera exactamente {n} reformulaciones de la siguiente consulta:\n\n"
    '"{query}"\n\n'
    "Responde solo con un JSON array. Ejemplo: "
    '["reformulación 1", "reformulación 2", "reformulación 3"]'
)


async def expand_query(query: str, n: int = EXPANSION_COUNT) -> list[str]:
    """
    Call the LLM to generate n rephrased versions of the query.
    Returns a list of expanded queries (may be shorter than n on failure).
    """
    url = f"{LLAMA_LIGHT_URL}{LLAMA_CHAT_ENDPOINT}"
    payload = {
        "messages": [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": EXPANSION_USER_TEMPLATE.format(n=n, query=query)},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": EXPANSION_MAX_TOKENS,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            content = re.sub(r"</?think>\s*", "", content).strip()

            # Parse JSON array from response
            # Handle cases where the model wraps in markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    l for l in lines if not l.startswith("```")
                )

            expanded = json.loads(content)
            if isinstance(expanded, list):
                return [str(q).strip() for q in expanded if str(q).strip()][:n]
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)

    return []
