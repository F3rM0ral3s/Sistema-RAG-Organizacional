"""Query expansion via the LLM (llama-server)."""

import json
import logging

from ..config import EXPANSION_COUNT, EXPANSION_MAX_TOKENS, LLAMA_LIGHT_URL
from .llm_client import chat_completion

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


def _strip_markdown_fence(content: str) -> str:
    if not content.startswith("```"):
        return content
    return "\n".join(l for l in content.split("\n") if not l.startswith("```"))


async def expand_query(query: str, n: int = EXPANSION_COUNT) -> list[str]:
    """Call the LLM to generate n rephrased versions. Empty list on failure."""
    try:
        content = await chat_completion(
            base_url=LLAMA_LIGHT_URL,
            system=EXPANSION_SYSTEM_PROMPT,
            user=EXPANSION_USER_TEMPLATE.format(n=n, query=query),
            max_tokens=EXPANSION_MAX_TOKENS,
            timeout=60.0,
        )
        expanded = json.loads(_strip_markdown_fence(content))
        if isinstance(expanded, list):
            return [str(q).strip() for q in expanded if str(q).strip()][:n]
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
    return []
