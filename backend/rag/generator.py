"""LLM response generation via llama-server."""

import logging
import re

import httpx

from ..config import GENERATOR_TOP_K, LLAMA_GENERATOR_URL, LLM_MAX_TOKENS
from ..models import SourceChunk
from .llm_client import chat_completion

logger = logging.getLogger(__name__)

# Generator has 98304 context. At ~3.5 chars/token for Spanish:
#   98304 - 500 (system) - 2048 (response) = ~95756 tokens ≈ 335K chars
# Cap at ~120K chars (~34K tokens) to leave headroom for response.
MAX_CONTEXT_CHARS = 120_000

SYSTEM_PROMPT = (
    "Eres un asistente experto de la UNAM. Respondes preguntas usando SOLO "
    "la información del CONTEXTO proporcionado.\n\n"
    "INSTRUCCIONES:\n"
    "- Lee TODOS los fragmentos del contexto cuidadosamente. La respuesta "
    "puede estar repartida en varios fragmentos.\n"
    "- Escribe la respuesta con los datos concretos: nombres, fechas, cifras y hechos.\n"
    "- PROHIBIDO decir \"el texto menciona\", \"según el documento\" o "
    "\"en el contexto\". Escribe la información directamente.\n"
    "- Si encuentras información parcial, repórtala. Solo di "
    "\"No encontré información suficiente sobre este tema en la Gaceta\" "
    "cuando NINGÚN fragmento contenga información relevante.\n"
    "- Responde en español, de forma clara y completa.\n"
    "- Si hay mucha información relevante, organízala con viñetas.\n"
    "- Ignora fragmentos que no sean relevantes a la pregunta."
)

CONTEXT_TEMPLATE = "CONTEXTO:\n{chunks}\n\nPREGUNTA: {query}"


_GARBAGE_RE = re.compile(r"[^\w\s.,;:!?¿¡\-()\"\'/@#%&=+\[\]{}áéíóúüñÁÉÍÓÚÜÑ°]")


def _is_usable_chunk(text: str, max_garbage_ratio: float = 0.15) -> bool:
    """Return False if the chunk is mostly OCR garbage."""
    if not text or len(text) < 20:
        return False
    garbage_chars = len(_GARBAGE_RE.findall(text))
    return (garbage_chars / len(text)) < max_garbage_ratio


def _format_chunks(chunks: list[SourceChunk]) -> str:
    """Format chunks into context string, respecting the context budget."""
    parts = []
    total_chars = 0
    for c in chunks:
        if not _is_usable_chunk(c.text):
            continue
        meta = []
        if c.issue_date:
            meta.append(f"Fecha: {c.issue_date}")
        if c.source_pdf:
            meta.append(f"Fuente: {c.source_pdf}")
        header = " | ".join(meta)
        part = f"{header}\n{c.text}" if header else c.text

        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 500:
                parts.append(part[:remaining])
                logger.info(
                    "Context budget reached at chunk %d/%d (%d chars)",
                    len(parts), len(chunks), total_chars + remaining,
                )
            break
        parts.append(part)
        total_chars += len(part)
    logger.info("Formatted %d/%d chunks, %d chars total", len(parts), len(chunks), total_chars)
    return "\n\n---\n\n".join(parts)


async def generate_answer(query: str, chunks: list[SourceChunk]) -> str:
    """Generate a RAG answer using the LLM with retrieved context."""
    context = _format_chunks(chunks[:GENERATOR_TOP_K])
    user_content = CONTEXT_TEMPLATE.format(chunks=context, query=query)
    try:
        return await chat_completion(
            base_url=LLAMA_GENERATOR_URL,
            system=SYSTEM_PROMPT,
            user=user_content,
            max_tokens=LLM_MAX_TOKENS,
            timeout=120.0,
        )
    except httpx.HTTPStatusError as e:
        logger.error("LLM returned %s: %s", e.response.status_code, e.response.text[:300])
        raise RuntimeError(f"LLM error {e.response.status_code}") from e
