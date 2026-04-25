"""
FastAPI backend for the organizational RAG system.

Architecture:
- Event-based: submit query → poll for result (processing / processed / rejected / failed)
- Rate limiter: one query per user at a time
- Query expansion + RRF fusion
- Jailbreak detection (runs inside the pipeline so submit returns immediately)
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QueryRequest,
    QueryResultResponse,
    QueryStatus,
    QuerySubmitResponse,
)
from .rag.embedder import embedder
from .rag.expander import expand_query
from .rag.generator import generate_answer
from .rag.guard import detect_jailbreak
from .rag.llm_client import close_client
from .rag.retriever import retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Retain query results for one hour, then evict on next write.
RESULT_TTL_SECONDS = 3600

# query_id → (created_at_monotonic, response)
query_results: dict[str, tuple[float, QueryResultResponse]] = {}
# user_id → query_id (active query per user, for rate limiting)
active_user_queries: dict[str, str] = {}
user_lock = asyncio.Lock()


def _save_result(query_id: str, response: QueryResultResponse) -> None:
    now = time.monotonic()
    query_results[query_id] = (now, response)
    cutoff = now - RESULT_TTL_SECONDS
    for qid in [k for k, (ts, _) in query_results.items() if ts < cutoff]:
        del query_results[qid]


def _get_result(query_id: str) -> QueryResultResponse | None:
    item = query_results.get(query_id)
    return item[1] if item else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models...")
    embedder.load()
    retriever.load()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")
    await close_client()


app = FastAPI(
    title="Gaceta UNAM RAG API",
    description="Sistema RAG organizacional para la Gaceta UNAM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_rag_pipeline(query_id: str, query: str, user_id: str):
    """Full pipeline: guard → expand → embed → retrieve (RRF) → generate."""
    try:
        rejection = await detect_jailbreak(query)
        if rejection:
            _save_result(query_id, QueryResultResponse(
                query_id=query_id,
                status=QueryStatus.REJECTED,
                rejection_reason=rejection,
            ))
            return

        expanded_queries = await expand_query(query)
        logger.info("Expanded %d queries from original", len(expanded_queries))

        all_queries = [query] + expanded_queries
        all_embeddings = await asyncio.to_thread(embedder.embed_queries, all_queries)

        chunks = await asyncio.to_thread(
            retriever.search_multi_rrf, all_embeddings, original_query=query
        )
        logger.info("Retrieved %d chunks via RRF", len(chunks))

        answer = await generate_answer(query, chunks)

        _save_result(query_id, QueryResultResponse(
            query_id=query_id,
            status=QueryStatus.PROCESSED,
            answer=answer,
            sources=chunks,
            expanded_queries=expanded_queries,
        ))

    except Exception as e:
        logger.error("RAG pipeline failed for query_id=%s: %s", query_id, e, exc_info=True)
        _save_result(query_id, QueryResultResponse(
            query_id=query_id,
            status=QueryStatus.FAILED,
            answer=f"Error al procesar la consulta: {e}",
        ))
    finally:
        async with user_lock:
            if active_user_queries.get(user_id) == query_id:
                del active_user_queries[user_id]


@app.post("/api/query", response_model=QuerySubmitResponse)
async def submit_query(req: QueryRequest):
    """Submit a query for processing. Rate limited: one query per user at a time."""
    async with user_lock:
        if req.user_id in active_user_queries:
            raise HTTPException(
                status_code=429,
                detail="Ya tienes una consulta en proceso. Espera a que termine antes de enviar otra.",
            )
        query_id = str(uuid.uuid4())
        active_user_queries[req.user_id] = query_id

    _save_result(query_id, QueryResultResponse(
        query_id=query_id,
        status=QueryStatus.PROCESSING,
    ))
    asyncio.create_task(run_rag_pipeline(query_id, req.query, req.user_id))
    return QuerySubmitResponse(query_id=query_id, status=QueryStatus.PROCESSING)


@app.get("/api/query/{query_id}", response_model=QueryResultResponse)
async def get_query_result(query_id: str):
    """Poll for query result. Returns current status."""
    result = _get_result(query_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Consulta no encontrada.")
    return result


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "embedder": embedder.model is not None,
        "retriever": retriever.client is not None,
    }
