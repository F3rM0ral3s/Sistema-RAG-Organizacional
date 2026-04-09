"""
FastAPI backend for the organizational RAG system.

Architecture:
- Event-based: submit query → poll for result (processing / processed / rejected)
- Rate limiter: one query per user at a time
- Query expansion + RRF fusion
- Jailbreak detection
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QueryRequest,
    QueryResultResponse,
    QueryStatus,
    QuerySubmitResponse,
    SourceChunk,
)
from .rag.embedder import embedder
from .rag.expander import expand_query
from .rag.generator import generate_answer
from .rag.guard import detect_jailbreak
from .rag.retriever import retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── In-memory stores ─────────────────────────────────────────────────────────
# query_id → QueryResultResponse
query_results: dict[str, QueryResultResponse] = {}
# user_id → query_id (tracks the active query per user for rate limiting)
active_user_queries: dict[str, str] = {}
# Lock for active_user_queries
user_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and connect to Qdrant on startup."""
    logger.info("Starting up — loading models...")
    embedder.load()
    retriever.load()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


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


# ── RAG pipeline (runs in background task) ────────────────────────────────────

async def run_rag_pipeline(query_id: str, query: str, user_id: str):
    """Full RAG pipeline: expand → embed → retrieve (RRF) → generate."""
    try:
        # 1. Query expansion (generate 3 rephrased queries via LLM)
        expanded_queries = await expand_query(query)
        logger.info("Expanded %d queries from original", len(expanded_queries))

        # 2. Embed all queries (original + expanded) — dense + sparse
        all_queries = [query] + expanded_queries
        all_embeddings = await asyncio.to_thread(embedder.embed_queries, all_queries)

        # 3. Hybrid retrieval: dense + sparse search with RRF fusion
        chunks = await asyncio.to_thread(
            retriever.search_multi_rrf, all_embeddings, original_query=query
        )
        logger.info("Retrieved %d chunks via RRF", len(chunks))

        # 5. Generate answer using LLM + retrieved context
        answer = await generate_answer(query, chunks)

        # 6. Store result
        query_results[query_id] = QueryResultResponse(
            query_id=query_id,
            status=QueryStatus.PROCESSED,
            answer=answer,
            sources=chunks,
            expanded_queries=expanded_queries,
        )

    except Exception as e:
        logger.error("RAG pipeline failed for query_id=%s: %s", query_id, e, exc_info=True)
        query_results[query_id] = QueryResultResponse(
            query_id=query_id,
            status=QueryStatus.PROCESSED,
            answer=f"Error al procesar la consulta: {e}",
            sources=[],
            expanded_queries=[],
        )
    finally:
        # Release user rate limit
        async with user_lock:
            if active_user_queries.get(user_id) == query_id:
                del active_user_queries[user_id]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QuerySubmitResponse)
async def submit_query(req: QueryRequest):
    """
    Submit a query for processing.
    Rate limited: one query per user at a time.
    """
    # Jailbreak check (LLM-as-judge)
    rejection = await detect_jailbreak(req.query)
    if rejection:
        query_id = str(uuid.uuid4())
        query_results[query_id] = QueryResultResponse(
            query_id=query_id,
            status=QueryStatus.REJECTED,
            rejection_reason=rejection,
        )
        return QuerySubmitResponse(query_id=query_id, status=QueryStatus.REJECTED)

    # Rate limit: one query per user
    async with user_lock:
        if req.user_id in active_user_queries:
            raise HTTPException(
                status_code=429,
                detail="Ya tienes una consulta en proceso. Espera a que termine antes de enviar otra.",
            )
        query_id = str(uuid.uuid4())
        active_user_queries[req.user_id] = query_id

    # Initialize as processing
    query_results[query_id] = QueryResultResponse(
        query_id=query_id,
        status=QueryStatus.PROCESSING,
    )

    # Launch pipeline in background
    asyncio.create_task(run_rag_pipeline(query_id, req.query, req.user_id))

    return QuerySubmitResponse(query_id=query_id, status=QueryStatus.PROCESSING)


@app.get("/api/query/{query_id}", response_model=QueryResultResponse)
async def get_query_result(query_id: str):
    """Poll for query result. Returns current status."""
    if query_id not in query_results:
        raise HTTPException(status_code=404, detail="Consulta no encontrada.")
    return query_results[query_id]


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "embedder": embedder.model is not None,
        "retriever": retriever.client is not None,
    }
