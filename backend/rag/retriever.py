"""Qdrant retrieval with hybrid dense + sparse search and RRF fusion."""

import logging
import re
import unicodedata
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedSparseVector,
    NamedVector,
    Prefetch,
    Query,
    QueryRequest,
    SparseVector,
)

from ..config import (
    DENSE_VECTOR_NAME,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RRF_K,
    RRF_TOP_K,
    SPARSE_VECTOR_NAME,
    SPARSE_WEIGHT,
    TOP_K_PER_QUERY,
)
from ..models import SourceChunk

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase and strip accents for keyword matching."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query, filtering stopwords."""
    stopwords = {
        "el", "la", "los", "las", "de", "del", "en", "un", "una", "y", "o",
        "que", "es", "por", "con", "para", "al", "se", "su", "ha", "fue",
        "como", "mas", "pero", "sus", "le", "ya", "lo", "me", "sin", "sobre",
        "este", "ser", "son", "no", "si", "muy", "puede", "todos", "esta",
        "puedes", "mencionar", "cuales", "cuáles", "dime", "cuantos",
        "a", "ante", "bajo", "cabe", "contra", "desde", "entre", "hacia",
        "hasta", "segun", "tras",
    }
    words = re.findall(r"[a-záéíóúüñ]+", _normalize(query))
    return [w for w in words if len(w) > 2 and w not in stopwords]


def _keyword_boost(text: str, keywords: list[str]) -> float:
    """Score how many distinct keywords appear in the text (0.0 to 1.0)."""
    if not keywords:
        return 0.0
    norm_text = _normalize(text)
    hits = sum(1 for kw in keywords if kw in norm_text)
    return hits / len(keywords)


class Retriever:
    def __init__(self):
        self.client: QdrantClient | None = None

    def load(self):
        logger.info("Connecting to Qdrant server at %s", QDRANT_URL)
        self.client = QdrantClient(url=QDRANT_URL, timeout=60)
        info = self.client.get_collection(QDRANT_COLLECTION)
        logger.info(
            "Qdrant collection '%s': %d points",
            QDRANT_COLLECTION,
            info.points_count,
        )

    def _search_dense(
        self, dense_vec: np.ndarray, top_k: int
    ) -> list[tuple[str, float, dict]]:
        """Run a dense-only vector search."""
        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=dense_vec.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return [(str(h.id), h.score, h.payload) for h in response.points]

    def _search_sparse(
        self, sparse_weights: dict, top_k: int
    ) -> list[tuple[str, float, dict]]:
        """Run a sparse-only vector search."""
        if not sparse_weights:
            return []
        indices = sorted(sparse_weights.keys())
        values = [sparse_weights[i] for i in indices]
        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return [(str(h.id), h.score, h.payload) for h in response.points]

    def search_multi_rrf(
        self,
        query_embeddings: list[dict],
        top_k_per: int = TOP_K_PER_QUERY,
        top_k_final: int = RRF_TOP_K,
        k: int = RRF_K,
        original_query: str = "",
    ) -> list[SourceChunk]:
        """
        Hybrid search: run dense + sparse searches for each query variant,
        then fuse all results using RRF + keyword boost.

        Each query_embedding is {'dense': np.ndarray, 'sparse': dict}.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        payloads: dict[str, dict] = {}

        num_searches = 0
        for qe in query_embeddings:
            # Dense search
            dense_hits = self._search_dense(qe["dense"], top_k=top_k_per)
            for rank, (pid, _score, payload) in enumerate(dense_hits, start=1):
                rrf_scores[pid] += 1.0 / (k + rank)
                if pid not in payloads:
                    payloads[pid] = payload
            num_searches += 1

            # Sparse search
            sparse_hits = self._search_sparse(qe["sparse"], top_k=top_k_per)
            for rank, (pid, _score, payload) in enumerate(sparse_hits, start=1):
                rrf_scores[pid] += 1.0 / (k + rank)
                if pid not in payloads:
                    payloads[pid] = payload
            num_searches += 1

        logger.info(
            "Ran %d searches (%d dense + %d sparse), %d unique candidates",
            num_searches, num_searches // 2, num_searches // 2, len(rrf_scores),
        )

        # Normalize RRF scores
        max_score = num_searches / (k + 1)
        if max_score == 0:
            max_score = 1.0

        # Keyword re-ranking
        keywords = _extract_keywords(original_query) if original_query else []
        if keywords:
            logger.info("Keyword re-ranking with: %s", keywords)

        scored = []
        for point_id, rrf_score in rrf_scores.items():
            norm_rrf = rrf_score / max_score
            text = payloads[point_id].get("text", "")
            kw_boost = _keyword_boost(text, keywords) if keywords else 0.0
            final_score = norm_rrf * (1.0 + kw_boost)
            scored.append((point_id, final_score, kw_boost))

        scored.sort(key=lambda x: x[1], reverse=True)

        chunks = []
        for point_id, final_score, kw_boost in scored[:top_k_final]:
            p = payloads[point_id]
            chunks.append(SourceChunk(
                text=p.get("text", ""),
                doc_id=p.get("doc_id", ""),
                chunk_id=p.get("chunk_id", ""),
                corpus=p.get("corpus", ""),
                decade=p.get("decade", ""),
                issue_date=p.get("issue_date", ""),
                source_pdf=p.get("source_pdf", ""),
                chunk_index=p.get("chunk_index") or 0,
                score=round(final_score, 4),
            ))
            if kw_boost > 0:
                logger.debug(
                    "Chunk %s: kw_boost=%.2f final=%.4f",
                    point_id[:8], kw_boost, final_score,
                )
        return chunks


retriever = Retriever()
