"""Embedding service using BAAI/bge-m3 (dense + sparse)."""

import logging

import numpy as np

from ..config import EMBEDDING_DIM, EMBEDDING_MODEL_ID

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self.model = None

    def load(self):
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_ID)
        self.model = BGEM3FlagModel(EMBEDDING_MODEL_ID, use_fp16=True, device="cuda")
        logger.info("Embedder loaded (BGE-M3, dense + sparse)")

    def embed_query(self, query: str) -> dict:
        """Embed a single query. Returns {'dense': np.ndarray, 'sparse': dict}."""
        out = self.model.encode(
            [query], return_dense=True, return_sparse=True
        )
        dense = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        assert dense.shape == (EMBEDDING_DIM,), f"Expected dim {EMBEDDING_DIM}, got {dense.shape}"
        sparse = out["lexical_weights"][0]  # {token_id: weight}
        return {"dense": dense, "sparse": sparse}

    def embed_queries(self, queries: list[str]) -> list[dict]:
        """Embed multiple queries. Returns list of {'dense': np.ndarray, 'sparse': dict}."""
        out = self.model.encode(
            queries, return_dense=True, return_sparse=True
        )
        results = []
        for i in range(len(queries)):
            dense = np.asarray(out["dense_vecs"][i], dtype=np.float32)
            sparse = out["lexical_weights"][i]
            results.append({"dense": dense, "sparse": sparse})
        return results


embedder = Embedder()
