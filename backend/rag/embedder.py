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

    def embed_queries(self, queries: list[str]) -> list[dict]:
        """Embed queries. Returns list of {'dense': np.ndarray, 'sparse': dict}."""
        out = self.model.encode(queries, return_dense=True, return_sparse=True)
        results = []
        for i in range(len(queries)):
            dense = np.asarray(out["dense_vecs"][i], dtype=np.float32)
            if dense.shape != (EMBEDDING_DIM,):
                raise RuntimeError(
                    f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {dense.shape}"
                )
            results.append({"dense": dense, "sparse": out["lexical_weights"][i]})
        return results


embedder = Embedder()
