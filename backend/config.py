"""Centralized configuration for the RAG backend."""

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_COLLECTION = "rag_documents"

# ── Embedding model (must match what was used to index) ───────────────────────
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
# Dense vector name in Qdrant (named vectors)
DENSE_VECTOR_NAME = "dense"
# Sparse vector name in Qdrant
SPARSE_VECTOR_NAME = "sparse"
# Weight for combining dense and sparse scores (0=dense only, 1=sparse only)
SPARSE_WEIGHT = 0.3

# ── LLM (llama-servers) ──────────────────────────────────────────────────────
# Light server: high concurrency, small context (guard + expander)
LLAMA_LIGHT_URL = "http://127.0.0.1:8080"
# Generator server: low concurrency, large context (RAG answer)
LLAMA_GENERATOR_URL = "http://127.0.0.1:8082"
LLAMA_CHAT_ENDPOINT = "/v1/chat/completions"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2048

# ── Query expansion ──────────────────────────────────────────────────────────
EXPANSION_COUNT = 3          # number of rephrased queries to generate
EXPANSION_MAX_TOKENS = 512

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_PER_QUERY = 50         # results per individual vector search (cast wider net)
RRF_K = 60                   # RRF constant (standard value)
RRF_TOP_K = 30               # final chunks returned after RRF fusion
GENERATOR_TOP_K = 30         # max chunks sent to the LLM
# ── Rate limiter ──────────────────────────────────────────────────────────────
# One query per user at a time (enforced by user_id)

# ── Jailbreak detection ──────────────────────────────────────────────────────
# Uses LLM-as-judge via the same llama-server (no extra config needed)
