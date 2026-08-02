"""
Shared retrieval-pipeline constants, used by search.py, embed_and_index.py,
and hybrid_search.py so the embedding model name (and the hybrid-search
tuning knobs) live in one place instead of drifting between scripts.
"""

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

VECTOR_TOP_N = 15
BM25_TOP_N = 15
RRF_K = 60
FUSED_SHORTLIST_SIZE = 10
FINAL_TOP_K = 5
MAX_PER_DOC = 2
