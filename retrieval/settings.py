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

# --- query_rewrite.py ---
# Deliberately the same model as evaluation/settings.py's JUDGE_MODEL_NAME
# (llama-3.1-8b-instant), for the same reason: a separate Groq free-tier
# quota bucket from the generation model (agent/settings.py's MODEL_NAME,
# llama-3.3-70b-versatile), and no tool-calling involved so 8b-instant's
# known tool_use_failed unreliability (see agent/settings.py) doesn't
# apply. Declared independently here rather than imported from
# evaluation/settings.py — evaluation/ already depends on retrieval/, and
# importing the other way would create a circular dependency direction.
QUERY_REWRITE_MODEL_NAME = "llama-3.1-8b-instant"
