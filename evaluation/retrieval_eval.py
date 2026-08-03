"""
Retrieval evaluation: compares four retrieval strategies against the "kb"
and "combined" ground truth questions (the ones with a known-correct
expected_doc_id) — vector-only, BM25-only, hybrid RRF fusion (no rerank),
and the actual Phase 5 production pipeline (hybrid RRF + cross-encoder
rerank + per-doc diversity cap).

Metrics: Hit Rate@K (did the expected doc_id appear anywhere in the top K
*distinct documents*) and MRR@K (mean reciprocal rank of the expected doc's
first appearance). Scored at the document level, not the chunk level,
because a document can legitimately produce multiple chunks in a ranked
list — what actually matters for answer quality is whether the right
*document* surfaced early, not which specific chunk of it did.

This exists to put a number on the claim retrieval/hybrid_search.py's
docstring already makes qualitatively: that pure vector search let one
document's chunks crowd out a document that was actually the right answer
(Phase 4's "why would loyalty point balances be wrong" query returning 5/5
chunks from PM-001 and excluding RULE-001 entirely). If that's still true,
vector-only's Hit Rate@K should visibly trail hybrid+rerank's here.

Run inside the app container:
    docker compose exec app python evaluation/retrieval_eval.py
"""
import json

from pgvector.psycopg2 import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer

from common.postgresql import get_connection
from evaluation.settings import EVAL_TOP_K, GROUND_TRUTH_PATH, RETRIEVAL_RESULTS_PATH
from retrieval.hybrid_search import (
    bm25_search,
    load_all_chunks,
    reciprocal_rank_fusion,
    rerank_and_diversify,
    vector_search,
)
from retrieval.settings import (
    BM25_TOP_N,
    EMBED_MODEL_NAME,
    FUSED_SHORTLIST_SIZE,
    RERANK_MODEL_NAME,
    VECTOR_TOP_N,
)


def load_ground_truth():
    rows = []
    with open(GROUND_TRUTH_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row.get("expected_doc_id"):
                rows.append(row)
    return rows


def doc_ids_in_order(chunk_ids, chunk_by_id):
    """Chunk-id ranking -> doc-id ranking, deduped to first occurrence.
    Multiple chunks from the same doc collapse to one entry at the rank of
    the doc's earliest-appearing chunk — this is what makes the
    "one document crowds out the ranking" failure mode visible/measurable."""
    seen = []
    for cid in chunk_ids:
        doc_id = chunk_by_id[cid]["doc_id"]
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def hit_and_reciprocal_rank(doc_id_ranking, expected_doc_id, k=EVAL_TOP_K):
    top_k = doc_id_ranking[:k]
    if expected_doc_id in top_k:
        return 1, 1.0 / (top_k.index(expected_doc_id) + 1)
    return 0, 0.0


def evaluate_strategy(name, ranking_fn, ground_truth, results_log):
    hits, rrs = [], []
    for row in ground_truth:
        ranking = ranking_fn(row["question"])
        hit, rr = hit_and_reciprocal_rank(ranking, row["expected_doc_id"])
        hits.append(hit)
        rrs.append(rr)
        results_log.append({
            "gt_id": row["id"],
            "strategy": name,
            "question": row["question"],
            "expected_doc_id": row["expected_doc_id"],
            "top_k_doc_ids": ranking[:EVAL_TOP_K],
            "hit": hit,
            "reciprocal_rank": rr,
        })
    hit_rate = sum(hits) / len(hits) if hits else 0.0
    mrr = sum(rrs) / len(rrs) if rrs else 0.0
    return hit_rate, mrr


def main():
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth questions with a retrieval target.")
    if not ground_truth:
        print(f"No rows with expected_doc_id in {GROUND_TRUTH_PATH} — "
              "run evaluation/generate_ground_truth.py first.")
        return

    conn = get_connection()
    register_vector(conn)
    all_chunks = load_all_chunks(conn)
    chunk_by_id = {c["chunk_id"]: c for c in all_chunks}
    if not all_chunks:
        print("retrieval.kb_chunks is empty — run retrieval/embed_and_index.py first.")
        conn.close()
        return

    print("Loading embedding + reranker models...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    def vector_only(query):
        chunk_ids = vector_search(conn, embed_model, query, VECTOR_TOP_N)
        return doc_ids_in_order(chunk_ids, chunk_by_id)

    def bm25_only(query):
        chunk_ids = bm25_search(all_chunks, query, BM25_TOP_N)
        return doc_ids_in_order(chunk_ids, chunk_by_id)

    def hybrid_fused(query):
        vector_ranked = vector_search(conn, embed_model, query, VECTOR_TOP_N)
        bm25_ranked = bm25_search(all_chunks, query, BM25_TOP_N)
        fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
        return doc_ids_in_order([cid for cid, _ in fused], chunk_by_id)

    def hybrid_reranked(query):
        vector_ranked = vector_search(conn, embed_model, query, VECTOR_TOP_N)
        bm25_ranked = bm25_search(all_chunks, query, BM25_TOP_N)
        fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
        shortlist_ids = [cid for cid, _ in fused[:FUSED_SHORTLIST_SIZE]]
        if not shortlist_ids:
            return []
        reranked = rerank_and_diversify(query, shortlist_ids, chunk_by_id, reranker)
        return doc_ids_in_order([cid for cid, _ in reranked], chunk_by_id)

    strategies = [
        ("vector-only", vector_only),
        ("bm25-only", bm25_only),
        ("hybrid-rrf (no rerank)", hybrid_fused),
        ("hybrid-rrf + rerank (production)", hybrid_reranked),
    ]

    print(f"\nEvaluating {len(strategies)} retrieval strategies at K={EVAL_TOP_K}...\n")
    header = f"{'strategy':38s} {'hit rate@' + str(EVAL_TOP_K):>12s} {'mrr@' + str(EVAL_TOP_K):>10s}"
    print(header)
    print("-" * len(header))

    results_log = []
    for name, fn in strategies:
        hit_rate, mrr = evaluate_strategy(name, fn, ground_truth, results_log)
        print(f"{name:38s} {hit_rate:12.3f} {mrr:10.3f}")

    conn.close()

    RETRIEVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_RESULTS_PATH, "w") as f:
        for row in results_log:
            f.write(json.dumps(row) + "\n")
    print(f"\nPer-question results -> {RETRIEVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
