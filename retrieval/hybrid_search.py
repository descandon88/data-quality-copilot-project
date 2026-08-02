"""
Hybrid search + reranking over retrieval.kb_chunks: combines BM25 (keyword)
and vector (semantic) retrieval via Reciprocal Rank Fusion, then reranks the
fused shortlist with a cross-encoder.

Why hybrid, not just vector search: Phase 4's pure vector search returned
all 5 top results from a single document (PM-001) for the query "why would
loyalty point balances be wrong" — semantically reasonable, but it silently
excluded RULE-001, which is the actual fix for that exact problem. Dense
embeddings capture topical similarity well but don't guarantee exact-term
overlap (`points_balance`, `loyalty_id`) stays weighted the way BM25 would
weight it. BM25 catches what vector search can under-weight; vector search
catches paraphrases BM25 can't match at all. Fusing both ranked lists, then
reranking the fused shortlist with a cross-encoder — which scores the
(query, chunk) pair jointly instead of comparing two independent embeddings
— gets closer to what a person skimming both lists would actually pick.

Pipeline: vector search (top N) + BM25 search (top N) -> Reciprocal Rank
Fusion -> cross-encoder rerank of the fused shortlist -> final top K, capped
at MAX_PER_DOC chunks per document so one document can't crowd out the rest
of the result set (the same failure mode observed in Phase 4).

Run inside the app container:
    docker compose exec app python retrieval/hybrid_search.py "why would loyalty point balances be wrong"
"""
import re
import sys

from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from common.postgresql import get_connection
from retrieval.settings import (
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    VECTOR_TOP_N,
    BM25_TOP_N,
    RRF_K,
    FUSED_SHORTLIST_SIZE,
    FINAL_TOP_K,
    MAX_PER_DOC,
)


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_all_chunks(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, doc_id, title, section, chunk_text
            FROM retrieval.kb_chunks
        """)
        rows = cur.fetchall()
    return [
        {"chunk_id": r[0], "doc_id": r[1], "title": r[2], "section": r[3], "text": r[4]}
        for r in rows
    ]


def vector_search(conn, embed_model, query, top_n):
    query_embedding = embed_model.encode(query, normalize_embeddings=True)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id
            FROM retrieval.kb_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (query_embedding, top_n))
        rows = cur.fetchall()
    return [r[0] for r in rows]


def bm25_search(all_chunks, query, top_n):
    if not all_chunks:
        return []
    corpus_tokens = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(all_chunks)), key=lambda i: scores[i], reverse=True)
    return [all_chunks[i]["chunk_id"] for i in ranked[:top_n] if scores[i] > 0]


def reciprocal_rank_fusion(*ranked_lists, k=RRF_K):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank_and_diversify(query, shortlist_ids, chunk_by_id, reranker,
                          top_k=FINAL_TOP_K, max_per_doc=MAX_PER_DOC):
    pairs = [(query, chunk_by_id[cid]["text"]) for cid in shortlist_ids]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(shortlist_ids, rerank_scores), key=lambda x: x[1], reverse=True)

    final_results = []
    per_doc_count = {}
    for chunk_id, score in reranked:
        doc_id = chunk_by_id[chunk_id]["doc_id"]
        if per_doc_count.get(doc_id, 0) >= max_per_doc:
            continue
        final_results.append((chunk_id, score))
        per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
        if len(final_results) >= top_k:
            break
    return final_results


def main():
    query = " ".join(sys.argv[1:]) or "why would loyalty point balances be wrong"
    print(f"Query: {query!r}\n")

    conn = get_connection()
    register_vector(conn)

    all_chunks = load_all_chunks(conn)
    chunk_by_id = {c["chunk_id"]: c for c in all_chunks}

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    vector_ranked = vector_search(conn, embed_model, query, VECTOR_TOP_N)
    bm25_ranked = bm25_search(all_chunks, query, BM25_TOP_N)
    conn.close()

    fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
    shortlist_ids = [chunk_id for chunk_id, _ in fused[:FUSED_SHORTLIST_SIZE]]

    print(f"Fused shortlist ({len(shortlist_ids)} candidates, vector + BM25 via RRF):")
    for chunk_id in shortlist_ids:
        c = chunk_by_id[chunk_id]
        print(f"  {c['doc_id']} — {c['section']}")
    print()

    reranker = CrossEncoder(RERANK_MODEL_NAME)
    final_results = rerank_and_diversify(query, shortlist_ids, chunk_by_id, reranker)

    print(f"Final top {len(final_results)} after cross-encoder rerank (max {MAX_PER_DOC}/doc):\n")
    for chunk_id, score in final_results:
        c = chunk_by_id[chunk_id]
        print(f"[{score:.3f}] {c['doc_id']} — {c['title']} — {c['section']}")
        print(f"   {c['text'][:160].strip()}...")
        print()


if __name__ == "__main__":
    main()