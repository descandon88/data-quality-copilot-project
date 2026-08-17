"""
Unit tests for retrieval/hybrid_search.py's pure ranking logic — tokenize,
Reciprocal Rank Fusion, BM25 search, and rerank + per-doc diversity cap.
Deliberately does NOT test vector_search() or load_all_chunks(), which
require a live Postgres connection with pgvector — that path is exercised
live by evaluation/retrieval_eval.py against real data instead, not
mocked here.

rerank_and_diversify() takes a real CrossEncoder in production; tests here
pass a small fake with a .predict(pairs) method instead, so these tests run
in milliseconds and don't need a downloaded model.
"""
from retrieval.hybrid_search import (
    bm25_search,
    reciprocal_rank_fusion,
    rerank_and_diversify,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_splits_on_non_alnum(self):
        assert tokenize("Why would loyalty point balances be WRONG?") == [
            "why", "would", "loyalty", "point", "balances", "be", "wrong",
        ]

    def test_splits_hyphenated_ids_into_separate_tokens(self):
        # PM-001 has no letters/digits contiguous across the hyphen, so it
        # becomes two tokens ("pm", "001") — worth pinning down explicitly
        # since this affects how well BM25 matches a half-remembered
        # incident id like "PM-001" against the corpus.
        assert tokenize("PM-001 root cause") == ["pm", "001", "root", "cause"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []


class TestReciprocalRankFusion:
    def test_single_list_preserves_relative_order(self):
        fused = reciprocal_rank_fusion(["a", "b", "c"])
        ids = [chunk_id for chunk_id, _ in fused]
        assert ids == ["a", "b", "c"]

    def test_item_ranked_high_in_both_lists_beats_item_in_only_one(self):
        vector_ranked = ["a", "b", "c"]
        bm25_ranked = ["a", "c", "b"]
        fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
        ids = [chunk_id for chunk_id, _ in fused]
        # "a" is rank 0 in both lists -> highest combined score.
        assert ids[0] == "a"

    def test_result_is_sorted_descending_by_score(self):
        fused = reciprocal_rank_fusion(["a", "b"], ["b", "a", "c"])
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_smaller_k_amplifies_rank_differences(self):
        # RRF's k dampens the effect of rank position; a smaller k should
        # widen the score gap between rank 0 and rank 1 for the same
        # ranked list, not narrow it.
        ranked = ["a", "b"]
        fused_small_k = dict(reciprocal_rank_fusion(ranked, k=1))
        fused_large_k = dict(reciprocal_rank_fusion(ranked, k=1000))
        gap_small_k = fused_small_k["a"] - fused_small_k["b"]
        gap_large_k = fused_large_k["a"] - fused_large_k["b"]
        assert gap_small_k > gap_large_k


class TestBm25Search:
    CHUNKS = [
        {"chunk_id": "pm-001__root-cause", "text": "duplicate loyalty point credits from non-idempotent earn events"},
        {"chunk_id": "pm-002__root-cause", "text": "null loyalty_id from an undetected upstream schema change"},
        {"chunk_id": "rule-004__what-it-checks", "text": "no account may have a points_balance below zero"},
    ]

    def test_ranks_exact_keyword_match_first(self):
        results = bm25_search(self.CHUNKS, "duplicate earn events", top_n=10)
        assert results[0] == "pm-001__root-cause"

    def test_excludes_chunks_with_zero_score(self):
        results = bm25_search(self.CHUNKS, "duplicate earn events", top_n=10)
        assert "rule-004__what-it-checks" not in results

    def test_respects_top_n(self):
        results = bm25_search(self.CHUNKS, "loyalty points balance account", top_n=1)
        assert len(results) <= 1

    def test_empty_corpus_returns_empty_list(self):
        assert bm25_search([], "anything", top_n=5) == []

    def test_query_with_no_matching_terms_returns_empty_list(self):
        results = bm25_search(self.CHUNKS, "completely unrelated banana smoothie", top_n=10)
        assert results == []


class FakeReranker:
    """Stands in for the real CrossEncoder — scores each (query, text) pair
    by a fixed lookup table instead of running a model, so tests are fast
    and deterministic."""

    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text

    def predict(self, pairs):
        return [self.scores_by_text.get(text, 0.0) for _query, text in pairs]


class TestRerankAndDiversify:
    def _chunk_by_id(self, entries):
        # entries: list of (chunk_id, doc_id, text)
        return {cid: {"chunk_id": cid, "doc_id": doc_id, "text": text} for cid, doc_id, text in entries}

    def test_orders_by_descending_rerank_score(self):
        entries = [("c1", "PM-001", "low relevance text"), ("c2", "PM-002", "high relevance text")]
        chunk_by_id = self._chunk_by_id(entries)
        reranker = FakeReranker({"low relevance text": 0.1, "high relevance text": 0.9})

        results = rerank_and_diversify("query", ["c1", "c2"], chunk_by_id, reranker, top_k=5, max_per_doc=5)
        assert [cid for cid, _ in results] == ["c2", "c1"]

    def test_caps_chunks_per_document(self):
        entries = [
            ("c1", "PM-001", "text a"), ("c2", "PM-001", "text b"), ("c3", "PM-001", "text c"),
            ("c4", "RULE-001", "text d"),
        ]
        chunk_by_id = self._chunk_by_id(entries)
        # All PM-001 chunks score higher than the RULE-001 chunk, so without
        # a diversity cap PM-001 would fill every slot — this is the exact
        # failure mode hybrid_search.py's own docstring describes.
        reranker = FakeReranker({
            "text a": 0.9, "text b": 0.8, "text c": 0.7, "text d": 0.5,
        })

        results = rerank_and_diversify(
            "query", ["c1", "c2", "c3", "c4"], chunk_by_id, reranker, top_k=5, max_per_doc=2
        )
        pm001_count = sum(1 for cid, _ in results if chunk_by_id[cid]["doc_id"] == "PM-001")
        assert pm001_count == 2
        # RULE-001's chunk should have made it in precisely because the cap
        # freed up a slot, even though it scored lowest overall.
        assert "c4" in [cid for cid, _ in results]

    def test_respects_top_k(self):
        entries = [(f"c{i}", f"DOC-{i}", f"text {i}") for i in range(5)]
        chunk_by_id = self._chunk_by_id(entries)
        reranker = FakeReranker({f"text {i}": float(i) for i in range(5)})

        results = rerank_and_diversify(
            "query", [cid for cid, _, _ in entries], chunk_by_id, reranker, top_k=2, max_per_doc=5
        )
        assert len(results) == 2

    def test_empty_shortlist_returns_empty_list(self):
        reranker = FakeReranker({})
        assert rerank_and_diversify("query", [], {}, reranker) == []
