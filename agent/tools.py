"""
Tool implementations exposed to the agent's tool-use loop in copilot.py.

Two tools:
  - search_knowledge_base: RAG retrieval over postmortems/rules (reuses the
    Phase 5 hybrid search + rerank pipeline), for "what happened / what's
    our policy" questions.
  - query_warehouse: read-only SQL against the actual data (bronze/raw/
    retrieval schemas), for "what's true right now" questions the knowledge
    base doesn't and can't answer — it only knows about documented
    incidents, not live data.

Splitting these into two tools, instead of one do-everything RAG call, is
the actual point of "agent routing" here: the model decides per-question
whether it needs documented context, current data, or both, rather than a
fixed pipeline that always does the same thing regardless of what's asked.
"""
import re

from pgvector.psycopg2 import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer

from common.postgresql import get_connection
from retrieval.hybrid_search import (
    bm25_search,
    load_all_chunks,
    reciprocal_rank_fusion,
    rerank_and_diversify,
    vector_search,
)
from retrieval.query_rewrite import rewrite_query
from retrieval.settings import (
    BM25_TOP_N,
    EMBED_MODEL_NAME,
    FUSED_SHORTLIST_SIZE,
    RERANK_MODEL_NAME,
    VECTOR_TOP_N,
)

# Loaded lazily and cached at module level — the agent may call this tool
# several times within one conversation turn, and reloading a transformer
# model on every call would be needlessly slow.
_embed_model = None
_reranker = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def search_knowledge_base(query: str, k: int = 5) -> str:
    """Rewrites the query (retrieval/query_rewrite.py — a cheap, separate-
    quota LLM call that reformulates the question into the knowledge
    base's own phrasing, fails open to the original query on any error),
    then runs the Phase 5 hybrid search + rerank pipeline and returns a
    formatted string of the top results, each tagged with its doc_id so the
    model can cite sources in its final answer."""
    search_query = rewrite_query(query)

    conn = get_connection()
    register_vector(conn)
    try:
        all_chunks = load_all_chunks(conn)
        chunk_by_id = {c["chunk_id"]: c for c in all_chunks}

        embed_model = _get_embed_model()
        vector_ranked = vector_search(conn, embed_model, search_query, VECTOR_TOP_N)
        bm25_ranked = bm25_search(all_chunks, search_query, BM25_TOP_N)
    finally:
        conn.close()

    fused = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
    shortlist_ids = [chunk_id for chunk_id, _ in fused[:FUSED_SHORTLIST_SIZE]]
    if not shortlist_ids:
        return "No matching documents found in the knowledge base."

    reranker = _get_reranker()
    results = rerank_and_diversify(search_query, shortlist_ids, chunk_by_id, reranker, top_k=k)
    if not results:
        return "No matching documents found in the knowledge base."

    formatted = []
    for chunk_id, score in results:
        c = chunk_by_id[chunk_id]
        formatted.append(
            f"[{c['doc_id']}] {c['title']} — {c['section']} (relevance: {score:.2f})\n{c['text']}"
        )
    return "\n\n---\n\n".join(formatted)


# Query-level SQL safety guard. This is a demo-appropriate guard, not a
# production one — the real hardening step would be a dedicated read-only
# Postgres role (GRANT SELECT only, no write/DDL privileges at all) so the
# database itself refuses anything this regex misses, rather than relying
# solely on pattern-matching the SQL text. Worth adding before this ever
# runs against anything other than a disposable local dataset.
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|create|"
    r"attach|copy|vacuum|exec|execute|call)\b",
    re.IGNORECASE,
)


def query_warehouse(sql: str, max_rows: int = 50) -> str:
    """Runs a read-only SQL query against the warehouse and returns the
    result as a formatted table. Rejects anything that isn't a plain
    SELECT, and caps the row count so one query can't dump the warehouse
    into the model's context."""
    stripped = sql.strip().rstrip(";")

    if not re.match(r"(?is)^\s*select\b", stripped):
        return "Rejected: only SELECT statements are allowed."

    if _FORBIDDEN_SQL.search(stripped):
        return "Rejected: query contains a disallowed keyword (only read-only SELECTs are permitted)."

    if not re.search(r"\blimit\s+\d+\b", stripped, re.IGNORECASE):
        stripped = f"{stripped} LIMIT {max_rows}"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(stripped)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
    except Exception as e:
        return f"Query failed: {e}"
    finally:
        conn.close()

    if not rows:
        return "Query returned no rows."

    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the data quality knowledge base (postmortems, validation "
                "rules, data contracts) for documented incidents and policies. "
                "Use this for 'what happened', 'why do we do X', or 'what's our "
                "policy on Y' questions. Does NOT know about current/live data — "
                "only what's been written up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_warehouse",
            "description": (
                "Run a read-only SQL SELECT query against the warehouse to answer "
                "questions about current/actual data — row counts, specific values, "
                "current state. Does NOT know why something happened or what policy "
                "applies — only what the data currently says. Only SELECT statements "
                "are allowed; anything else is rejected.\n\n"
                "Known tables and columns:\n"
                "- bronze.loyalty_transactions(transaction_id, loyalty_id, order_id, "
                "points_earned, points_redeemed, transaction_type, transaction_timestamp)\n"
                "- bronze.loyalty_accounts(loyalty_id, customer_unique_id, tier, "
                "points_balance, enrollment_date, last_activity_date, email_opt_in)\n"
                "- bronze.olist_orders_dataset(order_id, customer_id, order_status, "
                "order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, "
                "order_delivered_customer_date, order_estimated_delivery_date)\n"
                "- bronze.olist_customers_dataset(customer_id, customer_unique_id, "
                "customer_zip_code_prefix, customer_city, customer_state)\n"
                "- bronze.olist_order_items_dataset, bronze.olist_order_payments_dataset, "
                "bronze.olist_order_reviews_dataset, bronze.olist_products_dataset, "
                "bronze.olist_sellers_dataset, bronze.product_category_name_translation "
                "— standard Olist e-commerce tables (products, sellers, payments, reviews).\n"
                "- retrieval.kb_chunks(chunk_id, doc_id, doc_type, title, section, "
                "chunk_text, source_file, metadata, embedding, updated_at) — do not "
                "query this directly, use search_knowledge_base instead.\n\n"
                "Never invent a table or column name not listed here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A read-only SELECT query."},
                },
                "required": ["sql"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_knowledge_base": lambda tool_input: search_knowledge_base(tool_input["query"]),
    "query_warehouse": lambda tool_input: query_warehouse(tool_input["sql"]),
}
