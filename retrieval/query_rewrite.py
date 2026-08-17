"""
Query rewriting — an optional preprocessing step that reformulates a
user's natural-language question into a query better suited to this
knowledge base's actual phrasing, before hybrid_search.py's vector/BM25
retrieval ever runs.

Why this helps: a vague or conversational question ("why would loyalty
point balances be wrong") shares fewer exact terms with how the knowledge
base actually talks about the problem (RULE-004's own language is
"points_balance < 0", "hard-stop enforcement") than a rewritten version
that surfaces those terms explicitly — which matters most for BM25's
side of the hybrid pipeline, since exact-term overlap is precisely what
BM25 scores and pure paraphrase can't help it find. See
evaluation/retrieval_eval.py's "hybrid-rrf + rerank + query-rewrite"
strategy for a measured before/after, not just an assumption that this
helps.

Deliberately kept separate from hybrid_search.py itself (not baked into
vector_search()/bm25_search()) so hybrid_search.py stays a pure,
LLM-free retrieval module — tests/test_hybrid_search.py exercises it with
no network dependency, and that should keep being true.

Fails open by design: if the Groq call errors for any reason (daily
quota, transient network issue, malformed response), rewrite_query()
returns the original question unchanged rather than raising — a failed
rewrite should never be the reason a search comes back empty. Runs on
QUERY_REWRITE_MODEL_NAME (llama-3.1-8b-instant), a separate free-tier
quota bucket from the generation model — see retrieval/settings.py's
comment.

Run inside the app container:
    docker compose exec app python retrieval/query_rewrite.py "why would loyalty point balances be wrong"
"""
import os
import sys

from openai import OpenAI

from retrieval.settings import QUERY_REWRITE_MODEL_NAME

REWRITE_PROMPT = """You are rewriting a user's question into a better search query for a \
data-quality knowledge base about a retail loyalty platform. The knowledge base contains \
incident postmortems (PM-XXX), validation rules (RULE-XXX), and data contracts \
(CONTRACT-XXX), covering: duplicate/non-idempotent earn events, null loyalty_id join keys, \
orphaned loyalty transactions, negative points balances, and stale tier assignments.

Rewrite the question below into a concise, keyword-rich search query that surfaces the most \
relevant documents — expand vague or conversational phrasing into the specific technical terms \
these documents actually use (e.g. "points_balance", "loyalty_id", "earn event", "enforcement"). \
Do not invent facts, numbers, or specific rule/incident IDs that aren't implied by the question \
itself, and do not answer the question — only rewrite it.

Return ONLY the rewritten query, one line, no quotes, no explanation.

Question: {question}

Rewritten query:"""


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set in .env")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def rewrite_query(question: str, client=None, verbose: bool = False) -> str:
    """Returns a rewritten search query, or the original `question`
    unchanged if the rewrite call fails for any reason (see module
    docstring — fail-open is deliberate). `client` is injectable for
    testing (see tests/test_query_rewrite.py) without needing a real
    GROQ_API_KEY."""
    try:
        client = client or _get_client()
        response = client.chat.completions.create(
            model=QUERY_REWRITE_MODEL_NAME,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
            temperature=0,
            # A rewritten query is expected to be one short line — this
            # bounds worst-case cost the same way agent/settings.py's
            # MAX_ANSWER_TOKENS and agent_eval.py's judge max_tokens do.
            max_tokens=60,
        )
        rewritten = (response.choices[0].message.content or "").strip().strip('"')
        return rewritten if rewritten else question
    except Exception as e:  # noqa: BLE001 — fail open, never block retrieval over this
        if verbose:
            print(f"  [query_rewrite] falling back to original query: {e}")
        return question


def main():
    question = " ".join(sys.argv[1:]) or "why would loyalty point balances be wrong"
    print(f"Original:  {question!r}")
    rewritten = rewrite_query(question, verbose=True)
    print(f"Rewritten: {rewritten!r}")


if __name__ == "__main__":
    main()
