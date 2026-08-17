"""
Second generation approach for Phase 7's LLM evaluation: a naive, fixed
always-retrieve-then-answer RAG baseline, compared against the real agent
(agent/copilot.py's ask()) in llm_eval_compare.py.

Why this baseline specifically, not a different prompt/temperature variant:
agent/copilot.py's own docstring makes an explicit claim — "this is genuinely
agentic routing, not a fixed pipeline bolted onto an LLM for its own sake."
That claim is untested without something to compare against. This baseline
is the most direct test of it: always call search_knowledge_base (never
query_warehouse, never a routing decision), stuff whatever comes back into
the prompt, answer in one shot. Same underlying model (agent.settings.MODEL_NAME)
as the real agent, so the comparison isolates the orchestration strategy
(fixed single-tool pipeline vs. model-decided multi-tool routing), not model
quality.

Expected result, stated up front so it's a real prediction and not
after-the-fact rationalization: near-parity on "kb" category questions
(both approaches have the same retrieval available), a clear gap opening up
on "warehouse" and "combined" questions, where this baseline structurally
cannot produce a row count or current-state number — it has no SQL access
at all, so the only honest answer it can give is "the knowledge base doesn't
contain live data for that."

Not meant to be run standalone — imported by evaluation/llm_eval_compare.py.
"""
from agent.copilot import get_client
from agent.settings import MAX_ANSWER_TOKENS, MODEL_NAME
from agent.tools import search_knowledge_base

NAIVE_SYSTEM_PROMPT = """You are a search assistant for a retail loyalty \
platform's internal data quality knowledge base (postmortems, validation \
rules, data contracts).

You will be given a question and a fixed set of knowledge-base search \
results already retrieved for you. Answer using ONLY that retrieved \
context — you have no other tools and no access to live/current warehouse \
data (row counts, current balances, current violations, etc.). If the \
retrieved context doesn't contain the current/live number or fact the \
question asks for, say so plainly instead of guessing — do not fabricate a \
number.

Cite the doc id (e.g. "per PM-001") for anything you state from the \
retrieved context. Keep answers grounded and concise.
"""

NAIVE_USER_PROMPT = """Question: {question}

Retrieved knowledge base context:
{context}
"""


def naive_rag_answer(question: str, client=None):
    """Fixed pipeline: always retrieve top-5 via the same hybrid search +
    rerank used by search_knowledge_base, then one non-tool-calling
    completion. No routing decision, no query_warehouse access — by
    construction, not by the model choosing not to call it.

    Returns (answer, {"tokens_used": n}) — the extra-fields dict shape
    evaluation/llm_eval_compare.py's run_approach() already expects (same
    convention as agent/copilot.py's ask() returning tools_called), so the
    comparison script can sum real Groq spend per approach instead of only
    finding out the daily quota is gone when a call fails."""
    client = client or get_client()
    context = search_knowledge_base(question, k=5)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
            {"role": "user", "content": NAIVE_USER_PROMPT.format(question=question, context=context)},
        ],
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    tokens_used = response.usage.total_tokens if response.usage else 0
    return response.choices[0].message.content, {"tokens_used": tokens_used}
