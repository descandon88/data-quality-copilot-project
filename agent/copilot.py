"""
Data Quality Incident Copilot — RAG generation + agent tool routing.

Given a question, the model decides (via function calling) whether it needs to
search the knowledge base (search_knowledge_base), query live data
(query_warehouse), both, or neither — then generates a final answer grounded
in whatever it retrieved, citing sources (PM-XXX/RULE-XXX doc ids for
knowledge base results, or the query itself for SQL results).

This is genuinely agentic routing, not a fixed pipeline bolted onto an LLM
for its own sake: "why do we hard-stop on duplicate earn events" only needs
the knowledge base; "how many duplicate earn rows exist right now" only
needs SQL; "are we currently violating RULE-001" needs both. The model
decides which tool(s) a given question actually requires, instead of always
running the same steps regardless of what was asked.

Uses Groq's Chat Completions API (https://console.groq.com/docs/tool-use),
via the OpenAI-compatible `openai` Python client pointed at Groq's base_url.
Switched from the Responses API (client.responses.create): Groq's own
tool-use documentation exclusively demonstrates function calling through
Chat Completions, and in practice the Responses API path produced malformed
`<function=...>` text instead of real structured tool calls — a known
failure mode, independent of which model was used. Chat Completions is
Groq's mature, documented path for this.

Run inside the app container:
    docker compose exec app python agent/copilot.py "why would loyalty point balances be wrong"
"""
import json
import os
import sys

from openai import BadRequestError, OpenAI

from agent.settings import MAX_ANSWER_TOKENS, MAX_TOOL_ROUNDS, MODEL_NAME, SYSTEM_PROMPT
from agent.tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set in .env")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def ask(question: str, verbose: bool = True, return_trace: bool = False):
    """Runs the agent loop for a single question.

    By default returns just the final answer string (unchanged CLI
    behavior). Pass return_trace=True to additionally get back the ordered
    list of tool names actually called (evaluation/agent_eval.py uses this
    to score tool-routing accuracy against a question's expected_tools
    without having to scrape stdout) and the total Groq tokens consumed
    across every call this question made (every tool-call round plus the
    final answer round) — the eval harnesses sum this across a run to show
    running consumption against the free tier's 100k-tokens/day cap instead
    of only finding out it's exhausted when a call fails.
    """
    client = get_client()
    tools_called = []
    total_tokens = 0

    def finish(answer):
        return (answer, tools_called, total_tokens) if return_trace else answer

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        # Groq's llama-3.3-70b-versatile has a known, currently-open bug
        # (confirmed on Groq's own community forum) where it occasionally
        # emits malformed <function=...> text instead of a structured tool
        # call, raising openai.BadRequestError(code="tool_use_failed"). This
        # is model/backend flakiness, not a bug in this loop.
        # parallel_tool_calls=False forces the single-call decoding path
        # (which is what our system prompt's "search first, then query"
        # sequencing already assumes).
        #
        # 3 attempts (2 retries), not 1: a 47-question Phase 7 eval run
        # measured 3/14 questions still failing after a single retry — if
        # failures are roughly independent per attempt, that implies a
        # ~45-50% per-call failure rate right now, meaning 2 attempts only
        # gets to ~20-25% still failing. A 3rd attempt should bring that
        # down to single digits. This is absorbing a currently-elevated
        # Groq-side failure rate, not compensating for a bug in this loop.
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    parallel_tool_calls=False,
                    max_tokens=MAX_ANSWER_TOKENS,
                )
                break
            except BadRequestError as e:
                if getattr(e, "code", None) == "tool_use_failed" and attempt < 2:
                    if verbose:
                        print(f"  [retry] Groq tool_use_failed, retrying (attempt {attempt + 2}/3)...")
                    continue
                raise
        # Known undercount, stated plainly rather than glossed over: a
        # failed tool_use_failed attempt is a 400 error from Groq's API, so
        # the openai client's BadRequestError doesn't carry a usable
        # usage/token count for that attempt — only the winning attempt's
        # response.usage is countable here. Retried questions really did
        # spend more tokens than this total reflects; there's no client-side
        # way to recover that number from the SDK's exception object.
        # response.usage can also be None on some mock/edge-case responses;
        # guard rather than let token tracking crash a real answer.
        if response.usage:
            total_tokens += response.usage.total_tokens
        message = response.choices[0].message

        if not message.tool_calls:
            return finish(message.content)

        # Explicit dict, not the raw SDK message object — matches Groq's
        # documented message shape exactly and avoids any ambiguity around
        # how the client would serialize a pydantic object back into the
        # messages list.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        })

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            tools_called.append(tool_call.function.name)
            if verbose:
                print(f"  [tool call] {tool_call.function.name}({args})")
            tool_fn = TOOL_FUNCTIONS.get(tool_call.function.name)
            result = tool_fn(args) if tool_fn else f"Unknown tool: {tool_call.function.name}"
            if verbose:
                print(f"  [tool result] {result[:300]}{'...' if len(result) > 300 else ''}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": result,
            })

    return finish("Reached max tool-call rounds without a final answer — the "
                  "question may be too complex or ambiguous for the current tool set.")


def main():
    question = " ".join(sys.argv[1:]) or "why would loyalty point balances be wrong"
    print(f"Question: {question!r}\n")
    answer = ask(question)
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
