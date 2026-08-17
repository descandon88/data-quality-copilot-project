"""
Data Quality Incident Copilot — Streamlit UI (Phase 8).

Thin chat interface over agent/copilot.py's ask(). Deliberately does not
reimplement any retrieval, tool-routing, or generation logic here — this
file only renders the conversation and surfaces the agent's own tool-call
trace, so the UI stays honest about what the agent actually did on each
turn (which is the point of a "copilot", not a chatbot with a fixed
script).

Phase 9 addition: every real turn is logged to monitoring.conversations
(question, answer, tools called, tokens, response time, error flag) via
monitoring/db.py, and each assistant message gets thumbs up/down buttons
that write to monitoring.feedback — monitoring/dashboard.py reads both.
Eval runs (evaluation/agent_eval.py etc.) call agent.copilot.ask()
directly and never touch this logging path, so the dashboard reflects
real usage, not eval noise.

Each assistant message also gets an opt-in "Judge this answer" button —
NOT automatic on every turn, since that would spend Groq quota on every
chat message (exactly what this project's --budget flags elsewhere exist
to avoid). Reuses evaluation/agent_eval.py's judge_answer(), which runs on
JUDGE_MODEL_NAME (llama-3.1-8b-instant) — a separate free-tier quota
bucket from the generation model (llama-3.3-70b-versatile), so clicking it
doesn't compete with the quota generation itself needs.

Run inside the app container (see docker-compose.yml — port 8501 is
already mapped to the host for this):
    docker compose exec app streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
Then open http://localhost:8501
"""
import os
import sys
import time

import streamlit as st

# Scripts under app/ are invoked by Streamlit as a direct file path, which
# puts app/'s own directory on sys.path, not /app — same reason
# docker-compose.yml sets PYTHONPATH=/app for every other script in this
# project (see CLAUDE.md). Streamlit does honor PYTHONPATH from the
# container environment already, but this makes `streamlit run app/app.py`
# from a different cwd work too, without relying on that env var alone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.copilot import ask, get_client  # noqa: E402
from agent.settings import MODEL_NAME  # noqa: E402
from evaluation.agent_eval import judge_answer  # noqa: E402
from evaluation.settings import JUDGE_MODEL_NAME  # noqa: E402
from monitoring.db import init_db, save_conversation, save_feedback  # noqa: E402

EXAMPLE_QUESTIONS = [
    "Why would loyalty point balances be wrong?",
    "Are we currently violating RULE-001, and how many rows are affected?",
    "How many accounts are on the gold tier right now?",
    "What's our policy on orphaned loyalty transactions?",
]

TOOL_LABELS = {
    "search_knowledge_base": "📚 searched the knowledge base",
    "query_warehouse": "🗄️ queried the warehouse",
}

st.set_page_config(page_title="Data Quality Incident Copilot", page_icon="🔎", layout="centered")

st.title("🔎 Data Quality Incident Copilot")
st.caption(
    "Ask about past data-quality incidents, validation rules, or the live state of the "
    "loyalty warehouse. The agent decides on its own whether it needs the knowledge base, "
    "a live SQL query, both, or neither — watch the \"tools used\" note under each answer."
)

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY isn't set. Add it to `.env` at the repo root and restart the "
        "`app` container — see `.env.example`."
    )
    st.stop()

# Idempotent (CREATE SCHEMA/TABLE IF NOT EXISTS) — safe to call on every
# page load. Monitoring logging is a nice-to-have, not the chat's core
# function, so a Postgres hiccup here degrades to "no logging" with a
# one-time sidebar warning rather than breaking the chat.
if "monitoring_available" not in st.session_state:
    try:
        init_db()
        st.session_state.monitoring_available = True
    except Exception as e:  # noqa: BLE001
        st.session_state.monitoring_available = False
        st.session_state.monitoring_error = str(e)

with st.sidebar:
    st.subheader("Try a question")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    st.subheader("How this works")
    st.markdown(
        "- **search_knowledge_base** — documented postmortems, validation rules, "
        "and data contracts\n"
        "- **query_warehouse** — read-only SQL against the actual loyalty data\n\n"
        "The model picks whichever tool(s) a question actually needs, per-turn — "
        "this isn't a fixed retrieve-then-generate pipeline."
    )
    st.caption(f"Model: `{MODEL_NAME}` via Groq")
    st.caption(
        f"Each answer also has a 🧑‍⚖️ **Judge this answer** button — runs an "
        f"LLM-as-judge pass on `{JUDGE_MODEL_NAME}` (a separate quota from the model "
        f"above). Opt-in per message, not automatic, on purpose."
    )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if not st.session_state.get("monitoring_available", True):
        st.divider()
        st.warning(f"Monitoring logging is unavailable: {st.session_state.get('monitoring_error')}")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_tool_note(tools_called, tokens_used):
    if tools_called:
        labels = ", ".join(TOOL_LABELS.get(t, t) for t in tools_called)
        note = f"🛠️ {labels}"
    else:
        note = "💬 answered directly, no tools called"
    if tokens_used is not None:
        note += f" · {tokens_used} tokens"
    return note


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption(render_tool_note(msg.get("tools_called"), msg.get("tokens_used")))

            conversation_id = msg.get("conversation_id")
            if conversation_id is not None:
                fb_col, judge_col, _ = st.columns([2, 3, 5])

                with fb_col:
                    if msg.get("feedback"):
                        st.caption("👍 Thanks!" if msg["feedback"] == 1 else "👎 Thanks, noted.")
                    else:
                        up_col, down_col = st.columns(2)
                        if up_col.button("👍", key=f"up_{i}"):
                            try:
                                save_feedback(conversation_id, source="user", score=1)
                                msg["feedback"] = 1
                            except Exception as e:  # noqa: BLE001
                                st.caption(f"Couldn't save feedback: {e}")
                            st.rerun()
                        if down_col.button("👎", key=f"down_{i}"):
                            try:
                                save_feedback(conversation_id, source="user", score=-1)
                                msg["feedback"] = -1
                            except Exception as e:  # noqa: BLE001
                                st.caption(f"Couldn't save feedback: {e}")
                            st.rerun()

                with judge_col:
                    if msg.get("judge_result"):
                        st.caption(f"🧑‍⚖️ {msg['judge_result']['relevance']}")
                    else:
                        if st.button("🧑‍⚖️ Judge this answer", key=f"judge_{i}"):
                            with st.spinner(f"Asking {JUDGE_MODEL_NAME}..."):
                                try:
                                    judge_client = get_client()
                                    result = judge_answer(judge_client, msg["question"], msg["content"])
                                    save_feedback(
                                        conversation_id,
                                        source="judge",
                                        relevance=result.get("relevance", "UNPARSEABLE"),
                                        explanation=result.get("explanation", ""),
                                    )
                                    msg["judge_result"] = result
                                except Exception as e:  # noqa: BLE001
                                    st.caption(f"Judge call failed: {e}")
                            st.rerun()

                if msg.get("judge_result", {}).get("explanation"):
                    st.caption(f"🧑‍⚖️ *{msg['judge_result']['explanation']}*")

pending = st.session_state.pop("pending_question", None)
typed = st.chat_input("Ask about an incident, a rule, or the current warehouse state...")
question = pending or typed

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        had_error = False
        start = time.perf_counter()
        with st.spinner("Thinking..."):
            try:
                answer, tools_called, tokens_used = ask(question, verbose=False, return_trace=True)
            except Exception as e:  # noqa: BLE001 — surface any Groq/tool error to the user, don't crash the app
                had_error = True
                answer = (
                    f"Something went wrong answering that: {e}\n\n"
                    "This is usually a transient Groq API issue (rate limit or a malformed "
                    "tool call) — try again in a moment."
                )
                tools_called, tokens_used = [], None
        response_time = time.perf_counter() - start
        st.markdown(answer)
        st.caption(render_tool_note(tools_called, tokens_used))

    conversation_id = None
    if st.session_state.get("monitoring_available"):
        try:
            conversation_id = save_conversation(
                question=question,
                answer=answer,
                model=MODEL_NAME,
                tools_called=tools_called,
                tokens_used=tokens_used,
                response_time_seconds=response_time,
                had_error=had_error,
            )
        except Exception as e:  # noqa: BLE001
            st.caption(f"(couldn't log this turn to monitoring: {e})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "question": question,
        "tools_called": tools_called,
        "tokens_used": tokens_used,
        "conversation_id": conversation_id,
    })
    st.rerun()
