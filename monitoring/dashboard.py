"""
Phase 9 — monitoring dashboard.

Two halves, on purpose (see CLAUDE.md's rubric-mapping notes): the course
rubric's monitoring criterion grades the RAG app's own performance (user
feedback, cost/latency), not general data-quality monitoring — a
RULE-00X-violations-only dashboard would satisfy a *different*,
self-invented question ("is the business data healthy") but not the one
actually graded ("is the RAG app performing well / are users satisfied").
So this page shows both: live chat performance from monitoring/db.py's
conversations/feedback tables (populated by app/app.py), and the
violation-tracking view over dbt's silver.rule_violations mart (this
project's own differentiator, still worth showing alongside the required
half, not instead of it).

Run on a separate port from the chat UI (app/app.py uses 8501):
    docker compose exec app streamlit run monitoring/dashboard.py --server.port 8502 --server.address 0.0.0.0
Then open http://localhost:8502
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.db import get_conversations_df, get_stats, get_violation_summary, init_db  # noqa: E402

st.set_page_config(page_title="Copilot Monitoring", page_icon="📊", layout="wide")
init_db()  # idempotent — safe on every load, no separate bootstrap step

st.title("📊 Data Quality Copilot — Monitoring")

# ---------------------------------------------------------------- RAG performance
st.header("RAG app performance")
st.caption("Populated by real chat turns in the UI (app/app.py) — not eval runs.")

stats = get_stats()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Conversations logged", stats.total_conversations)
c2.metric("Avg response time", f"{stats.avg_response_time:.2f}s" if stats.avg_response_time else "—")
c3.metric("Avg tokens/answer", f"{stats.avg_tokens:.0f}" if stats.avg_tokens is not None else "—")
c4.metric("Error rate", f"{stats.error_rate:.0%}" if stats.error_rate is not None else "—")
total_feedback = stats.thumbs_up + stats.thumbs_down
c5.metric(
    "Thumbs-up rate",
    f"{stats.thumbs_up / total_feedback:.0%}" if total_feedback else "no feedback yet",
)

conv_df = get_conversations_df()
if conv_df.empty:
    st.info("No conversations logged yet — ask something in the chat UI (`app/app.py`) first.")
else:
    chart_df = conv_df.sort_values("created_at").set_index("created_at")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Response time over time")
        st.line_chart(chart_df["response_time_seconds"])
    with col_b:
        st.subheader("Tokens used over time")
        st.line_chart(chart_df["tokens_used"])

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Tool usage breakdown")
        tool_counts = (
            conv_df["tools_called"]
            .apply(lambda t: t if t else ["(none — answered directly)"])
            .explode()
            .value_counts()
        )
        st.bar_chart(tool_counts)
    with col_d:
        st.subheader("User feedback")
        if total_feedback == 0:
            st.caption("No feedback submitted yet — thumbs up/down in the chat UI to populate this.")
        else:
            feedback_counts = conv_df["feedback_score"].dropna().map({1: "👍 up", -1: "👎 down"}).value_counts()
            st.bar_chart(feedback_counts)

    st.subheader("🧑‍⚖️ Judge relevance (opt-in, per-message)")
    st.caption(
        "Only counts messages someone actually clicked \"Judge this answer\" on in the "
        "chat UI — not every conversation. See Phase 7's agent_eval.py for the "
        "systematic, full-ground-truth version of this same judge."
    )
    judged = conv_df.dropna(subset=["judge_relevance"])
    if judged.empty:
        st.caption("No answers judged yet.")
    else:
        st.caption(f"{len(judged)}/{len(conv_df)} recent conversations judged so far.")
        st.bar_chart(judged["judge_relevance"].value_counts())

    st.subheader("Recent conversations")
    for _, row in conv_df.head(15).iterrows():
        tools = ", ".join(row["tools_called"]) if row["tools_called"] else "none"
        fb = {1: "👍", -1: "👎"}.get(row["feedback_score"], "—")
        status = "⚠️ error" if row["had_error"] else "ok"
        st.markdown(f"**{row['question']}**")
        answer_preview = row["answer"][:200] + ("..." if len(row["answer"]) > 200 else "")
        st.caption(answer_preview)
        st.caption(
            f"tools: {tools} · {row['response_time_seconds']:.2f}s · "
            f"{row['tokens_used'] if row['tokens_used'] is not None else '—'} tokens · "
            f"{status} · feedback: {fb} · "
            f"judge: {row['judge_relevance'] if pd.notna(row['judge_relevance']) else 'not judged'}"
        )
        st.divider()

# ---------------------------------------------------------------- data quality
st.header("Data quality: live RULE-00X violations")
st.caption(
    "Read directly from dbt's silver.rule_violations mart (dbt_project/) — "
    "run `dbt run --project-dir dbt_project` to refresh after new data lands."
)
viol_df = get_violation_summary()
if viol_df.empty:
    st.info("No rows in `silver.rule_violations` yet — run `dbt run --project-dir dbt_project` first.")
else:
    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Violations by rule")
        st.bar_chart(viol_df.groupby("rule_id")["violation_count"].sum())
    with col_f:
        st.subheader("Violations by enforcement level")
        st.bar_chart(viol_df.groupby("enforcement")["violation_count"].sum())

    st.subheader("Detail")
    st.dataframe(viol_df, use_container_width=True)
