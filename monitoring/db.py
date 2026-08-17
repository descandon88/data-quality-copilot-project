"""
Phase 9 (monitoring) — Postgres schema + queries for live-chat logging,
user feedback, and the aggregate stats monitoring/dashboard.py renders.

Deliberately separate from bronze/staging/silver (dlt/dbt's medallion
schemas — see CLAUDE.md) and from evaluation/ (Phase 7's offline harness
writes its own JSONL files under data/processed/, not this DB): this
`monitoring` schema holds only *live* traffic from app/app.py, so the
dashboard reflects real usage, not eval-run noise. Reuses
common/postgresql.py's get_connection() — no second credentials source.

init_db() is idempotent (CREATE SCHEMA/TABLE IF NOT EXISTS) and is called
automatically from both app/app.py and monitoring/dashboard.py on startup
— there's no separate manual bootstrap step to remember or forget.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from common.postgresql import get_connection


def init_db():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS monitoring")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS monitoring.conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tools_called TEXT[] NOT NULL DEFAULT '{}',
                    tokens_used INTEGER,
                    response_time_seconds DOUBLE PRECISION NOT NULL,
                    had_error BOOLEAN NOT NULL DEFAULT false,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            # source distinguishes a real user's thumbs up/down from an
            # opt-in LLM-judge re-score (see save_feedback()'s docstring —
            # judging is a per-message button in app/app.py, not automatic,
            # specifically so it doesn't spend quota on every turn). score
            # is only set for source='user'; relevance/explanation only for
            # source='judge' — enforced for new installs by the CHECK
            # below.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS monitoring.feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL
                        REFERENCES monitoring.conversations(id),
                    source TEXT NOT NULL CHECK (source IN ('user', 'judge')),
                    score SMALLINT CHECK (score IN (-1, 1)),
                    relevance TEXT,
                    explanation TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT feedback_source_shape CHECK (
                        (source = 'user' AND score IS NOT NULL AND relevance IS NULL)
                        OR (source = 'judge' AND relevance IS NOT NULL AND score IS NULL)
                    )
                )
            """)
            # Migration path for a table created by an earlier version of
            # this file (source/relevance/explanation didn't exist yet,
            # score was NOT NULL) — ADD COLUMN IF NOT EXISTS and DROP NOT
            # NULL are both no-ops on a fresh install where CREATE TABLE
            # above already has the current shape. Not attempting to
            # retroactively add feedback_source_shape on an old table —
            # not worth the risk of failing on data written by an earlier
            # session for a demo project; new rows are still validated by
            # save_feedback()'s own checks either way.
            cur.execute("ALTER TABLE monitoring.feedback ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'user'")
            cur.execute("ALTER TABLE monitoring.feedback ALTER COLUMN score DROP NOT NULL")
            cur.execute("ALTER TABLE monitoring.feedback ADD COLUMN IF NOT EXISTS relevance TEXT")
            cur.execute("ALTER TABLE monitoring.feedback ADD COLUMN IF NOT EXISTS explanation TEXT")
        conn.commit()
    finally:
        conn.close()


def save_conversation(
    question: str,
    answer: str,
    model: str,
    tools_called: list[str],
    tokens_used: int | None,
    response_time_seconds: float,
    had_error: bool = False,
) -> int:
    """Logs one live chat turn. Returns the new conversation id, used to
    attach feedback (thumbs up/down) submitted for this turn later."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO monitoring.conversations
                    (question, answer, model, tools_called, tokens_used,
                     response_time_seconds, had_error)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (question, answer, model, tools_called, tokens_used,
                 response_time_seconds, had_error),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id


def save_feedback(
    conversation_id: int,
    source: str,
    score: int | None = None,
    relevance: str | None = None,
    explanation: str | None = None,
):
    """Two independent kinds of feedback on the same conversation_id:

    - source="user": a real 👍/👎 from the person chatting — score is +1
      or -1, relevance/explanation must be None. Free (no API call).
    - source="judge": an LLM-as-judge re-score, triggered by the opt-in
      "Judge this answer" button in app/app.py (never automatic) —
      relevance is RELEVANT/PARTLY_RELEVANT/NON_RELEVANT/etc. (whatever
      evaluation/agent_eval.py's judge_answer() returned), score must be
      None. Judging runs on JUDGE_MODEL_NAME (llama-3.1-8b-instant), a
      separate Groq free-tier quota bucket from the generation model
      (llama-3.3-70b-versatile) — see evaluation/settings.py's
      JUDGE_MODEL_NAME comment — so clicking it doesn't compete with the
      quota that generation itself depends on, but it's still a real API
      call, hence opt-in rather than automatic on every turn.
    """
    if source not in ("user", "judge"):
        raise ValueError(f"source must be 'user' or 'judge', got {source!r}")
    if source == "user":
        if score not in (-1, 1) or relevance is not None or explanation is not None:
            raise ValueError("source='user' requires score in (-1, 1) and no relevance/explanation")
    else:
        if not relevance or score is not None:
            raise ValueError("source='judge' requires a relevance string and no score")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO monitoring.feedback
                    (conversation_id, source, score, relevance, explanation)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (conversation_id, source, score, relevance, explanation),
            )
        conn.commit()
    finally:
        conn.close()


@dataclass
class Stats:
    total_conversations: int
    avg_response_time: float | None
    avg_tokens: float | None
    error_rate: float | None
    thumbs_up: int
    thumbs_down: int


def get_stats() -> Stats:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    AVG(response_time_seconds),
                    AVG(tokens_used),
                    AVG(had_error::int)
                FROM monitoring.conversations
            """)
            total, avg_rt, avg_tok, err_rate = cur.fetchone()
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE score = 1),
                    COUNT(*) FILTER (WHERE score = -1)
                FROM monitoring.feedback
                WHERE source = 'user'
            """)
            up, down = cur.fetchone()
    finally:
        conn.close()
    return Stats(
        total_conversations=total or 0,
        avg_response_time=avg_rt,
        avg_tokens=avg_tok,
        error_rate=err_rate,
        thumbs_up=up or 0,
        thumbs_down=down or 0,
    )


def get_conversations_df(limit: int = 200) -> pd.DataFrame:
    """One row per conversation. Two separate LEFT JOINs, not one — a
    conversation can now have up to two feedback rows (one source='user'
    thumbs, one source='judge' re-score, see save_feedback()), and each is
    still at most one row per conversation per source, so this stays a
    1:0-or-1 join per side rather than fanning out."""
    conn = get_connection()
    try:
        df = pd.read_sql(
            """
            SELECT
                c.id, c.question, c.answer, c.tools_called, c.tokens_used,
                c.response_time_seconds, c.had_error, c.created_at,
                uf.score AS feedback_score,
                jf.relevance AS judge_relevance,
                jf.explanation AS judge_explanation
            FROM monitoring.conversations c
            LEFT JOIN monitoring.feedback uf
                ON uf.conversation_id = c.id AND uf.source = 'user'
            LEFT JOIN monitoring.feedback jf
                ON jf.conversation_id = c.id AND jf.source = 'judge'
            ORDER BY c.created_at DESC
            LIMIT %(limit)s
            """,
            conn,
            params={"limit": limit},
        )
    finally:
        conn.close()
    return df


def get_violation_summary() -> pd.DataFrame:
    """Reads dbt's silver.rule_violations mart directly (dbt_project/, see
    CLAUDE.md) — the data-quality-specific half of this dashboard,
    alongside the RAG-performance half above. Requires `dbt run
    --project-dir dbt_project` to have been run at least once; returns an
    empty DataFrame (not an error) if the table doesn't exist yet, so the
    dashboard can render a friendly message instead of crashing."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'silver' AND table_name = 'rule_violations'
                )
            """)
            (exists,) = cur.fetchone()
        if not exists:
            return pd.DataFrame(columns=["rule_id", "enforcement", "entity_type", "violation_count"])
        df = pd.read_sql(
            """
            SELECT rule_id, enforcement, entity_type, COUNT(*) AS violation_count
            FROM silver.rule_violations
            GROUP BY rule_id, enforcement, entity_type
            ORDER BY rule_id
            """,
            conn,
        )
    finally:
        conn.close()
    return df


if __name__ == "__main__":
    init_db()
    print("monitoring schema ready (conversations, feedback tables).")
