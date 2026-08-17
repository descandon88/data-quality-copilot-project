"""
Agent-level evaluation: runs every ground truth question through the real
agent loop (agent/copilot.py's ask()) and scores two independent things —

1. Tool-routing accuracy: did the agent call the tool(s) the question
   actually needs? Each ground truth row carries an expected_tools list
   (e.g. a "kb" question expects only search_knowledge_base; a "combined"
   question expects both). Scored as precision/recall over the set of tool
   names called, plus a stricter exact-match flag — precision catches the
   agent calling a tool it didn't need, recall catches it skipping one it
   did.
2. Answer quality via LLM-as-judge: a second Groq call classifies the final
   answer as RELEVANT / PARTLY_RELEVANT / NON_RELEVANT given the question.
   This is a coarse signal, not a substitute for reading transcripts, but
   it's cheap to run on every question and catches gross failures (routing
   was correct but the answer still didn't address the question).

The judge runs on llama-3.1-8b-instant, not the generation model
(llama-3.3-70b-versatile) — see evaluation/settings.py's JUDGE_MODEL_NAME
comment. Every ask() call still runs on 70b, though, since that's the model
actually being evaluated, and 70b has a hard 100k-tokens/day cap on this
project's free Groq tier that the rest of the day's testing also draws
from. If that cap is hit mid-run, this script stops immediately instead of
looping through the remaining questions and re-hitting the same wall for
each one — see the RateLimitError handling in main().

Runs the live agent loop against the real Groq API and real Postgres, so
this is not free or instant — use --limit to smoke-test on a handful of
questions before running the full ground truth set.

Run inside the app container:
    docker compose exec app python evaluation/agent_eval.py --limit 10
    docker compose exec app python evaluation/agent_eval.py --categories warehouse,combined
    docker compose exec app python evaluation/agent_eval.py --budget 20000
    docker compose exec app python evaluation/agent_eval.py            # full set
"""
import argparse
import json
import re

from openai import RateLimitError

from agent.copilot import ask, get_client
from evaluation.settings import AGENT_RESULTS_PATH, GROUND_TRUTH_PATH, JUDGE_MODEL_NAME

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's answer to an \
internal data-engineering question about a retail loyalty platform.

Question: {question}

Generated answer: {answer}

Classify the generated answer's relevance to the question as exactly one of:
- "RELEVANT": the answer correctly and directly addresses what the question \
asked. A short, well-cited answer (e.g. citing a doc id like "PM-001") is \
still RELEVANT if it is factually correct and gives the specific \
information the question asked for — brevity and citing a source instead \
of restating its contents are NOT flaws.
- "PARTLY_RELEVANT": the answer addresses the question but is missing a \
specific piece of information the question actually asked for (a number, a \
named cause, a concrete mechanism), is vague where the question needs a \
concrete answer, or hedges unnecessarily.
- "NON_RELEVANT": the answer does not address the question, refuses without \
justification, or contradicts information in the question itself.

Judge only whether the specific information requested is present and \
correct — not writing style, length, or whether it elaborates beyond what \
was asked.

Return ONLY a JSON object, no other text: \
{{"relevance": "...", "explanation": "one brief sentence"}}
"""

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_ground_truth(limit=None, categories=None):
    rows = []
    with open(GROUND_TRUTH_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    return rows[:limit] if limit else rows


def judge_answer(client, question, answer):
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    response = client.chat.completions.create(
        model=JUDGE_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        # The expected output is a one-line JSON object with two short
        # fields — 150 tokens is generous headroom, not a real constraint,
        # but bounds worst-case cost the same way agent/copilot.py's
        # MAX_ANSWER_TOKENS does for the model actually being measured.
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(raw)
        if not match:
            return {"relevance": "UNPARSEABLE", "explanation": raw[:200]}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"relevance": "UNPARSEABLE", "explanation": raw[:200]}
    return parsed


def is_daily_quota_exhausted(e):
    """True for Groq's specific 'tokens per day' cap, as opposed to a
    per-minute rate limit or any other RateLimitError — those are worth
    retrying, this one isn't (it won't clear on the timescale of this
    script running). Checked by message content, not just exception type,
    since RateLimitError covers several distinct Groq limit types."""
    return isinstance(e, RateLimitError) and "tokens per day" in str(e).lower()


def routing_scores(expected_tools, actual_tools):
    expected, actual = set(expected_tools), set(actual_tools)
    exact_match = expected == actual
    intersection = expected & actual
    precision = len(intersection) / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = len(intersection) / len(expected) if expected else (1.0 if not actual else 0.0)
    return exact_match, precision, recall


def load_done_results():
    """Rows from a prior run that don't need re-doing: keyed by gt-id, kept
    only if ask() itself succeeded (no top-level "error" key) — those are
    the ones that actually burned quota productively. ask()-level failures
    (daily quota, exhausted retries) are deliberately re-attempted rather
    than skipped, since re-running is exactly how you'd recover from them
    once quota resets. A JUDGE_ERROR row still counts as done: ask()
    succeeded and consumed 70b quota, and re-running would burn that again
    just to retry the separate, cheaper judge call."""
    if not AGENT_RESULTS_PATH.exists():
        return {}
    done = {}
    with open(AGENT_RESULTS_PATH) as f:
        for line in f:
            row = json.loads(line)
            if "error" not in row:
                done[row["id"]] = row
    return done


def load_all_previous_results():
    """Every row currently on disk, error or not — used ONLY as the final
    merge fallback (see main()), never for computing `todo`. load_done_results()
    deliberately excludes error rows so they get retried; that's correct for
    building this run's queue, but if the final merge also fell back to that
    same dict, a previously-errored row whose category is outside this run's
    --categories filter would be in neither `results` (out of scope) nor
    `done` (it's an error) and would silently disappear from the output
    file. Real bug, not hypothetical: a warehouse-only run once erased a
    pre-existing kb error row this way (row count stayed at 16, but the old
    kb error was replaced by the new warehouse one) — see SESSION_HANDOFF.md."""
    if not AGENT_RESULTS_PATH.exists():
        return {}
    all_rows = {}
    with open(AGENT_RESULTS_PATH) as f:
        for line in f:
            row = json.loads(line)
            all_rows[row["id"]] = row
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N (post-filter) ground truth questions.")
    parser.add_argument("--categories", type=str, default=None,
                         help="Comma-separated subset of kb,warehouse,combined (default: all). "
                              "Ground truth is ordered kb (36 rows) then warehouse (6) then "
                              "combined (5), so a plain --limit run only ever reaches kb rows "
                              "until all 36 are done — use --categories warehouse,combined to "
                              "prioritize the routing-interesting rows instead.")
    parser.add_argument("--budget", type=int, default=None,
                         help="Stop after this run has spent roughly this many "
                              "llama-3.3-70b-versatile tokens (self-imposed, checked between "
                              "questions), instead of running until Groq's own 100k/day cap "
                              "errors out. Leaves headroom for other same-day usage (ad hoc "
                              "agent/copilot.py queries, evaluation/llm_eval_compare.py, etc). "
                              "Undercounts slightly on questions that needed a tool_use_failed "
                              "retry — see agent/copilot.py's ask() docstring.")
    args = parser.parse_args()

    categories = {c.strip() for c in args.categories.split(",")} if args.categories else None
    # full_ground_truth (unfiltered, stable id order) is kept separate from
    # the filtered/limited `ground_truth` used to pick this run's scope —
    # the final write-out below iterates over the FULL set, not the
    # filtered one, so a --categories run never drops previously-completed
    # rows outside that filter. This is the exact bug llm_eval_compare.py
    # hit once already (see SESSION_HANDOFF.md) — applying the same fix
    # here up front instead of rediscovering it.
    full_ground_truth = load_ground_truth()
    ground_truth = load_ground_truth(args.limit, categories)
    if not ground_truth:
        print(f"No rows in {GROUND_TRUTH_PATH} — run evaluation/generate_ground_truth.py first, "
              f"or check --categories matches kb/warehouse/combined.")
        return

    done = load_done_results()
    all_previous = load_all_previous_results()
    todo = [row for row in ground_truth if row["id"] not in done]
    print(f"Running agent eval over {len(ground_truth)} questions"
          f"{f' (categories={args.categories})' if args.categories else ''}"
          f"{f' (limited)' if args.limit else ''}"
          f" — {len(done)} already completed in {AGENT_RESULTS_PATH}, "
          f"{len(todo)} left to run..."
          f"{f' (self-imposed budget: {args.budget} tokens)' if args.budget else ''}\n")

    judge_client = get_client()

    results = []
    total_tokens_this_run = 0
    for i, row in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {row['id']} ({row['category']}): {row['question'][:80]}")
        try:
            answer, tools_called, tokens_used = ask(row["question"], verbose=False, return_trace=True)
        except Exception as e:
            print(f"    [error] {e}")
            results.append({
                **row, "answer": None, "tools_called": [], "error": str(e),
                "exact_match": False, "precision": 0.0, "recall": 0.0,
                "judge_relevance": "ERROR", "tokens_used": 0,
            })
            if is_daily_quota_exhausted(e):
                print(f"\n[stopped] Hit llama-3.3-70b-versatile's daily token quota after "
                      f"{i}/{len(todo)} remaining questions. This is a hard cap on Groq's free "
                      f"tier, not a bug here, and it won't clear on a short retry — stopping "
                      f"now instead of re-hitting the same wall for every remaining question. "
                      f"Results collected so far are still written out below. Re-run later "
                      f"once the quota has recovered (it behaves like a rolling window rather "
                      f"than a fixed midnight reset, based on how slowly 'Used' moves between "
                      f"runs on the same day).")
                break
            continue

        total_tokens_this_run += tokens_used
        exact_match, precision, recall = routing_scores(row["expected_tools"], tools_called)

        # Judge runs on a separate model/quota (see module docstring), but
        # it's still a live API call that can fail independently of the
        # agent call above — don't let a judge-side failure discard the
        # routing result we already have, and don't let it crash the whole
        # run and lose every prior result.
        try:
            judge = judge_answer(judge_client, row["question"], answer)
        except Exception as e:
            print(f"    [judge error] {e}")
            judge = {"relevance": "JUDGE_ERROR", "explanation": str(e)}
            if is_daily_quota_exhausted(e):
                results.append({
                    **row, "answer": answer, "tools_called": tools_called,
                    "exact_match": exact_match, "precision": precision, "recall": recall,
                    "judge_relevance": "JUDGE_ERROR", "judge_explanation": str(e),
                    "tokens_used": tokens_used,
                })
                print(f"\n[stopped] Hit the judge model's daily token quota after "
                      f"{i}/{len(todo)} remaining questions. Routing results collected so far "
                      f"(including this one) are still valid and written out below.")
                break

        print(f"    tools_called={tools_called} exact_match={exact_match} "
              f"judge={judge.get('relevance')} tokens_used={tokens_used} "
              f"(run total: {total_tokens_this_run})")

        results.append({
            **row,
            "answer": answer,
            "tools_called": tools_called,
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "tokens_used": tokens_used,
            "judge_relevance": judge.get("relevance"),
            "judge_explanation": judge.get("explanation"),
        })

        if args.budget and total_tokens_this_run >= args.budget:
            print(f"\n[stopped] Self-imposed --budget of {args.budget} tokens reached "
                  f"({total_tokens_this_run} spent) after {i}/{len(todo)} remaining questions — "
                  f"this is a deliberate early stop, not a Groq error, so there's real quota "
                  f"left for other work today. Re-run the same command (no flag needed to "
                  f"resume — checkpointing skips completed questions) whenever you want to "
                  f"spend more of today's budget on this.")
            break

    # Merge this run's rows with everything already on disk (all_previous —
    # error rows included), in ground-truth order — over the FULL 47-row
    # set, not the filtered/limited `ground_truth` used above to pick this
    # run's scope, so a --categories or --limit run never drops
    # previously-written rows outside that scope when the file is
    # rewritten. Deliberately all_previous here, NOT `done`: `done` excludes
    # error rows on purpose (so they get retried when in scope), but using
    # it as the merge fallback too meant an out-of-scope error row was in
    # neither `results` nor `done` and silently vanished from the output —
    # a real bug hit once already (see SESSION_HANDOFF.md).
    new_by_id = {row["id"]: row for row in results}
    merged = [new_by_id.get(row["id"], all_previous.get(row["id"])) for row in full_ground_truth]
    merged = [row for row in merged if row is not None]

    AGENT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENT_RESULTS_PATH, "w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    results = merged
    n = len(results)
    # "error" is only ever set on the ask()-level failure path (Groq
    # tool_use_failed after all retries, daily quota, etc.) — those rows
    # have no real tool-selection decision behind them (tools_called=[] is
    # an artifact of the failure, not the model choosing to call nothing).
    # Scoring them into routing precision/recall/exact_match conflates
    # "Groq's API failed this request" with "the model routed wrong",
    # which understates the model's actual routing judgment. Judge-quota
    # failures (JUDGE_ERROR) are different — ask() already succeeded and
    # produced a real tools_called, only the separate judge call failed —
    # so those rows stay in the routing denominator.
    routable = [r for r in results if "error" not in r]
    reliability_rate = len(routable) / n

    if routable:
        exact_match_rate = sum(r["exact_match"] for r in routable) / len(routable)
        mean_precision = sum(r["precision"] for r in routable) / len(routable)
        mean_recall = sum(r["recall"] for r in routable) / len(routable)
    else:
        exact_match_rate = mean_precision = mean_recall = 0.0

    relevance_counts = {}
    for r in results:
        relevance_counts[r["judge_relevance"]] = relevance_counts.get(r["judge_relevance"], 0) + 1

    total_tokens_all_rows = sum(r.get("tokens_used") or 0 for r in results)
    print(f"\n{'=' * 60}")
    print(f"TOKEN USAGE: {total_tokens_this_run} spent this run "
          f"(llama-3.3-70b-versatile; undercounts questions that needed a retry — see "
          f"ask()'s docstring) / {total_tokens_all_rows} total across all {n} rows on disk.")
    print("RELIABILITY")
    print(f"  completed without error: {len(routable)}/{n}  ({reliability_rate:.1%})")
    print(f"\nROUTING (tool selection vs. expected_tools, n={len(routable)} completed questions)")
    print(f"  exact match rate: {exact_match_rate:.3f}")
    print(f"  mean precision:   {mean_precision:.3f}  (didn't call unneeded tools)")
    print(f"  mean recall:      {mean_recall:.3f}  (called every tool it needed)")
    print(f"\nANSWER QUALITY (LLM-as-judge, all {n} rows incl. errors)")
    for label, count in sorted(relevance_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:16s} {count:3d}/{n}  ({count / n:.1%})")

    print("\nBy category:")
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)
    for cat, rows in by_category.items():
        cat_n = len(rows)
        cat_routable = [r for r in rows if "error" not in r]
        cat_exact = sum(r["exact_match"] for r in cat_routable) / len(cat_routable) if cat_routable else 0.0
        cat_relevant = sum(r["judge_relevance"] == "RELEVANT" for r in rows) / cat_n
        print(f"  {cat:12s} n={cat_n:3d}  reliability={len(cat_routable) / cat_n:.3f}  "
              f"exact_match={cat_exact:.3f}  relevant={cat_relevant:.3f}")

    print(f"\nPer-question results -> {AGENT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
