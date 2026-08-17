"""
Phase 7's "LLM evaluation" comparison: runs every ground truth question
through TWO generation approaches and judges both, so there's an actual
documented comparison driving which approach is used — not just one
approach evaluated in isolation.

Approach A — "agentic" (production): agent/copilot.py's ask(), the real
tool-routing loop. Decides per-question whether to call
search_knowledge_base, query_warehouse, both, or neither.

Approach B — "naive_rag" (baseline): evaluation/baseline_naive_rag.py's
naive_rag_answer(). Always retrieves via search_knowledge_base only, never
calls query_warehouse, no routing decision — a fixed pipeline. Same
underlying model (agent.settings.MODEL_NAME) as the agentic approach, so the
comparison isolates the orchestration strategy, not model quality.

Both answers are judged with the same JUDGE_PROMPT/judge model as
agent_eval.py (imported from there, not redefined) for an apples-to-apples
relevance classification.

Cost note: both approaches call agent.settings.MODEL_NAME
(llama-3.3-70b-versatile), so this draws from the SAME 100k-tokens/day quota
as agent_eval.py, at roughly 2x the cost per question (two generations
instead of one). Ground truth is ordered kb (gt-0001..0036), then warehouse
(gt-0037..0042), then combined (gt-0043..0047) — see
generate_ground_truth.py. Running warehouse+combined first (11 questions,
`--categories warehouse,combined`) is the cheapest way to see the sharpest
contrast, since that's structurally where the naive baseline can't reach
live data at all; kb questions (`--categories kb`) can be filled in across
later runs since results checkpoint/resume like agent_eval.py.

Run inside the app container:
    docker compose exec app python evaluation/llm_eval_compare.py --categories warehouse,combined
    docker compose exec app python evaluation/llm_eval_compare.py --categories kb --limit 15
    docker compose exec app python evaluation/llm_eval_compare.py            # everything left to do
    docker compose exec app python evaluation/llm_eval_compare.py --budget 15000   # stop early, deliberately

Every question's token spend (both approaches) is tracked and printed
running (see --budget above) — see agent/copilot.py's ask() and
evaluation/baseline_naive_rag.py's naive_rag_answer() for where it's
measured, and agent/settings.py's MAX_ANSWER_TOKENS for the per-call cap
that bounds worst-case cost.
"""
import argparse
import json

from agent.copilot import ask, get_client
from evaluation.agent_eval import JUDGE_PROMPT, is_daily_quota_exhausted, judge_answer, load_ground_truth
from evaluation.baseline_naive_rag import naive_rag_answer
from evaluation.settings import COMPARE_RESULTS_PATH

APPROACHES = ("agentic", "naive_rag")


def load_done_results():
    """Same checkpoint pattern as agent_eval.py's load_done_results: a row
    only counts as done if BOTH approaches produced an answer with no error
    on either side — a row where one approach failed still gets re-attempted
    in full, since a partial re-run would otherwise need to track four
    separate completion states instead of two."""
    if not COMPARE_RESULTS_PATH.exists():
        return {}
    done = {}
    with open(COMPARE_RESULTS_PATH) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("agentic_error") and not row.get("naive_rag_error"):
                done[row["id"]] = row
    return done


def load_all_previous_results():
    """Every row currently on disk, error or not — used ONLY as the final
    merge fallback (see main()), never for computing `todo`. load_done_results()
    deliberately excludes any row with an error on either side so it gets
    retried; using that same dict as the merge fallback meant a
    previously-errored row whose category fell outside this run's
    --categories filter was in neither `results` (out of scope) nor `done`
    (it's an error) and silently vanished from the output file — a real bug
    hit in agent_eval.py's identical pattern (see SESSION_HANDOFF.md), fixed
    here proactively rather than waiting to rediscover it in this file too."""
    if not COMPARE_RESULTS_PATH.exists():
        return {}
    all_rows = {}
    with open(COMPARE_RESULTS_PATH) as f:
        for line in f:
            row = json.loads(line)
            all_rows[row["id"]] = row
    return all_rows


def _run_agentic(question):
    answer, tools_called, tokens_used = ask(question, verbose=False, return_trace=True)
    return answer, {"tools_called": tools_called, "tokens_used": tokens_used}


def run_approach(name, fn, judge_client, question):
    """Runs one approach's answer generation + judging, returning a dict of
    result fields prefixed with the approach name. fn must return either an
    answer string, or an (answer, extra_fields_dict) pair — the agentic
    approach uses the latter to also carry tools_called through, so the
    output file shows which tool(s) it actually picked for e.g. a warehouse
    question, not just the relevance verdict. Never raises for ordinary
    failures — records them in *_error instead — except that a daily-quota
    exhaustion is re-raised so the caller can stop the whole run immediately
    instead of burning the rest of the day's questions hitting the same wall
    (same reasoning as agent_eval.py)."""
    extra = {}
    try:
        result = fn(question)
        answer, extra = result if isinstance(result, tuple) else (result, {})
    except Exception as e:
        if is_daily_quota_exhausted(e):
            raise
        return {f"{name}_answer": None, f"{name}_error": str(e),
                f"{name}_relevance": "ERROR", f"{name}_explanation": None}

    try:
        judge = judge_answer(judge_client, question, answer)
    except Exception as e:
        if is_daily_quota_exhausted(e):
            raise
        judge = {"relevance": "JUDGE_ERROR", "explanation": str(e)}

    return {
        f"{name}_answer": answer,
        f"{name}_error": None,
        f"{name}_relevance": judge.get("relevance"),
        f"{name}_explanation": judge.get("explanation"),
        **{f"{name}_{k}": v for k, v in extra.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N (post-filter) ground truth questions.")
    parser.add_argument("--categories", type=str, default=None,
                         help="Comma-separated subset of kb,warehouse,combined (default: all).")
    parser.add_argument("--budget", type=int, default=None,
                         help="Stop after this run has spent roughly this many "
                              "llama-3.3-70b-versatile tokens across BOTH approaches combined "
                              "(self-imposed, checked between questions), instead of running "
                              "until Groq's own 100k/day cap errors out. This script costs "
                              "~2x agent_eval.py per question (two generations instead of "
                              "one), so budget accordingly if both are running the same day.")
    args = parser.parse_args()

    # full_ground_truth (unfiltered, all 47, stable id order) is kept
    # separate from the filtered/limited `ground_truth` used to pick this
    # run's scope. The final write-out below must iterate over the FULL set
    # — a prior version of this script iterated over the filtered set there
    # too, which silently dropped every previously-completed row outside
    # the current --categories filter when the file was rewritten (lost an
    # 11-row warehouse+combined run this way once already). Never repeat
    # that: selection scope and output-preservation scope are not the same
    # list.
    full_ground_truth = load_ground_truth()
    ground_truth = full_ground_truth
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",")}
        ground_truth = [row for row in ground_truth if row["category"] in wanted]
    if args.limit:
        ground_truth = ground_truth[: args.limit]

    if not ground_truth:
        print("No ground truth rows match the given filters.")
        return

    done = load_done_results()
    all_previous = load_all_previous_results()
    todo = [row for row in ground_truth if row["id"] not in done]
    print(f"Comparing agentic vs. naive_rag over {len(ground_truth)} questions"
          f"{f' (categories={args.categories})' if args.categories else ''}"
          f"{f' (limited)' if args.limit else ''}"
          f" — {len(done)} already completed in {COMPARE_RESULTS_PATH}, "
          f"{len(todo)} left to run...\n")

    judge_client = get_client()

    results = []
    quota_hit = False
    total_tokens_this_run = 0
    for i, row in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {row['id']} ({row['category']}): {row['question'][:80]}")
        merged = dict(row)

        try:
            merged.update(run_approach("agentic", _run_agentic, judge_client, row["question"]))
        except Exception as e:
            print(f"\n[stopped] Hit the daily token quota during the agentic approach after "
                  f"{i - 1}/{len(todo)} completed this run. Partial results are still written "
                  f"out below — re-run the same command later to resume.")
            quota_hit = True
            break

        try:
            merged.update(run_approach("naive_rag", naive_rag_answer, judge_client, row["question"]))
        except Exception as e:
            # Agentic side already succeeded and is worth keeping — record it
            # with naive_rag marked as the thing that hit the wall, then stop.
            merged["naive_rag_answer"] = None
            merged["naive_rag_error"] = str(e)
            merged["naive_rag_relevance"] = "ERROR"
            merged["naive_rag_explanation"] = None
            total_tokens_this_run += merged.get("agentic_tokens_used") or 0
            results.append(merged)
            print(f"\n[stopped] Hit the daily token quota during the naive_rag approach after "
                  f"{i}/{len(todo)} questions this run (agentic side for this question is still "
                  f"kept). Re-run the same command later to resume.")
            quota_hit = True
            break

        question_tokens = (merged.get("agentic_tokens_used") or 0) + (merged.get("naive_rag_tokens_used") or 0)
        total_tokens_this_run += question_tokens
        print(f"    agentic={merged['agentic_relevance']}  naive_rag={merged['naive_rag_relevance']}  "
              f"tokens_used={question_tokens} (run total: {total_tokens_this_run})")
        results.append(merged)

        if args.budget and total_tokens_this_run >= args.budget:
            print(f"\n[stopped] Self-imposed --budget of {args.budget} tokens reached "
                  f"({total_tokens_this_run} spent) after {i}/{len(todo)} remaining questions — "
                  f"a deliberate early stop, not a Groq error. Re-run the same command "
                  f"whenever you want to spend more of today's budget on this.")
            break

    # Merge with everything already on disk (all_previous — error rows
    # included), preserving ground-truth order over the FULL 47-row set,
    # not the filtered `ground_truth` used above to pick this run's scope,
    # so a category-filtered run never drops previously-written rows
    # outside that filter. Deliberately all_previous here, NOT `done`:
    # `done` excludes error rows on purpose (so they retry when in scope),
    # but using it as the merge fallback too silently drops an out-of-scope
    # error row instead of preserving it untouched — see
    # load_all_previous_results()'s docstring.
    new_by_id = {row["id"]: row for row in results}
    merged_all = [new_by_id.get(row["id"], all_previous.get(row["id"])) for row in full_ground_truth]
    merged_all = [row for row in merged_all if row is not None]

    COMPARE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARE_RESULTS_PATH, "w") as f:
        for row in merged_all:
            f.write(json.dumps(row) + "\n")

    total_tokens_all_rows = sum(
        (r.get("agentic_tokens_used") or 0) + (r.get("naive_rag_tokens_used") or 0) for r in merged_all
    )
    print(f"\n{'=' * 70}")
    print(f"TOKEN USAGE: {total_tokens_this_run} spent this run (both approaches combined) / "
          f"{total_tokens_all_rows} total across all rows on disk.")
    print(f"Per-question results -> {COMPARE_RESULTS_PATH}\n")

    n = len(merged_all)
    for name in APPROACHES:
        relevant = sum(r.get(f"{name}_relevance") == "RELEVANT" for r in merged_all)
        partly = sum(r.get(f"{name}_relevance") == "PARTLY_RELEVANT" for r in merged_all)
        non_rel = sum(r.get(f"{name}_relevance") == "NON_RELEVANT" for r in merged_all)
        errors = sum(bool(r.get(f"{name}_error")) for r in merged_all)
        print(f"{name.upper()} (n={n}, of which {errors} errored):")
        print(f"  RELEVANT        {relevant:3d}/{n}  ({relevant / n:.1%})")
        print(f"  PARTLY_RELEVANT {partly:3d}/{n}  ({partly / n:.1%})")
        print(f"  NON_RELEVANT    {non_rel:3d}/{n}  ({non_rel / n:.1%})")

        print("  by category:")
        by_cat = {}
        for r in merged_all:
            by_cat.setdefault(r["category"], []).append(r)
        for cat, rows in by_cat.items():
            cat_n = len(rows)
            cat_relevant = sum(r.get(f"{name}_relevance") == "RELEVANT" for r in rows)
            print(f"    {cat:10s} n={cat_n:3d}  relevant={cat_relevant / cat_n:.1%}")
        print()

    agentic_relevant = sum(r.get("agentic_relevance") == "RELEVANT" for r in merged_all) / n
    naive_relevant = sum(r.get("naive_rag_relevance") == "RELEVANT" for r in merged_all) / n
    winner = "agentic" if agentic_relevant >= naive_relevant else "naive_rag"
    print(f"{'=' * 70}")
    print(f"CHOSEN APPROACH: {winner} "
          f"(agentic relevant={agentic_relevant:.1%} vs. naive_rag relevant={naive_relevant:.1%})")
    print("Note: naive_rag has no query_warehouse access by construction, so a low score on "
          "'warehouse'/'combined' rows reflects a structural capability gap, not a weaker model — "
          "that gap is the actual point of this comparison (see module docstring).")

    if quota_hit:
        # Bug fixed here: this used to compare len(merged_all) — every row
        # on disk across ALL categories, since the merge fix above now
        # preserves out-of-scope rows too — against len(ground_truth) — only
        # THIS run's requested/filtered scope. Comparing those produced
        # nonsense like "15/5 of the requested set" when a 5-question
        # --categories combined run had 15 total rows on disk from earlier
        # kb/warehouse runs. Restrict the numerator to rows whose id is
        # actually in this run's requested scope.
        requested_ids = {row["id"] for row in ground_truth}
        done_in_scope = sum(1 for row in merged_all if row["id"] in requested_ids)
        print(f"\n[incomplete] Stopped early on the daily quota — "
              f"{done_in_scope}/{len(ground_truth)} of the requested set has results so far.")


if __name__ == "__main__":
    main()
