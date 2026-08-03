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
- "RELEVANT": the answer directly and correctly addresses the question.
- "PARTLY_RELEVANT": the answer addresses the question but is incomplete, \
vague, hedges unnecessarily, or is only tangentially on-topic.
- "NON_RELEVANT": the answer does not address the question, refuses without \
justification, or contradicts information in the question itself.

Return ONLY a JSON object, no other text: \
{{"relevance": "...", "explanation": "one brief sentence"}}
"""

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_ground_truth(limit=None):
    rows = []
    with open(GROUND_TRUTH_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def judge_answer(client, question, answer):
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    response = client.chat.completions.create(
        model=JUDGE_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N ground truth questions (for smoke-testing).")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.limit)
    print(f"Running agent eval over {len(ground_truth)} questions"
          f"{f' (limited from full set)' if args.limit else ''}...\n")
    if not ground_truth:
        print(f"No rows in {GROUND_TRUTH_PATH} — run evaluation/generate_ground_truth.py first.")
        return

    judge_client = get_client()

    results = []
    for i, row in enumerate(ground_truth, 1):
        print(f"[{i}/{len(ground_truth)}] {row['id']} ({row['category']}): {row['question'][:80]}")
        try:
            answer, tools_called = ask(row["question"], verbose=False, return_trace=True)
        except Exception as e:
            print(f"    [error] {e}")
            results.append({
                **row, "answer": None, "tools_called": [], "error": str(e),
                "exact_match": False, "precision": 0.0, "recall": 0.0,
                "judge_relevance": "ERROR",
            })
            if is_daily_quota_exhausted(e):
                print(f"\n[stopped] Hit llama-3.3-70b-versatile's daily token quota after "
                      f"{i}/{len(ground_truth)} questions. This is a hard cap on Groq's free "
                      f"tier, not a bug here, and it won't clear on a short retry — stopping "
                      f"now instead of re-hitting the same wall for every remaining question. "
                      f"Results collected so far are still written out below. Re-run later "
                      f"once the quota has recovered (it behaves like a rolling window rather "
                      f"than a fixed midnight reset, based on how slowly 'Used' moves between "
                      f"runs on the same day).")
                break
            continue

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
                })
                print(f"\n[stopped] Hit the judge model's daily token quota after "
                      f"{i}/{len(ground_truth)} questions. Routing results collected so far "
                      f"(including this one) are still valid and written out below.")
                break

        print(f"    tools_called={tools_called} exact_match={exact_match} "
              f"judge={judge.get('relevance')}")

        results.append({
            **row,
            "answer": answer,
            "tools_called": tools_called,
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "judge_relevance": judge.get("relevance"),
            "judge_explanation": judge.get("explanation"),
        })

    AGENT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENT_RESULTS_PATH, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    n = len(results)
    exact_match_rate = sum(r["exact_match"] for r in results) / n
    mean_precision = sum(r["precision"] for r in results) / n
    mean_recall = sum(r["recall"] for r in results) / n

    relevance_counts = {}
    for r in results:
        relevance_counts[r["judge_relevance"]] = relevance_counts.get(r["judge_relevance"], 0) + 1

    print(f"\n{'=' * 60}")
    print("ROUTING (tool selection vs. expected_tools)")
    print(f"  exact match rate: {exact_match_rate:.3f}")
    print(f"  mean precision:   {mean_precision:.3f}  (didn't call unneeded tools)")
    print(f"  mean recall:      {mean_recall:.3f}  (called every tool it needed)")
    print("\nANSWER QUALITY (LLM-as-judge)")
    for label, count in sorted(relevance_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:16s} {count:3d}/{n}  ({count / n:.1%})")

    print("\nBy category:")
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)
    for cat, rows in by_category.items():
        cat_n = len(rows)
        cat_exact = sum(r["exact_match"] for r in rows) / cat_n
        cat_relevant = sum(r["judge_relevance"] == "RELEVANT" for r in rows) / cat_n
        print(f"  {cat:12s} n={cat_n:3d}  exact_match={cat_exact:.3f}  relevant={cat_relevant:.3f}")

    print(f"\nPer-question results -> {AGENT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
