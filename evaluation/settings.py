"""
Shared constants for the Phase 7 evaluation harness (generate_ground_truth.py,
retrieval_eval.py, agent_eval.py) — same rationale as common/postgresql.py and
retrieval/settings.py: one place for the knobs, instead of each script
re-declaring its own copy.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.jsonl"
RETRIEVAL_RESULTS_PATH = DATA_DIR / "retrieval_eval_results.jsonl"
AGENT_RESULTS_PATH = DATA_DIR / "agent_eval_results.jsonl"
COMPARE_RESULTS_PATH = DATA_DIR / "llm_eval_compare_results.jsonl"

# --- generate_ground_truth.py ---
# How many LLM-generated questions per knowledge_base doc. Kept small (3) to
# limit Groq calls on the free tier — this is a "dozens of chunks" project,
# not a "thousands of chunks" one, so 3/doc is enough to measure retrieval
# quality without burning the day's rate limit on question generation alone.
QUESTIONS_PER_DOC = 3

# --- retrieval_eval.py ---
# Cutoff for Hit Rate@K / MRR@K, and for how many results each retrieval
# strategy contributes before scoring. Matches retrieval/settings.py's
# FINAL_TOP_K so the "hybrid + rerank" strategy here evaluates the exact
# same top-K the production agent actually sees.
EVAL_TOP_K = 5

# --- agent_eval.py ---
# Judge model for LLM-as-judge answer scoring. Deliberately NOT
# llama-3.3-70b-versatile (the generation model, agent/settings.py's
# MODEL_NAME): that model has a hard 100k-tokens/day cap on this project's
# free Groq tier, and it needs to be spent on the thing actually being
# measured — agent/copilot.py's real tool-routing behavior — not burned on
# judging, which is a plain text-in/JSON-out call with no tool-calling
# involved. llama-3.1-8b-instant's documented unreliability (see
# agent/settings.py) is specifically about malformed *tool calls*
# (`tool_use_failed`); since judge_answer() never passes `tools=`, that
# failure mode doesn't apply here, making 8b-instant's separate quota
# bucket a safe place to move this cost to. Using a different, smaller
# model than the one being judged is also generally healthier for
# LLM-as-judge than self-evaluation — a nice side effect, not just a
# quota workaround.
JUDGE_MODEL_NAME = "llama-3.1-8b-instant"
