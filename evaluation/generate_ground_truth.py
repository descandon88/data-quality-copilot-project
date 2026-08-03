"""
Generates the ground truth set for Phase 7 evaluation: a mix of LLM-generated
questions (one batch per knowledge_base doc) and hand-authored questions that
require live warehouse data or both tools together.

Why two sources instead of pure LLM generation for everything:
- Per-doc questions ("kb" category) are exactly the LLM Zoomcamp pattern —
  ask an LLM to write realistic questions that a specific document should
  answer, then use (question -> doc_id) pairs to measure retrieval quality
  in retrieval_eval.py. An LLM can do this well because it only needs the
  document's own content, not any live system state.
- "warehouse" and "combined" questions need real column/table names and a
  claim about what tool(s) *should* fire — that's not something to trust an
  LLM to invent correctly from scratch (it would need to already know
  agent/tools.py's schema description, at which point hand-authoring next to
  that file is more reliable than round-tripping it through a prompt). These
  are what agent_eval.py uses to measure tool-routing accuracy — retrieval_eval.py
  doesn't touch them except for "combined", where the kb half is still a
  valid retrieval target.

Run inside the app container:
    docker compose exec app python evaluation/generate_ground_truth.py
"""
import json
import re
from pathlib import Path

import frontmatter

from agent.copilot import get_client
from evaluation.settings import GROUND_TRUTH_PATH, QUESTIONS_PER_DOC

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

DOC_TYPE_BY_FOLDER = {
    "postmortems": "postmortem",
    "rules": "rule",
    "contracts": "contract",
}

# Not llama-3.3-70b-versatile: question generation is a plain JSON-array
# completion with no tool-calling involved, so it doesn't need (and
# shouldn't spend) budget from that model's 100k-tokens/day free-tier cap —
# see evaluation/settings.py's JUDGE_MODEL_NAME comment for the same
# reasoning applied to the judge model.
GENERATION_MODEL = "llama-3.1-8b-instant"

GENERATION_PROMPT = """You are generating evaluation questions for a data quality \
incident search system used by data engineers.

Given the internal document below, write {n} realistic questions a data \
engineer might type into a search box, where THIS document is the correct, \
expected answer. Vary the phrasing and angle across the {n} questions \
(e.g. one about root cause, one about impact/numbers, one about the fix or \
the rule itself) — don't just reword the title {n} times. If a question \
references this document's own id the way a real engineer who half-remembers \
an incident number would, it MUST be exactly "{doc_id}" (this document's own \
id, given below) — never a different id, and never an id you haven't been \
given here.

Do not quote long verbatim phrases from the document. Return ONLY a JSON \
array of {n} strings — no markdown, no explanation, no surrounding text.

Document id: {doc_id}
Document title: {title}
Document type: {doc_type}

Content:
{content}
"""

# Hand-authored: needs live query_warehouse only, no knowledge base doc is
# the "right answer" for these — schema drawn directly from
# agent/tools.py's TOOL_DEFINITIONS description.
WAREHOUSE_QUESTIONS = [
    "How many loyalty accounts do we have in total?",
    "What's the average points_balance across all loyalty accounts?",
    "How many orders in olist_orders_dataset have order_status = 'delivered'?",
    "How many rows are in bronze.loyalty_transactions?",
    "How many accounts are on the gold tier right now?",
    "What's the total points_earned across all earn transactions?",
]

# Hand-authored: needs BOTH tools — search_knowledge_base to get the rule's
# real check logic, then query_warehouse to run it. expected_doc_id is the
# rule that owns each check, so retrieval_eval.py can still score the kb
# half of these.
COMBINED_QUESTIONS = [
    ("Are we currently violating RULE-001, and how many rows are affected?", "RULE-001"),
    ("Are we currently violating RULE-002, and how many rows are affected?", "RULE-002"),
    ("Are we currently violating RULE-003, and how many rows are affected?", "RULE-003"),
    ("Are we currently violating RULE-004, and how many accounts are affected?", "RULE-004"),
    ("Are we currently violating RULE-005, and how many accounts are affected?", "RULE-005"),
]

JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _request_questions(client, doc_id, title, doc_type, content, n):
    prompt = GENERATION_PROMPT.format(
        n=n, doc_id=doc_id, title=title, doc_type=doc_type,
        content=content[:3000],  # keep the prompt bounded; docs here are short anyway
    )
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = (response.choices[0].message.content or "").strip()

    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        # Groq occasionally wraps the array in prose despite instructions —
        # fall back to extracting the first [...] block before giving up.
        match = JSON_ARRAY_RE.search(raw)
        if not match:
            return [], raw
        try:
            questions = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], raw

    parsed = [q.strip() for q in questions if isinstance(q, str) and q.strip()][:n]
    return parsed, raw


def generate_questions_for_doc(client, doc_id, title, doc_type, content, n=QUESTIONS_PER_DOC):
    # One retry: an empty/unparseable completion has been observed to happen
    # for a single doc in an otherwise-clean run (raw output uncaptured at
    # the time), with no exception raised — i.e. the model just returned
    # something empty or malformed for that one call. Cheap to retry once on
    # 8b-instant; always print the raw output on a real failure this time so
    # a repeat isn't a silent "0 questions" with no way to tell why.
    for attempt in range(2):
        try:
            questions, raw = _request_questions(client, doc_id, title, doc_type, content, n)
        except Exception as e:
            print(f"  [warn] {doc_id}: request failed on attempt {attempt + 1} ({e!r})")
            questions, raw = [], None
        if questions:
            return questions
        if attempt == 0:
            raw_display = repr(raw[:200]) if raw is not None else "<request failed>"
            print(f"  [warn] {doc_id}: got 0 usable questions "
                  f"(raw output: {raw_display}), retrying once...")
    print(f"  [warn] {doc_id}: still 0 questions after retry — skipping this doc.")
    return []


def load_kb_docs():
    docs = []
    for folder, doc_type in DOC_TYPE_BY_FOLDER.items():
        folder_path = KB_DIR / folder
        if not folder_path.exists():
            continue
        for md_file in sorted(folder_path.glob("*.md")):
            post = frontmatter.load(md_file)
            meta = dict(post.metadata)
            docs.append({
                "doc_id": meta.get("id", md_file.stem),
                "title": meta.get("title", md_file.stem),
                "doc_type": doc_type,
                "content": post.content,
            })
    return docs


def main():
    client = get_client()
    docs = load_kb_docs()
    print(f"Found {len(docs)} knowledge base docs. Generating {QUESTIONS_PER_DOC} questions each...\n")

    ground_truth = []
    gt_id = 0

    for doc in docs:
        questions = generate_questions_for_doc(
            client, doc["doc_id"], doc["title"], doc["doc_type"], doc["content"]
        )
        print(f"  {doc['doc_id']}: {len(questions)} questions generated")
        for q in questions:
            gt_id += 1
            ground_truth.append({
                "id": f"gt-{gt_id:04d}",
                "question": q,
                "category": "kb",
                "expected_doc_id": doc["doc_id"],
                "expected_tools": ["search_knowledge_base"],
                "source": "generated",
            })

    for q in WAREHOUSE_QUESTIONS:
        gt_id += 1
        ground_truth.append({
            "id": f"gt-{gt_id:04d}",
            "question": q,
            "category": "warehouse",
            "expected_doc_id": None,
            "expected_tools": ["query_warehouse"],
            "source": "handwritten",
        })

    for q, doc_id in COMBINED_QUESTIONS:
        gt_id += 1
        ground_truth.append({
            "id": f"gt-{gt_id:04d}",
            "question": q,
            "category": "combined",
            "expected_doc_id": doc_id,
            "expected_tools": ["search_knowledge_base", "query_warehouse"],
            "source": "handwritten",
        })

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w") as f:
        for row in ground_truth:
            f.write(json.dumps(row) + "\n")

    by_category = {}
    for row in ground_truth:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1

    print(f"\nWrote {len(ground_truth)} ground truth rows -> {GROUND_TRUTH_PATH}")
    for cat, count in by_category.items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
