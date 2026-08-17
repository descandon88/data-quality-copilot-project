# Data Quality Incident Knowledge Base & Copilot

**Author:** David Escandón
**Project:** Final Project for [DataTalksClub's LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)

A retrieval-augmented copilot that lets data engineers ask natural-language
questions about past data-quality incidents, the validation rules those
incidents produced, and the current state of the warehouse — instead of
grepping through postmortem docs or hand-writing SQL against tables whose
schema they half-remember.

The target reader of this README is a data engineering / analytics
engineering hiring manager, not an AI-engineering one. The retrieval and
agent pieces exist to make an otherwise-ordinary data platform project
queryable in plain English — they are not the point of the project on their
own.

## Objective

Build a production-shaped, *evaluable* RAG system for a realistic
data-engineering scenario — not a demo notebook — and back every design
decision with a real, computed number instead of an assumption. Concretely:

- A hand-authored knowledge base of realistic postmortems, validation
  rules, and data contracts, cross-linked and grounded in real statistics
  computed from the underlying dataset — not invented numbers.
- Retrieval that's measured, not assumed: hybrid search, reranking, and
  query rewriting, each compared against simpler baselines on Hit
  Rate@5/MRR@5 (see [Evaluation](#evaluation)).
- An agent that decides per-question whether it needs documented history,
  live warehouse state, or both, with tool-routing accuracy scored against
  a hand-labeled ground-truth set, and compared against a fixed
  (non-agentic) baseline to prove the routing decision earns its keep.
- A dbt mart (`silver.rule_violations`) that turns each rule's documented
  SQL logic into one real, tested, queryable table instead of copy-pasted
  prose SQL.
- A chat UI and a monitoring dashboard that reflect real usage, not
  eval-run noise, with both free user feedback and opt-in LLM-judge
  scoring.
- Full reproducibility from a clean clone: pinned `requirements.txt`,
  Docker Compose, `.env.example`, and the single Quickstart below.

## The problem

Every data team accumulates tribal knowledge about its own data: *why*
`points_balance` occasionally went negative, *which* join key silently
started coming back null after an upstream API changed a field name,
*which* validation rule exists because of which incident, and whether that
rule is a hard-stop or a warn-and-continue and why. That knowledge usually
lives in Slack threads, closed incident tickets, and the memory of whoever
was on call — not in a place a new team member (or the on-call engineer at
2am) can search.

This project is a small, realistic version of that problem, end to end: a
synthetic-but-numerically-real retail loyalty platform with five
deliberately injected data-quality incidents, a knowledge base of
postmortems and validation rules written the way a real team would write
them (specific row counts, specific dates, explicit reasoning for why a
check is a hard-stop vs. a warn-and-continue), and a copilot that can
answer both "what happened and why" (from the knowledge base) and "is it
still happening right now" (from a live warehouse query) — and knows which
of those two questions it's actually being asked.

## Dataset

Two layers, real plus synthetic on top (full download/generation
instructions in `data/README.md`):

1. **Olist Brazilian E-Commerce dataset** — real, anonymized, ~100k orders,
   customers, products, order items, and reviews from an actual Brazilian
   marketplace (public on Kaggle: `olistbr/brazilian-ecommerce`). Used as
   the realistic backbone the synthetic loyalty layer attaches to, instead
   of fabricating an entire e-commerce dataset from scratch.
2. **Synthetic loyalty layer** (`scripts/generate_loyalty_data.py`) — no
   public dataset covers retail loyalty programs (point balances and
   redemptions are exactly the kind of data no company publishes), so this
   project generates one: `loyalty_accounts.csv` and
   `loyalty_transactions.csv`, seeded (`SEED = 42`) for reproducibility and
   keyed to real Olist customers/orders. It deliberately injects five
   labeled data-quality incidents at fixed sample fractions — not bugs,
   *manufactured* incidents, so the knowledge base's postmortems (below)
   have real column names, real row counts, and a real, if synthetic,
   trail to investigate instead of a made-up scenario.

## Architecture

```
Olist e-commerce data (real, ~100k orders)
        │
        ▼
scripts/generate_loyalty_data.py   synthetic loyalty layer + 5 injected incidents
        │
        ▼
ingestion/pipeline.py (dlt)        idempotent merge-load → Postgres `bronze` schema
        │
        ▼
knowledge_base/*.md                hand-authored postmortems, rules, contracts
        │  (real row counts / dates copied from the generator's own output)
        ▼
ingestion/chunk_knowledge_base.py  section-aware markdown chunking
        │
        ▼
retrieval/embed_and_index.py       sentence-transformers embeddings → pgvector
        │
        ▼
retrieval/query_rewrite.py         query-time: LLM rewrites the question into the KB's own terms
        │
        ▼
retrieval/hybrid_search.py         BM25 + vector search → RRF fusion → cross-encoder rerank
        │
        ▼
agent/copilot.py                   Groq-backed tool-routing agent (search_knowledge_base / query_warehouse)
        │
        ▼
evaluation/                        retrieval quality + agent routing + LLM-as-judge answer quality
```

Two Postgres schemas matter: `bronze` (raw-but-typed loyalty + Olist tables,
loaded by `dlt`) and `retrieval` (the chunked, embedded knowledge base). The
agent reads both — `bronze` via a guarded read-only SQL tool, `retrieval`
via hybrid search — and decides per question which one(s) it actually
needs.

### Why hybrid search, not just vector search

Pure vector search on the query *"why would loyalty point balances be
wrong"* returned all 5 top chunks from a single postmortem (`PM-001`) and
silently excluded `RULE-001` — the actual validation rule that catches that
exact problem — because dense embeddings weight topical similarity, not
exact-term overlap (`points_balance`, `loyalty_id`). BM25 catches what
vector search under-weights; vector search catches paraphrases BM25 can't
match at all. `retrieval/hybrid_search.py` fuses both ranked lists with
Reciprocal Rank Fusion, then reranks the fused shortlist with a
cross-encoder (which scores the (query, chunk) pair jointly, not two
independent embeddings), capped at 2 chunks per document so one document
can't crowd out the rest of the result set.

### Why query rewriting

A conversational question shares fewer exact terms with the knowledge
base's own phrasing than a rewritten one does — "why would loyalty point
balances be wrong" doesn't literally contain `points_balance`, which is
exactly what `RULE-004`'s check logic is written in terms of, and exact-term
overlap is what BM25's half of the hybrid pipeline scores. `search_knowledge_base`
runs `retrieval/query_rewrite.py` first: a small, separate-quota LLM call
(`llama-3.1-8b-instant`, not the generation model) that reformulates the
question before hybrid search ever runs. It fails open — any error falls
back to the original question unchanged, so a rewrite failure degrades
retrieval, never breaks it. Measured, not just asserted: see
[Retrieval evaluation](#retrieval-evaluation) above for the before/after
comparison once `retrieval_eval.py`'s 5th strategy has a real run.

### Why an agent, not a fixed RAG pipeline

*"Why do we hard-stop on duplicate earn events"* only needs the knowledge
base. *"How many duplicate earn rows exist right now"* only needs SQL
against the live warehouse. *"Are we currently violating RULE-001, and how
many rows are affected"* needs both — the rule's exact check logic from the
knowledge base, then that logic run as a live query. `agent/copilot.py`
uses Groq function-calling so the model decides, per question, which tool(s)
it actually needs, instead of always running the same retrieval step
regardless of what was asked. This is measured, not just asserted — see
[LLM evaluation](#llm-evaluation-agent-vs-a-fixed-pipeline) below.

## Project structure

```
agent/            Phase 6 — tool-routing agent + RAG generation (Groq)
common/           shared Postgres connection helpers
data/             raw CSVs (gitignored) + processed intermediates (chunks, eval results)
dbt_project/      staging models + silver.rule_violations mart (see below)
evaluation/       Phase 7 — retrieval eval, agent routing eval, LLM-as-judge comparison
ingestion/        Phase 3 — dlt pipeline (bronze schema) + knowledge-base chunker
knowledge_base/   Phase 2 — postmortems, validation rules, data contracts
retrieval/        Phase 4-5 — embeddings, pgvector indexing, hybrid search + rerank + query rewriting
scripts/          data acquisition/generation (Olist download, synthetic loyalty layer)
sql/              Postgres init (pgvector extension)
tests/            pytest unit tests (SQL guard, hybrid search, chunking)
app/              Phase 8 — Streamlit chat UI over the agent
monitoring/       Phase 9 — live-chat logging, feedback, performance + violations dashboard
orchestration/    planned Kestra scheduling (not yet built)
```

Folders listed as "not yet built" are intentionally empty — this project is
built in phases and this README reflects the current state honestly rather
than describing aspirational folders as if they were done.

## Quickstart

Requires Docker and Docker Compose. Two containers: `postgres`
(`pgvector/pgvector:pg16`) and `app` (all Python deps installed, this repo
live-mounted at `/app`). All commands below run inside `app` — there is no
supported host-Python path.

```bash
git clone <this-repo> && cd data-quality-copilot-project
cp .env.example .env        # fill in GROQ_API_KEY (free tier: console.groq.com) and Kaggle creds if using the API download path
docker compose up -d --build
```

> If your Docker Compose is the standalone binary rather than the `compose`
> plugin, use `docker-compose` (hyphenated) in place of `docker compose`
> throughout this README — both are the same containers/commands either way.

**1. Get the data** (see `data/README.md` for two download options):

```bash
docker compose exec app python scripts/api.py                      # Olist e-commerce CSVs, needs Kaggle creds
docker compose exec app python scripts/generate_loyalty_data.py    # synthetic loyalty layer + 5 injected incidents
```

**2. Load and index:**

```bash
docker compose exec app python ingestion/pipeline.py               # dlt → bronze schema, idempotent merge-load
docker compose exec app python ingestion/chunk_knowledge_base.py   # knowledge_base/*.md → data/processed/kb_chunks.jsonl
docker compose exec app python retrieval/embed_and_index.py        # chunks → pgvector (retrieval.kb_chunks)
```

**3. Build the data quality mart** (dbt — turns each `RULE-00X` check into
one real, tested table; see [Data quality mart](#data-quality-mart-dbt)
below):

```bash
docker compose exec app dbt run --project-dir dbt_project
docker compose exec app dbt test --project-dir dbt_project
```

**4. Run the test suite** (fast, no data or API calls needed — safe to run
any time, including before step 1):

```bash
docker compose exec app pytest
```

**5. Ask it something:**

```bash
docker compose exec app python agent/copilot.py "why would loyalty point balances be wrong"
docker compose exec app python agent/copilot.py "are we currently violating RULE-001, and how many rows are affected?"
```

**6. Or use the chat UI** instead of the CLI — same agent underneath, plus
it shows which tool(s) the agent actually called on each turn:

```bash
docker compose exec app streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

Then open `http://localhost:8501` (port already mapped in
`docker-compose.yml`). Every real turn here is logged for the monitoring
dashboard below — eval runs are not.

**7. Check the monitoring dashboard** — chat performance (response time,
tokens, tool usage, user feedback) plus live `RULE-00X` violations, side
by side:

```bash
docker compose exec app streamlit run monitoring/dashboard.py --server.port 8502 --server.address 0.0.0.0
```

Then open `http://localhost:8502`.

**8. Run the evaluation harness** (see [Evaluation](#evaluation) below for
what each script measures and the results so far):

```bash
docker compose exec app python evaluation/generate_ground_truth.py
docker compose exec app python evaluation/retrieval_eval.py
docker compose exec app python evaluation/agent_eval.py
docker compose exec app python evaluation/llm_eval_compare.py
```

The last two call `llama-3.3-70b-versatile` on Groq's free tier, which has
a 100k-tokens/day cap shared across everything using that model — both
scripts checkpoint their results file and resume from where they left off
on a rerun rather than redoing already-answered questions, so hitting that
cap mid-run is a "come back later" situation, not a lost run. Both also
track and print real Groq token spend per question and running for the
whole run, and accept `--budget N` to stop deliberately before Groq's own
cap errors out (useful for leaving quota for other same-day usage):

```bash
docker compose exec app python evaluation/agent_eval.py --budget 15000
docker compose exec app python evaluation/llm_eval_compare.py --categories kb --budget 15000
```

### All commands, in order

Every command above, in one block, for a single clean bootstrap from a
fresh clone (skip 6-8 if you only want the CLI and don't need the UI/eval
results):

```bash
# 0. Setup
git clone <this-repo> && cd data-quality-copilot-project
cp .env.example .env        # fill in GROQ_API_KEY and Kaggle creds
docker compose up -d --build

# 1. Get the data
docker compose exec app python scripts/api.py
docker compose exec app python scripts/generate_loyalty_data.py

# 2. Load and index
docker compose exec app python ingestion/pipeline.py
docker compose exec app python ingestion/chunk_knowledge_base.py
docker compose exec app python retrieval/embed_and_index.py

# 3. Build the data quality mart
docker compose exec app dbt run --project-dir dbt_project
docker compose exec app dbt test --project-dir dbt_project

# 4. Run the test suite
docker compose exec app pytest

# 5. Ask it something (CLI)
docker compose exec app python agent/copilot.py "why would loyalty point balances be wrong"

# 6. Chat UI -> http://localhost:8501
docker compose exec app streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0

# 7. Monitoring dashboard -> http://localhost:8502
docker compose exec app streamlit run monitoring/dashboard.py --server.port 8502 --server.address 0.0.0.0

# 8. Evaluation harness
docker compose exec app python evaluation/generate_ground_truth.py
docker compose exec app python evaluation/retrieval_eval.py
docker compose exec app python evaluation/agent_eval.py --budget 15000
docker compose exec app python evaluation/llm_eval_compare.py --budget 15000

# Teardown
docker compose down          # stop containers, keep data (pgdata volume persists)
docker compose down -v       # stop containers AND delete the Postgres volume (full reset)
```

## Evaluation

Three questions, three separate harnesses, because they're not the same
question: is retrieval finding the right document, is the agent calling
the right tool, and is the final answer actually good.

### Retrieval evaluation

Hit Rate@5 / MRR@5 across five strategies, scored at the *document* level
(a document can legitimately produce several chunks in a ranking — what
matters is whether the right document surfaced, not which of its chunks
did), against 41 ground-truth questions with a known-correct source
document:

| Strategy | Hit Rate@5 | MRR@5 |
|---|---|---|
| BM25-only | 0.829 | 0.598 |
| Vector-only | 0.927 | 0.744 |
| Hybrid RRF (no rerank) | **0.976** | 0.743 |
| Hybrid RRF + rerank (production) | 0.951 | 0.650 |
| Hybrid RRF + rerank + query-rewrite | *pending re-run* | *pending re-run* |

Hybrid beats both single-strategy baselines on Hit Rate@5, confirming the
motivating failure mode above. The production pipeline (RRF + rerank +
per-doc diversity cap) trades a small amount of raw Hit Rate/MRR against
unfused RRF for result *diversity* — the diversity cap exists specifically
so one document's chunks can't fill all 5 slots, which the aggregate metric
here doesn't reward but real answer quality depends on.

The 5th row (`retrieval/query_rewrite.py` — an LLM rewrite of the question
into the knowledge base's own terminology before hybrid search runs, the
project's 3rd best-practices bonus point alongside hybrid search and
reranking above) is implemented and unit-tested but hasn't had a real
`retrieval_eval.py` run against live Postgres/Groq yet — left as "pending
re-run" rather than a guessed number, per this project's own rule of
trusting a real computation over an assumption (see the `PM-001` 796→787
correction below).

### Agent tool-routing evaluation

Precision/recall/exact-match of `tools_called` against each question's
`expected_tools`, computed only over questions that completed without an
API-level error (infra failures are tracked separately as a reliability
rate, not folded into routing accuracy — an API timeout isn't a routing
mistake). Partial run so far (10 of 47 questions completed without error,
due to the same daily quota constraint noted above):

| Metric | Value |
|---|---|
| Exact match | 0.700 |
| Mean precision (didn't call an unneeded tool) | 0.850 |
| Mean recall (called every tool it needed) | 1.000 |

Recall of 1.000 on this partial sample means the agent never *missed* a
tool it needed; the gap to exact-match is entirely over-calling (e.g.
calling `search_knowledge_base` on a pure-warehouse question it could have
answered from `query_warehouse` alone). This harness is still running to
completion — see `evaluation/agent_eval.py`.

### LLM evaluation: agent vs. a fixed pipeline

The rubric-relevant question isn't just "is the agent's answer relevant" in
isolation — it's relevant *compared to what*. `evaluation/llm_eval_compare.py`
runs every ground-truth question through two approaches using the *same*
underlying model, judged by the *same* LLM-as-judge prompt, so the
comparison isolates the orchestration strategy, not model quality:

- **agentic** (production): `agent/copilot.py`'s real tool-routing loop.
- **naive_rag** (baseline): `evaluation/baseline_naive_rag.py` — always
  retrieves via `search_knowledge_base` only, never queries the warehouse,
  no routing decision. A fixed pipeline, by construction.

| | Relevant (warehouse) | Relevant (combined) | Relevant (kb, partial) |
|---|---|---|---|
| agentic | 50.0% | 40.0% | 83.3% |
| naive_rag | 16.7% | 0.0% | 66.7% |

The gap is largest exactly where the hypothesis predicts it should be:
`combined` questions ("are we currently violating RULE-004, and how many
accounts are affected?") need a live row count no amount of retrieval-only
context can produce, so `naive_rag` scores 0% relevant there by
construction, not because the underlying model is weaker. On `kb`
questions, where both approaches have the same retrieval available, the
gap narrows but doesn't close — the agent still edges out the fixed
pipeline. Both harnesses are still completing their full 47-question runs
as the shared daily Groq quota allows; numbers above are the real,
currently-available results, not a final report.

## Data quality incidents modeled

Five incidents, each with a postmortem, a validation rule with an explicit
enforcement rationale (hard-stop vs. warn-and-continue, tied to whether a
false negative is a financial liability or a recoverable data gap), and
real computed numbers from the synthetic dataset:

| | Incident | Rows affected | Enforcement |
|---|---|---|---|
| PM-001 / RULE-001 | Duplicate earn events (non-idempotent writes) | 787 / 115,139 transactions | hard-stop |
| PM-002 / RULE-002 | Null `loyalty_id` (upstream field rename) | 1,728 / 115,139 transactions | warn-and-continue |
| PM-003 / RULE-003 | Orphaned transactions (late-arriving dimension) | 345 / 115,139 transactions | warn-and-continue |
| PM-004 / RULE-004 | Negative points balances (redemption race) | 480 / 96,096 accounts | hard-stop |
| PM-005 / RULE-005 | Stale gold tiers (silently-failed job) | 3,879 / 9,698 gold accounts | warn-and-continue |

## Data quality mart (dbt)

`knowledge_base/rules/*.md` documents each check as SQL prose;
`dbt_project/` turns all five into one real, queryable, tested table
instead of copy-pasted queries: `silver.rule_violations`
(`rule_id, enforcement, entity_type, entity_id, detail, detected_at`).
Staging views (`stg_loyalty_accounts`, `stg_loyalty_transactions`) do light
renaming/casting only — no filtering, since hiding null `loyalty_id`s or
orphaned rows there would hide the exact incidents this mart exists to
surface.

```bash
docker compose exec app dbt run --project-dir dbt_project
docker compose exec app dbt test --project-dir dbt_project
```

Each rule's SQL was adapted, not copy-pasted: the docs filter to
`transaction_timestamp >= now() - interval '24 hours'`, written for a live
daily load window, which would return zero rows forever against this
project's static synthetic snapshot. The mart's models scan the full table
instead, so `select rule_id, count(*) from silver.rule_violations group by
rule_id` after a run should return every violation in the dataset:
`RULE-001`=787, `RULE-002`=1728, `RULE-003`=345, `RULE-004`=480,
`RULE-005`=3879 — verified against the real CSVs with a throwaway `duckdb`
check before the dbt models were written. That check caught a real
documentation drift bug along the way: `PM-001` said 796 duplicate rows;
the actual current dataset has 787. The doc was corrected, not the query.

Two data contracts (`CONTRACT-001`, `CONTRACT-002`) document the schema
guarantees and known failure modes of `bronze.loyalty_accounts` and
`bronze.loyalty_transactions`, cross-referencing the postmortems above.

## Monitoring

`monitoring/db.py` logs every real chat turn from `app/app.py` — question,
answer, which tool(s) were called, tokens used, response time, whether it
errored — to a `monitoring` Postgres schema, plus feedback per answer.
Eval runs never touch this table, so `monitoring/dashboard.py` reflects
real usage, not eval noise:

```bash
docker compose exec app streamlit run monitoring/dashboard.py --server.port 8502 --server.address 0.0.0.0
```

Two sections, seven charts: RAG app performance (response time over time,
tokens over time, tool-usage breakdown, user-feedback breakdown, an
opt-in judge-relevance breakdown, plus conversations/error-rate/thumbs-up
metric cards) and live data quality (`silver.rule_violations` by rule and
by enforcement level). The first section is what the course's monitoring
criterion actually grades — a violations-only dashboard would answer "is
the business data healthy," not "is the RAG app performing well" — so
both are here, not just one.

Two kinds of feedback per answer, independent of each other: a 👍/👎 from
whoever's chatting (free, no API call), and an opt-in **"Judge this
answer"** button that runs the same LLM-as-judge check as Phase 7's
`agent_eval.py` on that one message, on demand. It's opt-in rather than
automatic on every turn on purpose — re-judging every live message would
spend Groq quota on the thing this whole build has tried to conserve
(`--budget` flags, `MAX_ANSWER_TOKENS` caps). It's also cheap when you do
click it: the judge runs on `llama-3.1-8b-instant`, a separate free-tier
quota bucket from the `llama-3.3-70b-versatile` generation model, so it
doesn't compete with the quota your questions actually need.

## Testing

```bash
docker compose exec app pytest
```

61 unit tests over the SQL injection guard (`agent/tools.py`'s
`query_warehouse`), hybrid search's ranking logic (RRF fusion, BM25,
cross-encoder rerank + per-doc diversity cap), knowledge-base chunking, and
query rewriting's fail-open behavior (`retrieval/query_rewrite.py`) — all
mocked/faked at the DB and model boundary (see `tests/conftest.py`), so
the suite runs in under a second with no Postgres connection, no Groq
call, and no downloaded embedding model required. One test
(`test_known_gap_select_into_is_not_caught_by_the_denylist`) documents a
real, known gap rather than hiding it: `SELECT ... INTO` isn't caught by
the guard's keyword denylist.

## Tech stack

Postgres 16 + pgvector · `dlt` (idempotent merge-load ingestion) · dbt
(`silver.rule_violations` mart) · `sentence-transformers`
(`all-MiniLM-L6-v2` embeddings, CPU-only torch) · `rank-bm25` ·
cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) · Groq
(`llama-3.3-70b-versatile` generation / `llama-3.1-8b-instant` for
non-tool-calling eval tasks, via the OpenAI-compatible client) · Streamlit
(chat UI + monitoring dashboard) · Docker Compose (two services:
`postgres`, `app`).

## Status

| Phase | |
|---|---|
| 1 — Setup | Done |
| 2 — Knowledge base authoring | Done |
| 3 — Ingestion + chunking | Done |
| 4 — Embedding + indexing | Done |
| 5 — Hybrid search + reranking | Done |
| 6 — RAG generation + agent routing | Done |
| 7 — Evaluation harness | In progress — see [Evaluation](#evaluation) |
| 8 — Streamlit UI | Done |
| 9 — Monitoring (feedback + performance dashboard) | Done — see [Monitoring](#monitoring) |
| 10 — Testing, polish, documentation | In progress |

dbt (`dbt_project/`, see [Data quality mart](#data-quality-mart-dbt)
below) wasn't part of the original phase plan but is also done — built and
verified against real Postgres this session.
