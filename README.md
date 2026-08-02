# Data Quality Incident Knowledge Base & Copilot

Phase 1 starter kit. Drop these files into a new repo and follow the steps below.

## Target folder structure

```
dq-copilot/
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── sql/
│   └── init.sql
├── requirements.txt
├── .env               (copy from .env.example, never commit this one)
├── .env.example
├── knowledge_base/     # Phase 2 — authored rules, contracts, postmortems
├── ingestion/          # Phase 3 — dlt pipelines
├── dbt_project/        # Phase 3-4 — bronze/silver/gold models
├── orchestration/      # Phase 3 — Kestra flow definitions
├── retrieval/          # Phase 4-5 — embedding + hybrid search + reranking
├── agent/              # Phase 6 — routing + RAG generation
├── evaluation/         # Phase 7 — eval harness
├── app/                # Phase 8 — Streamlit UI
└── monitoring/         # Phase 9 — staleness dashboard
```

There are now **two containers**: `postgres` (the database) and `app` (a Python environment with all your dependencies installed, your code mounted live from disk). You do all your actual work inside `app` — it's the same code either way, but running it in the container means it always matches your teammates'/CI's environment exactly, not just "whatever's on your laptop."

## Phase 1 — Setup (target: ~3 hours)

1. **Create the repo.**
   ```bash
   mkdir dq-copilot && cd dq-copilot
   git init
   ```

2. **Add the folder structure above** (empty folders are fine for now — `mkdir -p knowledge_base ingestion dbt_project orchestration retrieval agent evaluation app monitoring sql`).

3. **Copy in `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `init.sql` (into `sql/`), `requirements.txt`, and `.env.example`** from this starter kit.

4. **Create your real `.env`** by copying `.env.example` to `.env`, filling in a real `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`), and adding `.env` to `.gitignore` immediately — never commit real keys.

5. *(Optional but recommended)* **Set up a local Python virtual environment too**, purely so your IDE gets autocomplete/type-checking — the container is what you actually run things in, this is just for editor support.
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

6. **Build and start both containers.**
   ```bash
   docker compose up -d --build
   ```
   This builds the `app` image from your `Dockerfile` (installing `requirements.txt` inside the container) and starts both `postgres` and `app`, on the same Docker network.

7. **Verify the database is up and the vector extension is enabled.**
   ```bash
   docker compose exec postgres psql -U dq_admin -d dq_copilot -c "\dx"
   ```
   You should see `vector` listed among the installed extensions.

8. **Verify the app container can reach Postgres — from inside the container, not your host.**
   ```bash
   docker compose exec app python -c "import psycopg2, os; psycopg2.connect(host=os.environ['POSTGRES_HOST'], dbname=os.environ['POSTGRES_DB'], user=os.environ['POSTGRES_USER'], password=os.environ['POSTGRES_PASSWORD']); print('connected')"
   ```
   Note this resolves `POSTGRES_HOST` to `postgres` (the service name), not `localhost` — that override is set in `docker-compose.yml` and is the one genuinely easy-to-miss gotcha in a two-container setup. If you ever run a script with your *local* venv instead of inside the container, switch `POSTGRES_HOST` back to `localhost` for that run.

**Phase 1 is done when:** `docker compose ps` shows both containers healthy/running, `\dx` shows the `vector` extension installed, and step 8's connection test prints `connected`. From here on, every phase's work happens by editing files in your IDE (they're live-mounted into the container) and running them with `docker compose exec app <command>`.