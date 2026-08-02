"""
Embeds knowledge_base chunks (data/processed/kb_chunks.jsonl) and indexes
them into pgvector.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim). Chosen for CPU
inference speed over raw retrieval quality — this container has no GPU, and
at this project's scale (dozens to low hundreds of chunks, not millions),
MiniLM is more than sufficient; the thing that will actually determine
answer quality downstream is retrieval strategy + reranking + prompt design
(Phase 5-6), not embedding model size. Swapping to a bigger model later
(e.g. bge-base-en-v1.5) is a one-line change if Phase 7 evaluation shows
it's worth the extra latency.

Embeddings are L2-normalized at encode time, so pgvector's `<=>` cosine
distance operator works directly at query time with no extra normalization.

Storage: retrieval.kb_chunks, one row per chunk, HNSW index on the embedding
column (cosine ops — HNSW builds its graph relative to a specific distance
function, so the index type has to match how you'll query it, which is why
this isn't just "add an index" but "add a vector_cosine_ops index"). Rows
are upserted on chunk_id, so re-running this after editing a knowledge base
doc and re-chunking is safe — same idempotent-write pattern as the ingestion
pipeline's merge-loaded tables (PM-001/RULE-001 again, effectively).

Run inside the app container:
    docker compose exec app python retrieval/embed_and_index.py
"""
import json
from pathlib import Path

from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from common.postgresql import get_connection
from retrieval.settings import EMBED_MODEL_NAME

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb_chunks.jsonl"
EMBEDDING_DIM = 384
BATCH_SIZE = 32

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS retrieval;

CREATE TABLE IF NOT EXISTS retrieval.kb_chunks (
    chunk_id        text PRIMARY KEY,
    doc_id          text NOT NULL,
    doc_type        text NOT NULL,
    title           text NOT NULL,
    section         text NOT NULL,
    chunk_text      text NOT NULL,
    source_file     text NOT NULL,
    metadata        jsonb NOT NULL,
    embedding       vector(384) NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw
    ON retrieval.kb_chunks
    USING hnsw (embedding vector_cosine_ops);
"""

UPSERT_SQL = """
INSERT INTO retrieval.kb_chunks
    (chunk_id, doc_id, doc_type, title, section, chunk_text, source_file, metadata, embedding, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    doc_type = EXCLUDED.doc_type,
    title = EXCLUDED.title,
    section = EXCLUDED.section,
    chunk_text = EXCLUDED.chunk_text,
    source_file = EXCLUDED.source_file,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding,
    updated_at = now();
"""


def load_chunks():
    if not CHUNKS_PATH.exists():
        raise SystemExit(
            f"{CHUNKS_PATH} not found. Run ingestion/chunk_knowledge_base.py first."
        )
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    conn = get_connection()
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()

    total = 0
    for batch in batched(chunks, BATCH_SIZE):
        texts = [c["embedding_text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        with conn.cursor() as cur:
            for chunk, embedding in zip(batch, embeddings):
                cur.execute(UPSERT_SQL, (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk["doc_type"],
                    chunk["title"],
                    chunk["section"],
                    chunk["text"],
                    chunk["source_file"],
                    json.dumps(chunk["metadata"]),
                    embedding,
                ))
        conn.commit()
        total += len(batch)
        print(f"  embedded + upserted {total}/{len(chunks)}")

    conn.close()
    print(f"\nDone. {total} chunks indexed into retrieval.kb_chunks.")
    print("Verify with:")
    print('  docker compose exec postgres psql -U dq_admin -d dq_copilot -c '
          '"select doc_type, count(*) from retrieval.kb_chunks group by doc_type;"')


if __name__ == "__main__":
    main()