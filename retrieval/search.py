"""
Quick semantic search test against retrieval.kb_chunks — sanity-checks that
embed_and_index.py actually worked, before building the full hybrid search +
reranking pipeline in Phase 5.

Run inside the app container:
    docker compose exec app python retrieval/search.py "why would loyalty point balances be wrong"
"""
import sys

from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from common.postgresql import get_connection
from retrieval.settings import EMBED_MODEL_NAME

SEARCH_SQL = """
SELECT chunk_id, doc_id, title, section, chunk_text,
       1 - (embedding <=> %s) AS similarity
FROM retrieval.kb_chunks
ORDER BY embedding <=> %s
LIMIT %s;
"""


def search(query: str, k: int = 5):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    query_embedding = model.encode(query, normalize_embeddings=True)

    conn = get_connection()
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(SEARCH_SQL, (query_embedding, query_embedding, k))
        rows = cur.fetchall()
    conn.close()
    return rows


def main():
    query = " ".join(sys.argv[1:]) or "why would loyalty point balances be wrong"
    print(f"Query: {query!r}\n")
    for chunk_id, doc_id, title, section, text, similarity in search(query):
        print(f"[{similarity:.3f}] {doc_id} — {title} — {section}")
        print(f"   {text[:160].strip()}...")
        print()


if __name__ == "__main__":
    main()