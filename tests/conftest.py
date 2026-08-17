"""
Shared pytest setup.

Stubs out heavy/optional ML dependencies (sentence-transformers, pgvector)
*only if they aren't actually installed*, so this test suite can exercise
pure logic — the SQL injection guard, RRF fusion, cross-encoder rerank +
diversity cap, markdown chunking — without needing torch, a downloaded
embedding model, or a live Postgres connection.

Inside the app container these packages ARE really installed and really
used end-to-end (see requirements.txt) — the try/except below prefers the
real import whenever it's available, so tests get full fidelity there. The
stub only kicks in when a package is genuinely missing (e.g. running this
suite somewhere lighter than the full container), so the unit tests stay
runnable without pulling in a multi-hundred-MB ML dependency just to import
a module and check its pure functions.

Run via:
    docker compose exec app pytest
"""
import sys
import types
from pathlib import Path

# repo root on sys.path as a fallback for environments that don't already
# set PYTHONPATH=/app the way docker-compose.yml does for the `app` service
# (see CLAUDE.md's "Dependency install order" section) — makes `import
# common`, `import agent`, etc. work regardless of how pytest was invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = type("SentenceTransformer", (), {})
    stub.CrossEncoder = type("CrossEncoder", (), {})
    sys.modules["sentence_transformers"] = stub

try:
    from pgvector.psycopg2 import register_vector  # noqa: F401
except ImportError:
    pgvector_stub = types.ModuleType("pgvector")
    pgvector_psycopg2_stub = types.ModuleType("pgvector.psycopg2")
    pgvector_psycopg2_stub.register_vector = lambda conn: None
    sys.modules["pgvector"] = pgvector_stub
    sys.modules["pgvector.psycopg2"] = pgvector_psycopg2_stub
