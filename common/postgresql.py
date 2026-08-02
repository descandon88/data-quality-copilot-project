"""
Shared Postgres connection settings, read from the same env vars in `.env`
that docker-compose passes into the `app` container (POSTGRES_HOST is
overridden to `postgres` there — see CLAUDE.md).

Consumers need the connection in different shapes:
  - scripts/load_raw_data.py: a SQLAlchemy Engine (psycopg2 driver)
  - ingestion/pipeline.py: a bare postgresql:// URL string for dlt's
    postgres destination, which manages its own driver
  - retrieval/search.py, retrieval/embed_and_index.py: a raw psycopg2
    connection (register_vector() needs a psycopg2 connection, not a
    SQLAlchemy Engine)
"""
import os

import psycopg2
from sqlalchemy import create_engine


def get_postgres_params() -> dict:
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "db": os.environ.get("POSTGRES_DB", "dq_copilot"),
        "user": os.environ.get("POSTGRES_USER", "dq_admin"),
        "password": os.environ.get("POSTGRES_PASSWORD", "dq_password"),
    }


def get_postgres_url(driver: str | None = "psycopg2") -> str:
    p = get_postgres_params()
    scheme = f"postgresql+{driver}" if driver else "postgresql"
    return f"{scheme}://{p['user']}:{p['password']}@{p['host']}:{p['port']}/{p['db']}"


def get_engine():
    return create_engine(get_postgres_url())


def get_connection():
    p = get_postgres_params()
    return psycopg2.connect(
        host=p["host"],
        port=p["port"],
        dbname=p["db"],
        user=p["user"],
        password=p["password"],
    )
