"""
dlt ingestion pipeline: loads the Olist + synthetic loyalty CSVs from
data/raw/ into a `bronze` schema in Postgres.

Design choices:
- Olist reference/fact tables: write_disposition="replace" (full snapshot
  reload each run). This is a static historical dataset, not a live source
  system 
- loyalty_accounts / loyalty_transactions: write_disposition="merge" with an
  explicit primary_key. This is a deliberate callback to PM-001/RULE-001 in
  the knowledge base: the raw-layer incident happened because writes weren't
  idempotent. The ingestion layer that feeds bronze shouldn't repeat that
  mistake — merge loads upsert on primary key, so re-running the pipeline is
  safe instead of re-appending duplicates.

This replaces scripts/load_raw_data.py, which was a Phase 1/2 stopgap flat
loader with no dedup, no typing discipline, and no lineage.

Run inside the app container:
    docker compose exec app python ingestion/pipeline.py
"""
from pathlib import Path

import dlt
import pandas as pd

from common.settings import get_postgres_url

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# table name -> primary key, for tables loaded with upsert semantics
LOYALTY_TABLES = {
    "loyalty_accounts": "loyalty_id",
    "loyalty_transactions": "transaction_id",
}

# table name -> full-snapshot reload each run
REPLACE_TABLES = [
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_orders_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
    "product_category_name_translation",
]


def make_replace_resource(table_name: str):
    @dlt.resource(name=table_name, write_disposition="replace")
    def _resource():
        df = pd.read_csv(RAW_DIR / f"{table_name}.csv")
        yield df.to_dict(orient="records")

    return _resource


def make_merge_resource(table_name: str, primary_key: str):
    @dlt.resource(name=table_name, write_disposition="merge", primary_key=primary_key)
    def _resource():
        df = pd.read_csv(RAW_DIR / f"{table_name}.csv")
        yield df.to_dict(orient="records")

    return _resource


def build_pipeline(destination=None, dataset_name: str = "bronze"):
    """Split out so tests can pass a different destination (e.g. duckdb)."""
    return dlt.pipeline(
        pipeline_name="retail_loyalty_bronze",
        destination=destination or dlt.destinations.postgres(credentials=get_postgres_url(driver=None)),
        dataset_name=dataset_name,
    )


def all_resources():
    resources = [make_replace_resource(t) for t in REPLACE_TABLES]
    resources += [make_merge_resource(t, pk) for t, pk in LOYALTY_TABLES.items()]
    return resources


def main():
    required = list(REPLACE_TABLES) + list(LOYALTY_TABLES)
    missing = [t for t in required if not (RAW_DIR / f"{t}.csv").exists()]
    if missing:
        raise SystemExit(
            "Missing CSVs in data/raw/: " + ", ".join(missing) + ". "
            "Run scripts/download_olist.py (or download manually) and "
            "scripts/generate_loyalty_data.py first — see data/README.md."
        )

    pipeline = build_pipeline()
    load_info = pipeline.run(all_resources())
    print(load_info)

    print("\nLoaded into schema 'bronze'. Verify with:")
    print('  docker compose exec postgres psql -U dq_admin -d dq_copilot -c "\\dt bronze.*"')


if __name__ == "__main__":
    main()