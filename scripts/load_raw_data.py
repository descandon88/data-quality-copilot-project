"""
Flat-loads every CSV in data/raw/ into a `raw` schema in Postgres, one table
per file (table name = filename without extension). This is intentionally
NOT a real pipeline — no incremental loads, no typing discipline, no
lineage. It exists only so Phase 2's knowledge base can reference real
table/column names (raw.loyalty_transactions, raw.olist_orders_dataset, ...).
Phase 3 replaces this with proper dlt ingestion into bronze.

Run inside the app container:
    docker compose exec app python scripts/load_raw_data.py
"""
from pathlib import Path

import pandas as pd

from common.settings import get_engine

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSVs found in {RAW_DIR}. Run scripts/generate_loyalty_data.py "
              f"first and make sure the Olist CSVs are in data/raw/ (see data/README.md).")
        return

    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS raw;")

    for csv_path in csv_files:
        table_name = csv_path.stem
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, engine, schema="raw", if_exists="replace", index=False)
        print(f"Loaded {len(df):>7} rows -> raw.{table_name}")

    print("\nDone. Verify with:")
    print('  docker compose exec postgres psql -U dq_admin -d dq_copilot -c "\\dt raw.*"')


if __name__ == "__main__":
    main()