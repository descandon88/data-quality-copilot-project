# Data sourcing

This project needs two layers of data: a real retail order base, and a synthetic
loyalty layer on top of it (no public dataset covers retail loyalty programs —
customer point balances / redemptions are exactly the kind of PII no company
publishes).

## 1. Retail order base — Olist Brazilian E-Commerce dataset

Real, anonymized data: ~100k orders, customers, products, order items, reviews.

**Get it one of these ways (no Kaggle login needed for the mirror):**

- GitHub mirror (raw CSVs, no auth): https://github.com/fortunewalla/olist
- Official source (requires free Kaggle account, most complete/canonical):
  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Download and place these files in `data/raw/`:
```
olist_customers_dataset.csv
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_products_dataset.csv
olist_order_payments_dataset.csv
product_category_name_translation.csv
```

Only `olist_customers_dataset.csv` and `olist_orders_dataset.csv` are required
for the loyalty-data generation step below; grab the rest now since Phase 3
ingestion will want the full set.

## 2. Synthetic loyalty layer — generated

Run:
```bash
docker compose exec app python scripts/generate_loyalty_data.py
```

This reads the two Olist files above and writes two new CSVs into `data/raw/`:

- `loyalty_accounts.csv` — one row per customer (`loyalty_id`, `customer_unique_id`,
  `tier`, `points_balance`, `enrollment_date`, `last_activity_date`, `email_opt_in`)
- `loyalty_transactions.csv` — points earn/redeem/adjust/expire events, some tied
  to real Olist `order_id`s (`transaction_id`, `loyalty_id`, `order_id`,
  `points_earned`, `points_redeemed`, `transaction_type`, `transaction_timestamp`)

**It also deliberately corrupts a controlled subset of rows** — null join keys,
duplicate earn events, negative balances, orphaned transactions, stale tiers.
These aren't bugs; they're manufactured incidents so the knowledge-base
postmortems in Phase 2 have real column names, real row counts, and a real
(if synthetic) trail to investigate, instead of a made-up scenario. The script
prints a summary of exactly what it corrupted and how many rows — copy those
numbers into the postmortem docs so they're accurate.

## 3. Load into Postgres

```bash
docker compose exec app python scripts/load_raw_data.py
```

Loads all CSVs in `data/raw/` into a `raw` schema in Postgres, one table per
file, column types inferred by pandas/`to_sql`. This is a flat load, not a
pipeline — Phase 3 replaces it with proper `dlt` ingestion. It exists purely
so Phase 2's knowledge base can reference real `raw.loyalty_transactions`-style
table and column names instead of placeholders.