"""
Generates a synthetic loyalty-program layer (accounts + point transactions)
keyed to real Olist customers/orders, then deliberately corrupts a controlled
subset of rows to manufacture the data-quality incidents the Phase 2 knowledge
base (postmortems, rules, contracts) will document.

Input  (data/raw/):  olist_customers_dataset.csv, olist_orders_dataset.csv
Output (data/raw/):  loyalty_accounts.csv, loyalty_transactions.csv

Run inside the app container:
    docker compose exec app python scripts/generate_loyalty_data.py
"""
import random
from datetime import timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TIERS = ["bronze", "silver", "gold"]
TIER_WEIGHTS = [0.6, 0.3, 0.1]


def load_olist():
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp"
    ])
    return customers, orders


def build_accounts(customers: pd.DataFrame) -> pd.DataFrame:
    unique_customers = customers.drop_duplicates("customer_unique_id").reset_index(drop=True)
    n = len(unique_customers)
    enrollment_dates = [
        fake.date_time_between(start_date="-3y", end_date="-30d") for _ in range(n)
    ]
    accounts = pd.DataFrame({
        "loyalty_id": [f"LOY-{i:07d}" for i in range(n)],
        "customer_unique_id": unique_customers["customer_unique_id"],
        "tier": random.choices(TIERS, weights=TIER_WEIGHTS, k=n),
        "points_balance": [random.randint(0, 5000) for _ in range(n)],
        "enrollment_date": enrollment_dates,
        "last_activity_date": [
            d + timedelta(days=random.randint(1, 700)) for d in enrollment_dates
        ],
        "email_opt_in": [random.random() < 0.7 for _ in range(n)],
    })
    return accounts


def build_transactions(accounts: pd.DataFrame, customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    cust_to_loyalty = customers[["customer_id", "customer_unique_id"]].merge(
        accounts[["loyalty_id", "customer_unique_id"]], on="customer_unique_id", how="inner"
    )
    orders_with_loyalty = orders.merge(cust_to_loyalty, on="customer_id", how="inner")

    rows = []
    txn_counter = 0
    for _, row in orders_with_loyalty.iterrows():
        points_earned = random.randint(5, 500)
        rows.append({
            "transaction_id": f"TXN-{txn_counter:08d}",
            "loyalty_id": row["loyalty_id"],
            "order_id": row["order_id"],
            "points_earned": points_earned,
            "points_redeemed": 0,
            "transaction_type": "earn",
            "transaction_timestamp": row["order_purchase_timestamp"],
        })
        txn_counter += 1

        if random.random() < 0.15:
            redeem_points = random.randint(50, points_earned + 200)
            rows.append({
                "transaction_id": f"TXN-{txn_counter:08d}",
                "loyalty_id": row["loyalty_id"],
                "order_id": row["order_id"],
                "points_earned": 0,
                "points_redeemed": redeem_points,
                "transaction_type": "redeem",
                "transaction_timestamp": row["order_purchase_timestamp"] + timedelta(days=random.randint(1, 60)),
            })
            txn_counter += 1

    return pd.DataFrame(rows)


def inject_incidents(accounts: pd.DataFrame, transactions: pd.DataFrame):
    """
    Corrupts a controlled, reproducible (seeded) subset of rows.
    Returns (accounts, transactions, summary_dict) — summary is what you
    paste into the Phase 2 postmortem docs.
    """
    summary = {}

    # Incident A: null join keys — ~1.5% of transactions lose loyalty_id,
    # simulating an upstream schema-drift bug that dropped the field silently.
    null_mask = transactions.sample(frac=0.015, random_state=SEED).index
    transactions.loc[null_mask, "loyalty_id"] = None
    summary["null_loyalty_id_rows"] = len(null_mask)

    # Incident B: duplicate earn events — ~0.8% of "earn" rows get double-counted,
    # simulating an at-least-once delivery bug in the event pipeline (no
    # idempotency key). The duplicate gets a NEW transaction_id, matching
    # PM-001's actual claim ("a second earn row with a new transaction_id but
    # the same order_id and loyalty_id") — deliberately NOT an exact-duplicate
    # row. An exact duplicate (same transaction_id) would get silently
    # deduped away by ingestion/pipeline.py's transaction_id-keyed merge load
    # before it ever reached bronze, masking the actual failure mode this
    # incident is meant to demonstrate. RULE-001's detection query groups by
    # (loyalty_id, order_id), not transaction_id, so it still catches this.
    earn_rows = transactions[transactions["transaction_type"] == "earn"]
    dup_sample = earn_rows.sample(frac=0.008, random_state=SEED).copy()
    next_id = len(transactions)
    dup_sample["transaction_id"] = [f"TXN-{next_id + i:08d}" for i in range(len(dup_sample))]
    transactions = pd.concat([transactions, dup_sample], ignore_index=True)
    summary["duplicate_earn_rows"] = len(dup_sample)

    # Incident C: orphaned transactions — ~0.3% reference a loyalty_id that
    # doesn't exist in loyalty_accounts, simulating a late-arriving-dimension race
    # (transaction lands before the account record is committed).
    orphan_mask = transactions.sample(frac=0.003, random_state=SEED + 1).index
    transactions.loc[orphan_mask, "loyalty_id"] = [
        f"LOY-{9000000 + i:07d}" for i in range(len(orphan_mask))
    ]
    summary["orphaned_transaction_rows"] = len(orphan_mask)

    # Incident D: negative balances — ~0.5% of accounts get a balance that went
    # negative, simulating a redemption race condition (two redemptions cleared
    # against the same balance check before either write landed).
    neg_mask = accounts.sample(frac=0.005, random_state=SEED).index
    accounts.loc[neg_mask, "points_balance"] = [
        -random.randint(10, 300) for _ in range(len(neg_mask))
    ]
    summary["negative_balance_rows"] = len(neg_mask)

    # Incident E: stale tiers — ~2% of accounts have last_activity_date more than
    # a year old but are still marked "gold", simulating a broken nightly
    # tier-reassessment job that silently stopped running.
    stale_candidates = accounts[accounts["tier"] == "gold"].sample(
        frac=0.4, random_state=SEED
    ).index
    accounts.loc[stale_candidates, "last_activity_date"] = pd.Timestamp("2024-01-15")
    summary["stale_gold_tier_rows"] = len(stale_candidates)

    return accounts, transactions, summary


def main():
    customers, orders = load_olist()
    accounts = build_accounts(customers)
    transactions = build_transactions(accounts, customers, orders)
    accounts, transactions, summary = inject_incidents(accounts, transactions)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    accounts.to_csv(RAW_DIR / "loyalty_accounts.csv", index=False)
    transactions.to_csv(RAW_DIR / "loyalty_transactions.csv", index=False)

    print(f"Wrote {len(accounts)} rows -> loyalty_accounts.csv")
    print(f"Wrote {len(transactions)} rows -> loyalty_transactions.csv")
    print("\nInjected incidents (paste these numbers into the postmortem docs):")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()