---
id: CONTRACT-002
title: loyalty_transactions data contract
owning_team: loyalty-platform
table: bronze.loyalty_transactions
primary_key: transaction_id
write_disposition: merge
tags: [data-contract, loyalty-transactions, schema]
---

## What this table is

An append-only ledger of loyalty point events: one row per `earn` or
`redeem` event tied to a purchase order. Populated by the purchase/event
consumer (separate write path from `loyalty_accounts` — see CONTRACT-001).
Loaded into `bronze` via `ingestion/pipeline.py` with
`write_disposition="merge"` keyed on `transaction_id`, so a redelivered
event with the same `transaction_id` upserts instead of appending a
duplicate — this is the direct fix that came out of PM-001.

## Schema

| Column | Type | Nullable | Meaning |
|---|---|---|---|
| `transaction_id` | text | no | Primary key. Format `TXN-XXXXXXXX`. |
| `loyalty_id` | text | should not be null in practice | Foreign key to `bronze.loyalty_accounts.loyalty_id`. **Must not be null** (RULE-002) and **must reference an existing account** (RULE-003) for the row to be usable downstream — see "Known failure modes." |
| `order_id` | text | no | Foreign key to `bronze.olist_orders_dataset.order_id`. The stable key used to backfill `loyalty_id` when it's missing or wrong (see PM-002, PM-003). |
| `points_earned` | integer | no | Points credited by this row. `0` for `redeem` rows. |
| `points_redeemed` | integer | no | Points debited by this row. `0` for `earn` rows. |
| `transaction_type` | text | no | One of `earn`, `redeem`. |
| `transaction_timestamp` | timestamp | no | When the event occurred. |

## Guarantees this table is supposed to provide

- At most one `earn` row per `(loyalty_id, order_id)` pair within any
  24-hour window — an event should be idempotently deduped by the consumer,
  not appear twice under two `transaction_id`s. See RULE-001.
- Every row has a non-null `loyalty_id` that resolves to a real row in
  `bronze.loyalty_accounts`. See RULE-002 (null keys) and RULE-003
  (unresolvable keys) — two distinct failure modes with two distinct rules.
- `transaction_id` is a stable, deterministic identifier for a given
  underlying event, so redelivery produces an upsert, not a duplicate
  append, at the ingestion layer.

## Known failure modes (documented incidents)

- **Duplicate `earn` events** — a non-idempotent events consumer produced a
  second `earn` row (new `transaction_id`, same `order_id`/`loyalty_id`)
  for a redelivered message. See PM-001 / RULE-001 (hard-stop — this is the
  one financial-liability case in this table's failure history).
- **Null `loyalty_id`** — an unversioned upstream payload change silently
  dropped the join-key field for a subset of events. See PM-002 / RULE-002
  (warn-and-continue).
- **Orphaned `loyalty_id`** — a late-arriving-dimension race let a
  transaction commit referencing an account that hadn't been created yet
  (or, rarely, never was). See PM-003 / RULE-003 (warn-and-continue).

## Consumers

- `agent/tools.py`'s `query_warehouse` tool (ad hoc analyst/agent queries).
- `loyalty_accounts.points_balance` reconciliation (should always tie out
  to `sum(points_earned) - sum(points_redeemed)` grouped by `loyalty_id`
  from this table — see CONTRACT-001).

## Related

- `knowledge_base/contracts/CONTRACT-001.md` — `loyalty_accounts`, the
  dimension this table's `loyalty_id` should always resolve to.
- `knowledge_base/rules/RULE-001.md`, `RULE-002.md`, `RULE-003.md` — the
  three validation rules enforcing this contract's guarantees.
