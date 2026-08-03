---
id: CONTRACT-001
title: loyalty_accounts data contract
owning_team: loyalty-platform
table: bronze.loyalty_accounts
primary_key: loyalty_id
write_disposition: merge
tags: [data-contract, loyalty-accounts, schema]
---

## What this table is

One row per enrolled loyalty member. Populated by the account-enrollment
write path (separate from `loyalty_transactions`, which is populated by the
purchase/redemption event path — see CONTRACT-002 for how the two relate).
Loaded into `bronze` via `ingestion/pipeline.py` with `write_disposition="merge"`
keyed on `loyalty_id`, so re-running the pipeline upserts rather than
re-appending.

## Schema

| Column | Type | Nullable | Meaning |
|---|---|---|---|
| `loyalty_id` | text | no | Primary key. Format `LOY-XXXXXXX`. Assigned at enrollment, never reused. |
| `customer_unique_id` | text | no | Foreign key to `bronze.olist_customers_dataset.customer_unique_id`. One loyalty account per unique customer. |
| `tier` | text | no | One of `bronze`, `silver`, `gold`. Reassessed nightly based on activity recency — see "Known failure modes" below. |
| `points_balance` | integer | no | Current redeemable point balance. **Must never be negative** — see RULE-004. |
| `enrollment_date` | timestamp | no | When the account was created. |
| `last_activity_date` | timestamp | no | Most recent account activity. Drives tier reassessment — an account whose `last_activity_date` ages past the retention window should be downgraded from `gold`, not left stale. |
| `email_opt_in` | boolean | no | Marketing consent flag. Not currently covered by any validation rule in this knowledge base. |

## Guarantees this table is supposed to provide

- Exactly one row per `customer_unique_id` that has ever enrolled.
- `points_balance` always reflects `sum(points_earned) - sum(points_redeemed)`
  from that account's rows in `loyalty_transactions`, and is never negative.
- `tier = 'gold'` implies `last_activity_date` is within the retention
  window (1 year) as of the most recent nightly reassessment run.

## Known failure modes (documented incidents)

- **Negative `points_balance`** — a redemption-path race condition can let
  two concurrent redemptions both pass a balance check based on the same
  stale read. See PM-004 / RULE-004 (hard-stop).
- **Stale `tier = 'gold'`** — the nightly reassessment job can silently
  stop reassessing (e.g. after an upstream schema change) while still
  reporting success, leaving inactive accounts at gold indefinitely. See
  PM-005 / RULE-005 (warn-and-continue).

## Consumers

- `agent/tools.py`'s `query_warehouse` tool (ad hoc analyst/agent queries).
- Downstream tier-based marketing segmentation and redemption-eligibility
  checks (not yet built as of this knowledge base's Phase 2/3 state — noted
  here as an intended consumer of the guarantees above, not a currently
  existing pipeline stage).

## Related

- `knowledge_base/contracts/CONTRACT-002.md` — `loyalty_transactions`, the
  other half of the loyalty data model, and the source of truth
  `points_balance` should reconcile against.
- `knowledge_base/rules/RULE-004.md`, `knowledge_base/rules/RULE-005.md` —
  the validation rules enforcing this contract's guarantees.
