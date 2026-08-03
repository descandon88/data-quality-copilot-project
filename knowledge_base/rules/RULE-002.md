---
id: RULE-002
title: Loyalty transaction join-key completeness check
enforcement: warn-and-continue
owning_team: loyalty-platform
table: bronze.loyalty_transactions
originated_from: PM-002
tags: [schema-drift, null-keys, silent-failure]
---

## What it checks

For every load window, no row in `bronze.loyalty_transactions` should have a
null `loyalty_id`.

```sql
select transaction_id, order_id, transaction_timestamp
from bronze.loyalty_transactions
where loyalty_id is null
  and transaction_timestamp >= now() - interval '24 hours';
```

Any row returned is a violation.

## Enforcement level: warn-and-continue, not hard-stop

Unlike RULE-001, this does not block the ingestion job. Rationale: a null
`loyalty_id` makes a transaction invisible to joins, which is a real data
quality problem, but it is not a financial liability the way an
over-credited point balance is (RULE-001) — no one can redeem points they
were never credited, and the row itself is otherwise intact and
recoverable via `order_id`. Blocking the whole day's load over rows that
can be backfilled after the fact would trade a contained problem for a
bigger one (every other row in that batch also getting delayed). Contrast
with RULE-001, where the cost of letting a violation through was real
money; here the cost of letting it through is a bounded, backfillable gap.

## Why it wasn't caught earlier

Before PM-002, the events consumer had no inbound payload validation — it
mapped whatever fields were present in an event and let missing fields
pass through as null rather than rejecting or flagging the event. This rule
exists because an unversioned upstream payload change went undetected for
four days and produced 1,728 unjoinable transaction rows before a manual
spot-check happened to notice a discrepancy between two counts that should
have matched.

## What "passing" actually fixes vs. what it doesn't

This rule catches new null-key rows going forward, on a daily cadence. It
does not retroactively fix historical data — see PM-002's backfill section
for how the affected rows were rejoined via `order_id`. It also does not
protect against a *different* field going null; it's a targeted check for
this one join key, not a general schema-completeness monitor.

## Related

- `knowledge_base/postmortems/PM-002.md` — the incident that produced this
  rule.
- `knowledge_base/rules/RULE-001.md` — contrast case for enforcement-level
  reasoning (hard-stop vs warn-and-continue).
