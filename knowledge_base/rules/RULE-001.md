---
id: RULE-001
title: Earn-event idempotency check
enforcement: hard-stop
owning_team: loyalty-platform
table: raw.loyalty_transactions
originated_from: PM-001
tags: [idempotency, duplicate-events, points-balance]
---

## What it checks

For every load window, no `(loyalty_id, order_id)` pair may have more than
one row where `transaction_type = 'earn'` within a 24-hour span.

```sql
select loyalty_id, order_id, count(*) as earn_count
from raw.loyalty_transactions
where transaction_type = 'earn'
  and transaction_timestamp >= now() - interval '24 hours'
group by loyalty_id, order_id
having count(*) > 1;
```

Any row returned is a violation.

## Enforcement level: hard-stop, not warn

This blocks the ingestion job — it does not just log and continue. Rationale:
`points_balance` is a liability. A silent duplicate credit isn't a data
quality nuisance, it's real financial exposure the first time a customer
redeems the extra points. The cost of a delayed load is lower than the cost
of honoring a redemption that was never earned. Contrast this with
lower-stakes checks (e.g. a missing `customer_city` value) that are fine to
warn-and-continue on.

## Why it wasn't caught earlier

Before PM-001, `raw.loyalty_transactions` was a pure-append table with no
uniqueness constraint and no dedup step — the messaging layer's at-least-once
delivery guarantee was never matched with idempotent writes on the consumer
side. This rule exists because that mismatch produced 796 duplicate `earn`
rows over a six-day window before anyone noticed.

## What "passing" actually fixes vs. what it doesn't

This rule catches new duplicates going forward. It does not retroactively fix
historical data — see PM-001's backfill section for how existing duplicates
were reconciled. A rule like this should ideally exist before the first
duplicate ever lands, not after.

## Related

- `knowledge_base/postmortems/PM-001.md` — the incident that produced this
  rule.