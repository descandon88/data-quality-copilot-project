---
id: RULE-003
title: Loyalty transaction referential-integrity check
enforcement: warn-and-continue
owning_team: loyalty-platform
table: bronze.loyalty_transactions
originated_from: PM-003
tags: [late-arriving-dimension, orphaned-rows, referential-integrity]
---

## What it checks

For every load window, every `loyalty_id` in `bronze.loyalty_transactions`
should have a matching row in `bronze.loyalty_accounts`.

```sql
select t.transaction_id, t.loyalty_id, t.order_id, t.transaction_timestamp
from bronze.loyalty_transactions t
left join bronze.loyalty_accounts a on t.loyalty_id = a.loyalty_id
where a.loyalty_id is null
  and t.loyalty_id is not null
  and t.transaction_timestamp >= now() - interval '24 hours';
```

Any row returned is a candidate violation — see below on why "returned"
doesn't automatically mean "confirmed orphan."

## Enforcement level: warn-and-continue, not hard-stop

This does not block the ingestion job. Rationale: per PM-003, a meaningful
share of rows flagged by this check are not permanent orphans — they're
transactions that arrived slightly ahead of their account row in a benign
race, and resolve on their own once the account write lands (often within
the same load window or the next one). Hard-stopping ingestion on a check
that produces transient false positives would halt otherwise-healthy loads
over races that fix themselves. Contrast with RULE-001, where every
violation is a real, permanent duplicate with financial exposure attached —
there's no "wait and it resolves itself" case there.

## Why it wasn't caught earlier

Before PM-003, there was no scheduled referential-integrity check between
`loyalty_transactions` and `loyalty_accounts` at all — the two tables were
populated by independent write paths with no ordering guarantee and no
audit connecting them. The 345-row rate found in PM-003's ad hoc audit had
been accumulating unmeasured; this rule turns that one-time audit into a
recurring check specifically so the rate is tracked over time instead of
rediscovered by accident.

## What "passing" actually fixes vs. what it doesn't

This rule surfaces candidate orphans daily; it does not itself distinguish
a race-condition orphan (will self-heal) from a true orphan (account was
never created). That triage is a manual step per PM-003's resolution —
re-running this same query a day later against the same `transaction_id`s
tells you which bucket each row is actually in. A future improvement would
be to only flag rows that are still orphaned after, say, 48 hours, to cut
down on the expected-to-self-heal noise — not implemented yet.

## Related

- `knowledge_base/postmortems/PM-003.md` — the incident (well, ongoing
  condition) that produced this rule.
- `knowledge_base/rules/RULE-002.md` — related but distinct failure mode:
  RULE-002 catches missing join keys (null), this rule catches present but
  unmatched join keys (orphaned).
