---
id: RULE-004
title: Non-negative points balance check
enforcement: hard-stop
owning_team: loyalty-platform
table: bronze.loyalty_accounts
originated_from: PM-004
tags: [race-condition, points-balance, financial-liability]
---

## What it checks

No account in `bronze.loyalty_accounts` may have a `points_balance` below
zero.

```sql
select loyalty_id, points_balance, last_activity_date
from bronze.loyalty_accounts
where points_balance < 0;
```

Any row returned is a violation.

## Enforcement level: hard-stop, not warn

This blocks the ingestion/reconciliation job — same enforcement level as
RULE-001, and for the same underlying reason. A negative balance means a
redemption was honored that the account couldn't actually afford: that's
real financial exposure the moment it happens, not a data-completeness
nuisance to clean up later. The cost of a delayed load is lower than the
cost of continuing to serve account state that's already proven a
redemption-path race is live. Contrast with RULE-002 and RULE-003, where
the underlying rows are recoverable and non-liability-bearing — this rule
sits with RULE-001, not with those.

## Why it wasn't caught earlier

Before PM-004, the redemption path used separate read-check-write steps
with no atomicity guarantee, so two concurrent redemption requests against
the same account could both pass a balance check based on the same
pre-decrement value. There was also no scheduled `points_balance < 0`
check running at all — unlike PM-001, which required a nontrivial
reconciliation query to catch, this condition is directly visible with a
single-column filter, and simply wasn't being checked before this rule
existed.

## What "passing" actually fixes vs. what it doesn't

This rule catches new negative balances going forward, daily. It does not
by itself distinguish "this negative balance came from the race condition
that's now been fixed" from "this negative balance indicates ongoing
abuse of a redemption path" — that triage is a manual step, per PM-004's
resolution, of reconstructing the true balance from transaction history
before deciding whether to correct or escalate.

## Related

- `knowledge_base/postmortems/PM-004.md` — the incident that produced this
  rule.
- `knowledge_base/rules/RULE-001.md` — the other hard-stop rule in this
  knowledge base, and the reference case for why financial-liability checks
  get this enforcement level.
