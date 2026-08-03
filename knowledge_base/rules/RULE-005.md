---
id: RULE-005
title: Gold-tier activity-recency check
enforcement: warn-and-continue
owning_team: loyalty-platform
table: bronze.loyalty_accounts
originated_from: PM-005
tags: [silent-job-failure, tier-reassessment, cost-leak]
---

## What it checks

No account with `tier = 'gold'` should have a `last_activity_date` older
than the retention window (1 year).

```sql
select loyalty_id, tier, last_activity_date
from bronze.loyalty_accounts
where tier = 'gold'
  and last_activity_date < now() - interval '1 year';
```

Any row returned is a violation.

## Enforcement level: warn-and-continue, not hard-stop

This does not block ingestion. Rationale: an inactive gold account is a
cost leak (subsidized perks going to a customer who's stopped engaging),
not a financial-ledger error the way a negative balance or a duplicate
credit is — nothing here is being incorrectly redeemed or double-counted,
it's an ongoing but bounded and correctable operating cost. Halting the
load pipeline over stale tier assignments would be disproportionate to the
actual risk; the right response is flagging the accounts for the
reassessment job to correct, not stopping ingestion until it's fixed.
Contrast with RULE-001/RULE-004, where the underlying issue is money
already (mis)moved.

## Why it wasn't caught earlier

Before PM-005, the nightly tier-reassessment job's health was monitored by
process exit code only — "did it complete without error" — not by whether
it actually touched the accounts it was supposed to reassess. A schema
change upstream caused the job's core query to silently match zero rows;
the job kept reporting success while doing nothing, for an unmeasured
period before an unrelated account-health audit noticed the tier
distribution didn't match activity recency. This rule exists to catch the
*symptom* (stale gold accounts accumulating) independently of whether the
reassessment job itself is healthy, so a future silent job failure is
caught by this check even if the job's own monitoring is fooled again the
same way.

## What "passing" actually fixes vs. what it doesn't

This rule surfaces stale gold accounts daily; it does not fix the
reassessment job if it breaks again — it's a downstream tripwire, not a
replacement for PM-005's fix to the job's own success criteria (touched-row-count
validation, not just exit code). Both layers exist because either one
alone would have let this recur: the job's own health check can be fooled
by a query that matches nothing, and this rule alone would only ever tell
you *after* the leak had already accumulated for a day.

## Related

- `knowledge_base/postmortems/PM-005.md` — the incident that produced this
  rule.
