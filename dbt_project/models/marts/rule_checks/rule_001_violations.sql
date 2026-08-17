-- RULE-001: earn-event idempotency check (hard-stop).
-- See knowledge_base/rules/RULE-001.md for the full rationale — this is
-- the same check, adapted for a static dataset. RULE-001.md's own SQL
-- filters to `transaction_timestamp >= now() - interval '24 hours'`,
-- written for a live daily load window. This project's data is a static
-- synthetic snapshot generated once (scripts/generate_loyalty_data.py),
-- not a continuously-arriving stream — a 24-hour filter against `now()`
-- would just return zero rows forever, since every transaction_timestamp
-- is dated whenever the snapshot was generated, not "recently". Every
-- rule_checks/*.sql model in this mart drops that filter for the same
-- reason and instead scans the full table — this materializes ALL known
-- violations in the dataset, not just a rolling window's worth.
-- Verified against the real data/raw/loyalty_transactions.csv in a
-- throwaway duckdb check before writing this model: 787 violating
-- (loyalty_id, order_id) groups, matching PM-001's documented count
-- exactly (PM-001.md was corrected from a stale 796 to 787 as part of
-- this same check — see SESSION_HANDOFF.md).
with duplicates as (
    select
        loyalty_id,
        order_id,
        count(*) as earn_count
    from {{ ref('stg_loyalty_transactions') }}
    where transaction_type = 'earn'
    group by loyalty_id, order_id
    having count(*) > 1
)
select
    'RULE-001' as rule_id,
    'hard-stop' as enforcement,
    'transaction' as entity_type,
    coalesce(loyalty_id, '(null)') || ':' || order_id as entity_id,
    'earn_count=' || earn_count::text as detail,
    current_timestamp as detected_at
from duplicates
