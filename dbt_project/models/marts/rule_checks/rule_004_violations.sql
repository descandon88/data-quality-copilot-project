-- RULE-004: non-negative points balance check (hard-stop).
-- See knowledge_base/rules/RULE-004.md. No 24h-window adaptation needed —
-- RULE-004.md's own check is already a point-in-time snapshot query
-- (`points_balance < 0`), not a load-window filter, so it transfers as-is.
-- Verified against the real CSV: 480 rows, matching PM-004's documented
-- count exactly.
select
    'RULE-004' as rule_id,
    'hard-stop' as enforcement,
    'account' as entity_type,
    loyalty_id as entity_id,
    'points_balance=' || points_balance::text as detail,
    current_timestamp as detected_at
from {{ ref('stg_loyalty_accounts') }}
where points_balance < 0
