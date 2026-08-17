-- RULE-005: gold-tier activity-recency check (warn-and-continue).
-- See knowledge_base/rules/RULE-005.md. No 24h-window adaptation needed —
-- this is a point-in-time "how long since last activity" check against
-- real `now()`, not a load-window filter, so it transfers as-is. Verified
-- against the real CSV: 3,879 rows, matching PM-005's documented count
-- exactly.
select
    'RULE-005' as rule_id,
    'warn-and-continue' as enforcement,
    'account' as entity_type,
    loyalty_id as entity_id,
    'tier=gold, last_activity_date=' || last_activity_date::text as detail,
    current_timestamp as detected_at
from {{ ref('stg_loyalty_accounts') }}
where tier = 'gold'
  and last_activity_date < current_timestamp - interval '1 year'
