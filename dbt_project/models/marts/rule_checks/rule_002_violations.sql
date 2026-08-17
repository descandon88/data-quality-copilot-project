-- RULE-002: loyalty transaction join-key completeness check (warn-and-continue).
-- See knowledge_base/rules/RULE-002.md. Same 24h-window adaptation as
-- rule_001_violations.sql — see that file's header comment for the full
-- rationale. Verified against the real CSV: 1,728 rows, matching PM-002's
-- documented count exactly.
select
    'RULE-002' as rule_id,
    'warn-and-continue' as enforcement,
    'transaction' as entity_type,
    transaction_id as entity_id,
    'loyalty_id is null, order_id=' || coalesce(order_id, '(null)') as detail,
    current_timestamp as detected_at
from {{ ref('stg_loyalty_transactions') }}
where loyalty_id is null
