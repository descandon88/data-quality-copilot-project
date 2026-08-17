-- RULE-003: loyalty transaction referential-integrity check (warn-and-continue).
-- See knowledge_base/rules/RULE-003.md. Same 24h-window adaptation as
-- rule_001_violations.sql — see that file's header comment for the full
-- rationale. Verified against the real CSV: 345 rows, matching PM-003's
-- documented count exactly. RULE-003.md itself notes these are "candidate"
-- violations, not confirmed permanent orphans (some are late-arriving
-- accounts that self-heal) — this mart still reports the raw candidate
-- count, same as the rule's own documented check; the self-heal triage is
-- explicitly a manual follow-up step per RULE-003.md, not something this
-- mart resolves automatically.
select
    'RULE-003' as rule_id,
    'warn-and-continue' as enforcement,
    'transaction' as entity_type,
    t.transaction_id as entity_id,
    'orphaned loyalty_id=' || t.loyalty_id || ', order_id=' || t.order_id as detail,
    current_timestamp as detected_at
from {{ ref('stg_loyalty_transactions') }} t
left join {{ ref('stg_loyalty_accounts') }} a on t.loyalty_id = a.loyalty_id
where a.loyalty_id is null
  and t.loyalty_id is not null
