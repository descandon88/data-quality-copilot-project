-- The actual point of this mart: one queryable table with every known
-- RULE-00X violation, instead of five raw SQL blocks copy-pasted out of
-- knowledge_base/rules/*.md every time someone (or the agent's
-- query_warehouse tool, or Phase 9's monitoring dashboard) needs the
-- current count. Each rule_checks/*.sql model is ephemeral (see
-- dbt_project.yml) — this is the only real table this mart writes.
--
-- Expected row counts after `dbt run`, verified against the real CSVs
-- before these models were written (see each rule_checks/*.sql for the
-- per-rule detail): RULE-001=787, RULE-002=1728, RULE-003=345,
-- RULE-004=480, RULE-005=3879. `select rule_id, count(*) from
-- {{ this }} group by rule_id` after a run should match those exactly —
-- if it doesn't, something about the bronze data changed since this was
-- written, not the mart logic.
select * from {{ ref('rule_001_violations') }}
union all
select * from {{ ref('rule_002_violations') }}
union all
select * from {{ ref('rule_003_violations') }}
union all
select * from {{ ref('rule_004_violations') }}
union all
select * from {{ ref('rule_005_violations') }}
