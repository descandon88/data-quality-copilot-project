-- Light rename/cast layer over bronze.loyalty_accounts — no business logic
-- here (that lives in models/marts/rule_checks/). Downstream models
-- reference this staging model (dbt's ref function, model name
-- stg_loyalty_accounts) instead of the raw source directly, so a future
-- bronze schema change gets absorbed in one place. NOTE: do not write that
-- ref call using real double-curly-brace Jinja syntax in this comment —
-- dbt's dependency parser scans for ref/source calls via Jinja rendering,
-- not SQL comment-awareness, so real Jinja braces even inside a `--`
-- comment get counted as a real edge and previously caused a false
-- self-referencing cycle error on this exact model.
select
    loyalty_id,
    customer_unique_id,
    lower(trim(tier)) as tier,
    points_balance,
    enrollment_date,
    last_activity_date,
    email_opt_in
from {{ source('bronze', 'loyalty_accounts') }}
