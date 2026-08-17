-- Light rename/cast layer over bronze.loyalty_transactions. Deliberately
-- does NOT filter out or coalesce null loyalty_id / orphaned rows — those
-- are exactly what rule_002_violations.sql and rule_003_violations.sql
-- need to see. Cleaning them here would hide the incidents this project's
-- whole rule_violations mart exists to surface.
select
    transaction_id,
    loyalty_id,
    order_id,
    points_earned,
    points_redeemed,
    lower(trim(transaction_type)) as transaction_type,
    transaction_timestamp
from {{ source('bronze', 'loyalty_transactions') }}
