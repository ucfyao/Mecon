
-- 训练集
with sand as (
    select sandwicher_address from flashbots.sandwiches
	where block_number >=13916166 and block_number < 14881677
    group by sandwicher_address limit 100
),
arb as (
    select account_address from flashbots.arbitrages
	where block_number >=13916166 and block_number < 14881677
    group by account_address limit 100
),
liq as (
    select liquidator_user from flashbots.liquidations
	where block_number >=13916166 and block_number < 14881677
    group by liquidator_user limit 100
),

-- 测试集
sand_test as (
    select sandwicher_address from flashbots.sandwiches
	where block_number >=14881677 and block_number < 16086234
    group by sandwicher_address limit 100
),
arb_test as (
    select account_address from flashbots.arbitrages
	where block_number >=14881677 and block_number < 16086234
    group by account_address limit 100
),
liq_test as (
    select liquidator_user from flashbots.liquidations
	where block_number >=14881677 and block_number < 16086234
    group by liquidator_user limit 100
)

-- select * from sand
-- select * from sand
select concat_ws(';',collect_list(sandwicher_address)) as addr
,'sand' as type
,'train' as data_type
from sand
union all
select concat_ws(';',collect_list(account_address)) as addr
,'arb' as type
,'train' as data_type
from arb
union all
select concat_ws(';',collect_list(liquidator_user)) as addr
,'liq' as type
,'train' as data_type
from liq
union all
select concat_ws(';',collect_list(sandwicher_address)) as addr
,'sand' as type
,'test' as data_type
from sand_test
union all
select concat_ws(';',collect_list(account_address)) as addr
,'arb' as type
,'test' as data_type
from arb_test
union all
select concat_ws(';',collect_list(liquidator_user)) as addr
,'liq' as type
,'test' as data_type
from liq_test




select concat_ws(';',collect_list(sandwicher_address)) as addr,'sand' as type from sand
union all
select concat_ws(';',collect_list(account_address)) as addr,'arb' as type from arb
union all
select concat_ws(';',collect_list(liquidator_user)) as addr,'liq' as type from liq