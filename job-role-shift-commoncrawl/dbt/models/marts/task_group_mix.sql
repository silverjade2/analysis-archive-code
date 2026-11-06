-- 출처 × 회차 × 직무군 × 업무 묶음 비중. 묶음 안 업무가 하나라도 목록에 있는 공고의 비율
-- 직무군 단위로 읽는 근거는 이 표다. 정의와 evidence 는 task_mix 와 같다
with totals as (
    select p.source, p.period, p.role_family, count(*) as n_postings
    from {{ ref('stg_task1_labels') }} l
    join {{ ref('stg_task1_postings') }} p on p.doc_id = l.doc_id
    where not l.is_non_data and p.source = 'jobkko'
    group by 1, 2, 3
),

per_group as (
    select
        pt.source, pt.period, pt.role_family, x.task_group,
        count(distinct pt.doc_id)                                       as n_with_task,
        count(distinct case when pt.is_primary then pt.doc_id end)      as n_primary
    from {{ ref('int_posting_tasks') }} pt
    join {{ ref('task_taxonomy') }} x on x.task_id = pt.task_id
    where pt.source = 'jobkko'
    group by 1, 2, 3, 4
),

groups as (
    select distinct task_group from {{ ref('task_taxonomy') }}
),

jobkko as (
    select
        t.source, t.period, t.role_family, g.task_group,
        '라벨'                                      as basis,
        '라벨'                                      as evidence,
        t.n_postings,
        coalesce(c.n_with_task, 0) / t.n_postings   as posting_share,
        null::float                                 as posting_lo,
        null::float                                 as posting_hi,
        coalesce(c.n_primary, 0) / t.n_postings     as primary_share,
        null::float                                 as primary_lo,
        null::float                                 as primary_hi
    from totals t
    cross join groups g
    left join per_group c
      on c.source = t.source and c.period = t.period and c.role_family = t.role_family and c.task_group = g.task_group
),

wdc as (
    select
        'wdc'                                   as source,
        crawl                                   as period,
        role_family,
        task_group,
        '보정'                                  as basis,
        any_value(evidence)                     as evidence,
        any_value(n_pop)                        as n_postings,
        max(case when kind = 'tasks'   then share end)  as posting_share,
        max(case when kind = 'tasks'   then ci_lo end)  as posting_lo,
        max(case when kind = 'tasks'   then ci_hi end)  as posting_hi,
        max(case when kind = 'primary' then share end)  as primary_share,
        max(case when kind = 'primary' then ci_lo end)  as primary_lo,
        max(case when kind = 'primary' then ci_hi end)  as primary_hi
    from {{ ref('task_share_wdc') }}
    where unit_type = 'group'
    group by 1, 2, 3, 4
),

unioned as (
    select * from jobkko
    union all
    select * from wdc
)

select
    source || '|' || period || '|' || role_family || '|' || task_group as cell_key,
    *
from unioned
