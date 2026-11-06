-- 출처 × 회차 × 직무군 × 업무 유형 비중. 공고 하나에 업무가 여러 개라 posting_share 합은 1보다 크다
-- 국내: Sonnet 라벨 (경계 사례는 사람이 덮어씀) 을 그대로 센 값. 분모는 라벨이 붙은 데이터 직무 공고 수 (none 제외), 구간 없음
-- wdc: 전체 공고에 모델을 적용한 값을 평가 표본으로 보정한 추정값 + 95% 구간 (seed task_share_wdc). period 'ALL' = 세 회차 합
--   evidence = '참고용' 인 행 (데이터 엔지니어링, AI 엔지니어링 그룹, DS 밖의 모델링) 은 근거 수치로 쓰지 않는다
with totals as (
    select p.source, p.period, p.role_family, count(*) as n_postings
    from {{ ref('stg_task1_labels') }} l
    join {{ ref('stg_task1_postings') }} p on p.doc_id = l.doc_id
    where not l.is_non_data and p.source = 'jobkko'
    group by 1, 2, 3
),

counts as (
    select
        source, period, role_family, task_id,
        count(*)                               as n_with_task,
        count_if(is_primary)                   as n_primary
    from {{ ref('int_posting_tasks') }}
    where source = 'jobkko'
    group by 1, 2, 3, 4
),

jobkko as (
    select
        t.source, t.period, t.role_family, x.task_id, x.task, x.task_group,
        '라벨'                                  as basis,
        '라벨'                                  as evidence,
        t.n_postings,
        coalesce(c.n_with_task, 0) / t.n_postings   as posting_share,
        null::float                             as posting_lo,
        null::float                             as posting_hi,
        coalesce(c.n_primary, 0) / t.n_postings     as primary_share,
        null::float                             as primary_lo,
        null::float                             as primary_hi,
        null::float                             as model_posting_share
    from totals t
    cross join {{ ref('task_taxonomy') }} x
    left join counts c
      on c.source = t.source and c.period = t.period and c.role_family = t.role_family and c.task_id = x.task_id
),

wdc as (
    select
        'wdc'                                   as source,
        s.crawl                                 as period,
        s.role_family,
        s.unit                                  as task_id,
        any_value(x.task)                       as task,
        any_value(x.task_group)                 as task_group,
        '보정'                                  as basis,
        any_value(s.evidence)                   as evidence,
        any_value(s.n_pop)                      as n_postings,
        max(case when s.kind = 'tasks'   then s.share end)       as posting_share,
        max(case when s.kind = 'tasks'   then s.ci_lo end)       as posting_lo,
        max(case when s.kind = 'tasks'   then s.ci_hi end)       as posting_hi,
        max(case when s.kind = 'primary' then s.share end)       as primary_share,
        max(case when s.kind = 'primary' then s.ci_lo end)       as primary_lo,
        max(case when s.kind = 'primary' then s.ci_hi end)       as primary_hi,
        max(case when s.kind = 'tasks'   then s.model_share end) as model_posting_share
    from {{ ref('task_share_wdc') }} s
    join {{ ref('task_taxonomy') }} x on x.task_id = s.unit
    where s.unit_type = 'type'
    group by 1, 2, 3, 4
),

unioned as (
    select * from jobkko
    union all
    select * from wdc
)

select
    source || '|' || period || '|' || role_family || '|' || task_id as cell_key,
    *
from unioned
