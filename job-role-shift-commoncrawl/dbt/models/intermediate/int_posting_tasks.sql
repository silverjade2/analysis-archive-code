-- 공고 × 업무 유형: 라벨 목록에 들어 있으면 한 행. 데이터 직무가 아닌 공고 (none) 는 빼고, 대표 업무 여부를 남긴다
select
    p.doc_id,
    p.source,
    p.period,
    p.role_family,
    p.is_main,
    t.value::string                            as task_id,
    t.value::string = l.primary_task           as is_primary
from {{ ref('stg_task1_labels') }} l
join {{ ref('stg_task1_postings') }} p on p.doc_id = l.doc_id,
lateral flatten(input => l.tasks) t
where not l.is_non_data
