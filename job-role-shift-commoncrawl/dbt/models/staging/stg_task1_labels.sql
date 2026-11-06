-- 공고마다 업무 유형 라벨 (경계 사례는 사람이 덮어씀). tasks 는 JSON 문자열이라 배열로 바꾼다
select
    doc_id,
    primary_task,
    primary_task = 'none'                      as is_non_data,
    parse_json(tasks)::array                   as tasks,
    agent_primary,
    override                                   as override_reason
from {{ source('task1', 'agent_labels') }}
