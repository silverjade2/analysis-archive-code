-- 공고 그룹 × 기술: 확장 사전 (seed skills) 의 정규식이 제목 + 본문에 한 번이라도 나오면 1
-- Snowflake 의 REGEXP_LIKE 는 문자열 전체가 맞아야 하므로 REGEXP_INSTR > 0 으로 "포함" 을 본다
with postings as (
    select group_key, max_by(text_lower, length(text_lower)) as text_lower   -- 같은 그룹은 본문이 가장 긴 것 하나
    from {{ ref('stg_jobkko_postings') }}
    group by group_key
)
select
    p.group_key,
    s.skill,
    s.category
from postings p
join {{ ref('skills') }} s
  on regexp_instr(p.text_lower, s.pattern) > 0
