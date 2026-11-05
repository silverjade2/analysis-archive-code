-- 기술별 jobkko 공고 등장률 (공고 그룹 기준)
with total as (
    select count(distinct group_key) as n_groups from {{ ref('stg_jobkko_postings') }}
)
select
    s.skill,
    count(distinct ps.group_key)                              as groups_with_skill,
    t.n_groups,
    count(distinct ps.group_key) / nullif(t.n_groups, 0)      as posting_share
from {{ ref('skills') }} s
cross join total t
left join {{ ref('int_posting_skills') }} ps on ps.skill = s.skill
group by s.skill, t.n_groups
