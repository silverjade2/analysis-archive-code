-- 기술 한 행: 국내 공고 등장률 (jobkko) · 글로벌 표본 등장률 (WDC) · PyPI 활동량과 3년 증가
-- 사분면: 수요 (공고 등장률) 와 활동량 (PyPI 다운로드) 이 각각 측정 기술 중앙값 이상인지
with base as (
    select
        s.skill,
        s.category,
        s.source,
        s.strength,
        d.posting_share,
        s.wdc_all                                                   as wdc_sample_share,
        a.downloads_13w,
        a.downloads_13w / nullif(a.downloads_13w_3y_ago, 0)          as growth_3y
    from {{ ref('skills') }} s
    left join {{ ref('int_skill_demand') }} d using (skill)
    left join {{ ref('int_pypi_recent') }}  a using (skill)
),
med as (
    select median(posting_share) as med_share, median(downloads_13w) as med_dl
    from base where downloads_13w is not null
)
select
    b.*,
    case
        when b.downloads_13w is null then '활동량 측정 안 함'
        when b.posting_share >= m.med_share and b.downloads_13w >= m.med_dl then '수요 높음 · 활동 높음'
        when b.posting_share >= m.med_share then '수요 높음 · 활동 낮음'
        when b.downloads_13w >= m.med_dl then '수요 낮음 · 활동 높음'
        else '수요 낮음 · 활동 낮음'
    end as quadrant
from base b
cross join med m
