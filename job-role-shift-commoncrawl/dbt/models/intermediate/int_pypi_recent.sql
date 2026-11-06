-- 기술별 최근 13주 PyPI 다운로드와 3년 전 같은 13주 대비 배수
{% set last_week = "(select max(week) from " ~ ref('stg_pypi_weekly') ~ ")" %}
with weekly as (
    select skill, week, sum(downloads) as downloads
    from {{ ref('stg_pypi_weekly') }}
    group by skill, week
)
select
    skill,
    sum(case when week >  dateadd(week, -13, {{ last_week }}) then downloads end)                          as downloads_13w,
    sum(case when week >  dateadd(week, -13 - 156, {{ last_week }})
              and week <= dateadd(week, -156, {{ last_week }}) then downloads end)                         as downloads_13w_3y_ago
from weekly
group by skill
