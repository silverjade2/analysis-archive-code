-- GitHub 저장소 × 주 × 이벤트. 2025 하반기부터 GH Archive 수집이 끊겨 (reports/task2_first_look) 2025-03 까지만 남긴다
select
    week::date      as week,
    repo_name,
    event_type,
    skill,
    strength,
    events,
    actors
from {{ source('activity', 'github_weekly') }}
where week::date < '2025-04-01'
