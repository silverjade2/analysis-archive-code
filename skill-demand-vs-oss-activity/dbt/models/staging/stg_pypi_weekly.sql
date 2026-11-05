-- PyPI 패키지 × 주 다운로드. 한 기술에 패키지가 여럿이면 int 층에서 더한다
select
    week::date      as week,
    project         as package,
    skill,
    strength,
    downloads
from {{ source('activity', 'pypi_weekly') }}
