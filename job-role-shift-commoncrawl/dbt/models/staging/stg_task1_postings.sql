-- 라벨 묶음의 공고 (국내 + WDC 글로벌 2022-12, 2023-12, 2024-12). 공고 한 건 = 한 행
select
    doc_id,
    source,
    crawl                                      as period,
    role_family,
    is_main,
    title
from {{ source('task1', 'label_pool') }}
