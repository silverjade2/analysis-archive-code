-- 공고 한 건 = 한 행. 같은 공고가 여러 사이트에 올라온 것은 그룹 키로 묶는다
select
    posting_id,
    coalesce(final_group_id, group_id, posting_id)        as group_key,
    site,
    company,
    title,
    first_seen_date,
    is_active,
    lower(coalesce(title, '') || '\n' || coalesce(description, '')) as text_lower
from {{ source('jobkko', 'postings') }}
