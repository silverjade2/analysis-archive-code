-- 사주 × MBTI 응답 분포: 최신 snapshot 만 (raw 가 이미 익명 집계라 행 단위 데이터는 없다)
CREATE OR REPLACE TABLE stg.saju_dist AS
SELECT
    _snapshot_date AS snapshot_date,
    dimension,
    value,
    n,
    score_sum,
    total,
    mapping_versions
FROM raw.saju_dist_latest;
