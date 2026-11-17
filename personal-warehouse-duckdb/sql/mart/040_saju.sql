-- 사주 × MBTI 응답 분포 (익명 집계)
-- share = 그 차원 안 비율, avg_score = overall_score 평균
-- mbti_pair 는 'MBTI>사주 MBTI', letters_matched 는 두 유형에서 같은 글자 수 (0~4)
CREATE OR REPLACE TABLE mart.saju_response_dist AS
SELECT
    snapshot_date,
    dimension,
    value,
    n,
    total,
    round(n / NULLIF(sum(n) OVER (PARTITION BY dimension), 0), 4) AS share,
    round(score_sum / NULLIF(n, 0), 1) AS avg_score,
    mapping_versions
FROM stg.saju_dist;
