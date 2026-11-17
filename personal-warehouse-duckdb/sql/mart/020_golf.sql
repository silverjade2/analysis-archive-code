-- 스코어 mart
-- golf_rounds        라운드 × 플레이어 요약 (일차 · 같은 날 몇 번째 라운드 · OUT/IN 합계 · par 대비 · 다 쳤는지)
-- golf_player_trend  플레이어 × 월: 18홀 다 친 라운드만 (앱의 평균과 같은 기준), 핸디 추정
-- golf_hole_stats    플레이어 × 골프장 × 홀: 평균 타수 · par 대비, 본인 전체 평균 대비, 4명 평균
CREATE OR REPLACE TABLE mart.golf_rounds AS
WITH rounds AS (
    SELECT
        r.*,
        dense_rank() OVER (ORDER BY r.round_date) AS day_no,
        row_number() OVER (PARTITION BY r.round_date ORDER BY r.created_at, r.round_id) AS round_of_day
    FROM stg.golf_rounds r
    WHERE NOT r.deleted AND r.round_date IS NOT NULL
),
per AS (
    SELECT
        round_id,
        player_id,
        any_value(player_name) AS player_name,
        count(*) AS holes_played,
        sum(strokes) AS strokes,
        sum(par) AS par,
        sum(strokes) FILTER (WHERE hole_no <= 9) AS out_strokes,
        sum(par) FILTER (WHERE hole_no <= 9) AS out_par,
        sum(strokes) FILTER (WHERE hole_no >= 10) AS in_strokes,
        sum(par) FILTER (WHERE hole_no >= 10) AS in_par
    FROM stg.golf_scores
    GROUP BY round_id, player_id
)
SELECT
    r.round_id,
    r.round_date,
    r.day_no,
    r.round_of_day,
    r.day_no || '일차' || CASE WHEN r.round_of_day > 1 THEN ' ' || r.round_of_day || 'R' ELSE '' END AS day_label,
    r.course_id,
    r.course_name,
    r.base_course,
    r.start_nine,
    r.holes,
    p.player_id,
    p.player_name,
    p.holes_played,
    p.holes_played = r.holes AS complete,
    p.strokes,
    p.par,
    p.strokes - p.par AS to_par,
    p.out_strokes,
    p.out_par,
    p.in_strokes,
    p.in_par
FROM rounds r
JOIN per p ON p.round_id = r.round_id;

-- 핸디 추정 (코스 레이팅 · 슬로프가 없어 WHS 공식 대신 근사): 그 달 말까지의 최근 20라운드 중 par 대비가 좋은 k개 평균
-- k = 라운드 수 20개 이상이면 8, 적으면 round(n × 0.4) (최소 1)
CREATE OR REPLACE TABLE mart.golf_player_trend AS
WITH r AS (
    SELECT * FROM mart.golf_rounds WHERE complete AND holes = 18
),
monthly AS (
    SELECT
        player_id,
        any_value(player_name) AS player_name,
        CAST(date_trunc('month', round_date) AS DATE) AS month,
        count(*) AS rounds,
        avg(strokes) AS avg_strokes,
        avg(to_par) AS avg_to_par,
        min(strokes) AS best_strokes
    FROM r
    GROUP BY player_id, CAST(date_trunc('month', round_date) AS DATE)
),
hist AS (
    SELECT
        m.player_id,
        m.month,
        r.to_par,
        row_number() OVER (PARTITION BY m.player_id, m.month ORDER BY r.round_date DESC, r.round_id DESC) AS recency
    FROM monthly m
    JOIN r ON r.player_id = m.player_id AND r.round_date < m.month + INTERVAL 1 MONTH
),
recent AS (
    SELECT
        player_id,
        month,
        to_par,
        row_number() OVER (PARTITION BY player_id, month ORDER BY to_par) AS rk,
        count(*) OVER (PARTITION BY player_id, month) AS n
    FROM hist
    WHERE recency <= 20
),
handicap AS (
    SELECT
        player_id,
        month,
        max(n) AS rounds_used,
        avg(to_par) FILTER (WHERE rk <= CASE WHEN n >= 20 THEN 8 ELSE greatest(1, CAST(round(n * 0.4) AS INTEGER)) END) AS handicap_est
    FROM recent
    GROUP BY player_id, month
)
SELECT
    m.player_id,
    m.player_name,
    m.month,
    m.rounds,
    round(m.avg_strokes, 1) AS avg_strokes,
    round(m.avg_to_par, 1) AS avg_to_par,
    m.best_strokes,
    h.rounds_used AS handicap_rounds,
    round(h.handicap_est, 1) AS handicap_est
FROM monthly m
JOIN handicap h ON h.player_id = m.player_id AND h.month = m.month;

CREATE OR REPLACE TABLE mart.golf_hole_stats AS
WITH by_hole AS (
    SELECT
        player_id,
        any_value(player_name) AS player_name,
        base_course,
        hole_no,
        any_value(par) AS par,
        count(*) AS plays,
        avg(strokes) AS avg_strokes,
        avg(to_par) AS avg_to_par,
        count(*) FILTER (WHERE to_par <= -1) AS birdie_or_better,
        count(*) FILTER (WHERE to_par = 0) AS pars,
        count(*) FILTER (WHERE to_par = 1) AS bogeys,
        count(*) FILTER (WHERE to_par >= 2) AS double_or_worse
    FROM stg.golf_scores
    GROUP BY player_id, base_course, hole_no
),
player_avg AS (
    SELECT player_id, base_course, avg(to_par) AS player_avg_to_par
    FROM stg.golf_scores
    GROUP BY player_id, base_course
),
field AS (
    SELECT base_course, hole_no, avg(to_par) AS field_avg_to_par
    FROM stg.golf_scores
    GROUP BY base_course, hole_no
)
SELECT
    h.player_id,
    h.player_name,
    h.base_course,
    h.hole_no,
    h.par,
    ch.hdcp,
    ch.yards_white,
    h.plays,
    round(h.avg_strokes, 2) AS avg_strokes,
    round(h.avg_to_par, 2) AS avg_to_par,
    round(pa.player_avg_to_par, 2) AS player_avg_to_par,
    round(h.avg_to_par - pa.player_avg_to_par, 2) AS vs_player_avg,
    round(f.field_avg_to_par, 2) AS field_avg_to_par,
    h.birdie_or_better,
    h.pars,
    h.bogeys,
    h.double_or_worse
FROM by_hole h
JOIN player_avg pa ON pa.player_id = h.player_id AND pa.base_course = h.base_course
JOIN field f ON f.base_course = h.base_course AND f.hole_no = h.hole_no
LEFT JOIN stg.golf_course_holes ch ON ch.base_course = h.base_course AND ch.hole_no = h.hole_no;
