-- 스코어 시트 정제
-- 시트는 '현재 상태' 원장이라 최신 snapshot 만 쓴다 (날짜별 snapshot 을 합치면 같은 행이 날짜 수만큼 겹침)
-- 같은 id 가 여러 행이면 updatedAt 이 최신인 행 (앱은 id 기준 upsert 라 원래 1행)
-- 플레이어 이름: updatedAt 이 비어 있으면 아직 바꾸지 않은 기본 이름
CREATE OR REPLACE TABLE stg.golf_players AS
WITH p AS (
    SELECT * FROM raw.golf_player_latest
    QUALIFY row_number() OVER (PARTITION BY id ORDER BY updated_at DESC, sheet_row DESC) = 1
),
defaults(player_id, default_name, player_order) AS (
    VALUES ('p1', '선수1', 1), ('p2', '선수2', 2), ('p3', '선수3', 3), ('p4', '선수4', 4)
)
SELECT
    d.player_id,
    coalesce(
        CASE WHEN nullif(trim(p.updated_at), '') IS NOT NULL THEN nullif(regexp_replace(trim(p.name), '\s+', ' ', 'g'), '') END,
        d.default_name
    ) AS player_name,
    d.player_order
FROM defaults d
LEFT JOIN p ON p.id = d.player_id;

-- 코스: 기준 골프장은 출발(OUT · IN)과 홀 수(18 · 9)마다 코스 id 가 따로 있다 -> 기준 골프장 · 출발 · 홀 수로 나눈다
-- pars 는 '친 순서' 18개 (IN 출발이면 10번 홀 par 가 첫 칸). 형식이 어긋나면 앱처럼 전부 4
CREATE OR REPLACE TABLE stg.golf_courses AS
WITH c AS (
    SELECT *, list_transform(string_split(coalesce(pars, ''), ','), lambda x: TRY_CAST(trim(x) AS INTEGER)) AS pars_raw
    FROM raw.golf_course_latest
    QUALIFY row_number() OVER (PARTITION BY id ORDER BY coalesce(nullif(updated_at, ''), created_at) DESC, sheet_row DESC) = 1
),
home(course_id, home_name, start_nine, holes, first_hole) AS (
    VALUES
        ('c_home_out', 'OUT 출발 · Maple Ridge CC', 'OUT', 18, 1),
        ('c_home_in', 'IN 출발 · Maple Ridge CC', 'IN', 18, 10),
        ('c_home_out9', 'OUT 9홀 · Maple Ridge CC', 'OUT', 9, 1),
        ('c_home_in9', 'IN 9홀 · Maple Ridge CC', 'IN', 9, 10)
)
SELECT
    c.id AS course_id,
    coalesce(h.home_name, regexp_replace(trim(c.name), '\s+', ' ', 'g')) AS course_name,
    CASE WHEN h.course_id IS NOT NULL THEN 'Maple Ridge CC' ELSE regexp_replace(trim(c.name), '\s+', ' ', 'g') END AS base_course,
    h.start_nine,
    coalesce(h.holes, 18) AS holes,
    coalesce(h.first_hole, 1) AS first_hole,
    h.course_id IS NOT NULL AS is_home,
    CASE
        WHEN len(c.pars_raw) = 18 AND coalesce(list_aggregate(list_transform(c.pars_raw, lambda x: x BETWEEN 3 AND 6), 'bool_and'), false)
            THEN c.pars_raw
        ELSE list_transform(range(18), lambda x: 4)
    END AS pars,
    nullif(trim(c.deleted_at), '') IS NOT NULL AS deleted
FROM c
LEFT JOIN home h ON h.course_id = c.id;

CREATE OR REPLACE TABLE stg.golf_rounds AS
WITH r AS (
    SELECT * FROM raw.golf_round_latest
    QUALIFY row_number() OVER (PARTITION BY id ORDER BY coalesce(nullif(updated_at, ''), created_at) DESC, sheet_row DESC) = 1
)
SELECT
    r.id AS round_id,
    TRY_CAST(r.date AS DATE) AS round_date,
    r.course_id,
    nullif(trim(r.memo), '') AS memo,
    nullif(trim(r.created_at), '') AS created_at,
    nullif(trim(r.updated_at), '') AS updated_at,
    nullif(trim(r.deleted_at), '') IS NOT NULL AS deleted,
    coalesce(c.course_name, '코스 미정') AS course_name,
    coalesce(c.base_course, '코스 미정') AS base_course,
    c.start_nine,
    coalesce(c.holes, 18) AS holes,
    coalesce(c.first_hole, 1) AS first_hole,
    coalesce(c.is_home, false) AS is_home,
    coalesce(c.pars, list_transform(range(18), lambda x: 4)) AS pars
FROM r
LEFT JOIN stg.golf_courses c ON c.course_id = r.course_id;

-- 점수: 시트의 hole 은 '친 순서'(1~18) -> 실제 홀 번호 = (첫 홀 - 1 + 친 순서 - 1) % 18 + 1
-- 빈 칸(strokes 없음) · 삭제된 라운드 · 9홀 라운드의 10번째 칸 이후는 뺀다
CREATE OR REPLACE TABLE stg.golf_scores AS
WITH s AS (
    SELECT *, TRY_CAST(hole AS INTEGER) AS play_order, TRY_CAST(nullif(trim(strokes), '') AS INTEGER) AS strokes_n
    FROM raw.golf_score_latest
    QUALIFY row_number() OVER (PARTITION BY id ORDER BY updated_at DESC, sheet_row DESC) = 1
)
SELECT
    s.round_id,
    r.round_date,
    r.course_id,
    r.course_name,
    r.base_course,
    r.start_nine,
    r.holes,
    s.player_id,
    p.player_name,
    s.play_order,
    (r.first_hole - 1 + s.play_order - 1) % 18 + 1 AS hole_no,
    s.strokes_n AS strokes,
    r.pars[s.play_order] AS par,
    s.strokes_n - r.pars[s.play_order] AS to_par
FROM s
JOIN stg.golf_rounds r ON r.round_id = s.round_id
JOIN stg.golf_players p ON p.player_id = s.player_id
WHERE NOT r.deleted
  AND s.strokes_n IS NOT NULL
  AND s.play_order BETWEEN 1 AND r.holes;
