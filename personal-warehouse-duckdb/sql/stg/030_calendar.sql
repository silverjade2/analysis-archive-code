-- 캘린더 정제
-- 시간대: 시간 일정은 Asia/Seoul 로 통일, 종일 일정은 날짜 그대로 (end 는 그날을 포함하지 않는 끝)
-- snapshot 합치기: 각 날짜는 그 날짜를 조회 기간에 포함한 가장 최근 snapshot 의 값 (지운 일정이 옛 snapshot 에서 살아나지 않게)
-- snapshot 목록은 일정 행이 아니라 calendar_window 에서 (일정이 0건인 snapshot 도 목록에 남게)
-- 반복 일정은 수집 단계에서 이미 회차별로 전개됨. 카테고리 = 캘린더 이름, 취소(CANCELLED)는 뺀다
CREATE OR REPLACE TABLE stg.calendar_events AS
WITH occ AS (
    SELECT
        *,
        CASE WHEN all_day THEN start_date ELSE CAST(timezone('Asia/Seoul', start_at) AS DATE) END AS local_date
    FROM raw.calendar
),
snaps AS (
    SELECT DISTINCT _snapshot_date, window_start, window_end FROM raw.calendar_window
),
latest AS (
    SELECT o.*
    FROM occ o
    WHERE NOT EXISTS (
        SELECT 1 FROM snaps s
        WHERE s._snapshot_date > o._snapshot_date
          AND o.local_date >= s.window_start
          AND o.local_date < s.window_end
    )
    QUALIFY row_number() OVER (PARTITION BY o.calendar, o.occurrence_id ORDER BY o._snapshot_date DESC) = 1
)
SELECT
    l.calendar,
    coalesce(m.category, l.calendar) AS category,
    l.uid,
    l.occurrence_id,
    l.title,
    l.all_day,
    l.recurring,
    l.status,
    CASE WHEN l.all_day THEN CAST(l.start_date AS TIMESTAMP) ELSE timezone('Asia/Seoul', l.start_at) END AS start_local,
    CASE WHEN l.all_day THEN CAST(l.end_date AS TIMESTAMP) ELSE timezone('Asia/Seoul', l.end_at) END AS end_local,
    l.local_date,
    CASE WHEN NOT l.all_day THEN date_diff('second', l.start_at, l.end_at) / 3600.0 END AS hours,
    CASE WHEN l.all_day THEN date_diff('day', l.start_date, l.end_date) END AS days,
    l._snapshot_date AS snapshot_date
FROM latest l
LEFT JOIN raw.calendar_categories m ON m.calendar = l.calendar
WHERE l.status <> 'CANCELLED'
  AND NOT coalesce(m.excluded, false);
