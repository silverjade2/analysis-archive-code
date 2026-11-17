-- 주별 카테고리별 시간 (주 = 월요일 시작, Asia/Seoul)
-- 시간 일정: 주 경계를 넘는 일정은 주마다 겹친 시간만큼 나눠 더한다
-- 종일 일정: 시간 합계에 넣지 않고 날 수(all_day_days)로 따로 센다 (하루 24시간으로 세면 합계가 왜곡됨)
-- 미래 주 기준일은 마지막 수집일 (운영 코드는 current_date, 결과가 실행 날짜에 따라 바뀌지 않게)
CREATE OR REPLACE TABLE mart.time_by_category_weekly AS
WITH timed AS (
    SELECT
        category,
        start_local,
        end_local,
        unnest(generate_series(
            date_trunc('week', start_local),
            date_trunc('week', end_local - INTERVAL 1 MICROSECOND),
            INTERVAL 7 DAY
        )) AS week_start
    FROM stg.calendar_events
    WHERE NOT all_day AND end_local > start_local
),
timed_w AS (
    SELECT
        CAST(week_start AS DATE) AS week_start,
        category,
        sum(date_diff('second', greatest(start_local, week_start), least(end_local, week_start + INTERVAL 7 DAY)) / 3600.0) AS hours,
        count(*) AS events
    FROM timed
    GROUP BY 1, 2
),
allday AS (
    SELECT
        category,
        unnest(generate_series(start_local, end_local - INTERVAL 1 DAY, INTERVAL 1 DAY)) AS day
    FROM stg.calendar_events
    WHERE all_day AND end_local > start_local
),
allday_w AS (
    SELECT CAST(date_trunc('week', day) AS DATE) AS week_start, category, count(*) AS all_day_days
    FROM allday
    GROUP BY 1, 2
),
last_snap AS (
    SELECT max(_snapshot_date) AS d FROM raw.calendar_window
)
SELECT
    coalesce(t.week_start, a.week_start) AS week_start,
    coalesce(t.category, a.category) AS category,
    round(coalesce(t.hours, 0), 2) AS hours,
    coalesce(t.events, 0) AS timed_events,
    coalesce(a.all_day_days, 0) AS all_day_days,
    coalesce(t.week_start, a.week_start) > CAST(date_trunc('week', (SELECT d FROM last_snap)) AS DATE) AS is_future
FROM timed_w t
FULL OUTER JOIN allday_w a ON a.week_start = t.week_start AND a.category = t.category;
