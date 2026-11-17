"""ground truth 평가: 스코어 행이 빠진 이유와 타수 합, 캘린더 회차와 반복 판정, 설문 집계와 식별 값

truth_*.csv 는 여기서만 읽음. 설문은 원본 테이블에서 집계를 따로 다시 계산해 raw 와 대조
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
from common import DATA, DB, RAW, RES, SOURCE, TODAY, TZ

con = duckdb.connect(str(DB), read_only=True)
q = lambda sql: con.execute(sql).df()  # noqa: E731
numbers, checks = [], []


def num(key: str, value, note: str) -> None:
    numbers.append({"key": key, "value": value, "note": note})


def check(name: str, expected, actual) -> None:
    checks.append({"check": name, "expected": expected, "actual": actual, "ok": expected == actual})


# ---------- 스코어 시트 ----------
truth = pd.read_csv(DATA / "truth_golf_scores.csv")
rounds_t = pd.read_csv(DATA / "truth_golf_rounds.csv")
raw_scores = q("SELECT round_id, player_id, CAST(hole AS INTEGER) AS play_order, strokes FROM raw.golf_score_latest")
stg_rounds = q("SELECT round_id, deleted, holes FROM stg.golf_rounds")
stg_scores = q("SELECT round_id, player_id, play_order, strokes FROM stg.golf_scores")

# 수집, 정제 결과만으로 센 분해 (truth 없이 raw 와 stg 만)
m = raw_scores.merge(stg_rounds, on="round_id", how="left")
pipe_deleted = int(m["deleted"].sum())
pipe_blank = int((~m["deleted"] & (m["strokes"].fillna("").str.strip() == "")).sum())
reasons = truth["truth_reason"].value_counts()
check("점수 raw 행 = 시트 행", len(truth), len(raw_scores))
check("삭제 라운드 수", int(rounds_t["truth_deleted"].sum()), int(stg_rounds["deleted"].sum()))
check("삭제 라운드에 있던 행", int(reasons.get("deleted_round", 0)), pipe_deleted)
check("빈 칸 행 (살아 있는 라운드)", int(reasons.get("blank", 0)), pipe_blank)
check("stg 점수 행", int(reasons.get("kept", 0)), len(stg_scores))

kept = truth[truth["truth_reason"] == "kept"]
sheet_sum = int(kept["truth_strokes"].sum())
mart_sum = int(q("SELECT sum(strokes) AS s FROM mart.golf_rounds")["s"].iloc[0])
check("타수 합: 시트 = mart", sheet_sum, mart_sum)
key = ["round_id", "player_id", "play_order"]
j = kept.merge(stg_scores, on=key, how="outer", indicator=True)
check("행 단위 타수 불일치", 0, int(((j["_merge"] != "both") | (j["truth_strokes"] != j["strokes"])).sum()))

num("golf_score_rows_raw", len(raw_scores), "시트에서 받은 스코어 행")
num("golf_rounds", len(stg_rounds), "라운드 행")
num("golf_rounds_deleted", int(stg_rounds["deleted"].sum()), "삭제 표시된 라운드")
num("golf_rows_in_deleted_rounds", pipe_deleted, "삭제 라운드에 있던 스코어 행")
num("golf_rows_blank", pipe_blank, "살아 있는 라운드의 빈 칸")
num("golf_score_rows_stg", len(stg_scores), "stg 에 남은 스코어 행")
num("golf_strokes_sum", mart_sum, "mart 타수 합 (시트 합과 같음)")

# ---------- 캘린더 ----------
tz = ZoneInfo(TZ)
occ_t = pd.read_csv(DATA / "truth_calendar.csv")
rules = pd.read_csv(DATA / "calendar_rules.csv")
raw_cal = q("""
    SELECT calendar, uid, occurrence_id, all_day,
           CASE WHEN all_day THEN CAST(start_date AS TIMESTAMP) ELSE timezone('Asia/Seoul', start_at) END
               AS start_local,
           CASE WHEN all_day THEN CAST(end_date AS TIMESTAMP) ELSE timezone('Asia/Seoul', end_at) END AS end_local,
           status
    FROM raw.calendar_latest
""").merge(rules, on="occurrence_id")
occ_t["start_local"] = pd.to_datetime(occ_t["truth_start_local"], format="ISO8601")
occ_t["end_local"] = pd.to_datetime(occ_t["truth_end_local"], format="ISO8601")
j = occ_t.merge(raw_cal, on=["calendar", "uid", "start_local"], how="outer", indicator=True)
check("회차: truth 와 raw 가 다른 행", 0, int((j["_merge"] != "both").sum()))
both = j[j["_merge"] == "both"]
check("회차 끝 시각 불일치", 0, int((both["end_local_x"] != both["end_local_y"]).sum()))
check("반복 판정 (원본 VEVENT) 이 truth 와 다른 행", 0, int((both["by_vevent"] != both["truth_recurring"]).sum()))
check(
    "반복 판정 (회차 RECURRENCE-ID) 이 truth 와 다른 행",
    int((~occ_t["truth_recurring"]).sum()),
    int((both["by_occurrence"] != both["truth_recurring"]).sum()),
)
check(
    "취소 일정 (stg 에서 빠짐)",
    int((occ_t["truth_status"] == "CANCELLED").sum()),
    len(raw_cal) - int(q("SELECT count(*) AS n FROM stg.calendar_events")["n"].iloc[0]),
)

# 가상 일정은 주 경계를 넘지 않아 시작 시각의 주로 묶으면 됨
timed = occ_t[~occ_t["truth_all_day"] & (occ_t["truth_status"] != "CANCELLED")].copy()
timed["week_start"] = (timed["start_local"] - pd.to_timedelta(timed["start_local"].dt.weekday, unit="D")).dt.normalize()
timed["h"] = (timed["end_local"] - timed["start_local"]).dt.total_seconds() / 3600
truth_week = timed.groupby(["week_start", "calendar"])["h"].sum().round(2)
mart_week = q(
    "SELECT CAST(week_start AS TIMESTAMP) AS week_start, category AS calendar, hours FROM "
    "mart.time_by_category_weekly WHERE timed_events > 0"
).set_index(["week_start", "calendar"])["hours"]
check("주 × 카테고리 시간이 truth 와 다른 칸", 0, int((truth_week.sub(mart_week, fill_value=-1).abs() > 0.005).sum()))
window = q("SELECT * FROM raw.calendar_window_latest").iloc[0]
n_cal = int(window["calendars"])

num("calendar_calendars", n_cal, "캘린더 수")
num("calendar_with_events", raw_cal["calendar"].nunique(), "일정이 있는 캘린더")
num("calendar_occurrences", len(raw_cal), "조회 기간 안 회차 (raw)")
num("calendar_recurring_by_occurrence", int(raw_cal["by_occurrence"].sum()), "처음 판정: 회차에 RECURRENCE-ID")
num("calendar_recurring_by_vevent", int(raw_cal["by_vevent"].sum()), "고친 판정: 원본 VEVENT 에 RRULE, 회차 수정")
num("calendar_all_day", int(raw_cal["all_day"].sum()), "종일 회차")
num("calendar_window", f"{window['window_start']:%Y-%m-%d}~{window['window_end']:%Y-%m-%d}", "조회 기간, 끝은 미포함")

# ---------- 설문 ----------
subs = pd.read_csv(SOURCE / "saju" / "submissions.csv", keep_default_na=False)
live = subs[~subs["is_test"]]
raw_saju = q("SELECT * FROM raw.saju_dist_latest")


def month(s: str) -> str:
    return datetime.fromisoformat(s).astimezone(tz).strftime("%Y-%m")


def matched(a: str, b: str) -> str:
    return "(없음)" if len(a) != 4 or len(b) != 4 else str(sum(x == y for x, y in zip(a, b)))


mbti = live["mbti"].replace("", "(없음)")
expect = (
    pd.concat(
        [
            pd.DataFrame({"dimension": d, "value": v})
            for d, v in [
                ("mbti", mbti),
                ("saju_mbti", live["saju_mbti"].replace("", "(없음)")),
                ("mbti_pair", mbti + ">" + live["saju_mbti"]),
                ("letters_matched", [matched(a, b) for a, b in zip(live["mbti"], live["saju_mbti"])]),
                ("day_master", live["day_master"]),
                ("score_band", live["overall_score"].map(lambda s: f"{s // 10 * 10:02d}~{s // 10 * 10 + 9:02d}")),
                ("hour_known", live["hour_known"].map({True: "시간 앎", False: "시간 모름"})),
                ("month", live["created_at"].map(month)),
            ]
        ]
    )
    .value_counts()
    .rename("n_expect")
    .reset_index()
)
j = expect.merge(raw_saju[["dimension", "value", "n"]], on=["dimension", "value"], how="outer")
check("집계 응답 수 = 테스트 뺀 응답", len(live), int(raw_saju["total"].iloc[0]))
check("(차원, 값) 건수 불일치", 0, int((j["n_expect"] != j["n"]).sum()))
ident = set()
for c in [
    "id",
    "slug",
    "nickname",
    "birth_date",
    "birth_time",
    "corrected_datetime",
    "pillars",
    "referrer_id",
    "user_agent",
]:
    ident |= set(subs[c].astype(str)) - {""}
check("raw 에 식별 값이 들어간 칸", 0, int(raw_saju["value"].isin(ident).sum()))
check(
    "raw 컬럼 = 선언한 컬럼",
    "dimension,value,n,score_sum,total,mapping_versions,_snapshot_date,_collected_at",
    ",".join(raw_saju.columns),
)

num("saju_responses", int(raw_saju["total"].iloc[0]), "집계에 들어간 응답 (테스트 제외)")
num("saju_test_excluded", int(subs["is_test"].sum()), "view 가 뺀 테스트 응답")
num("saju_agg_rows", len(raw_saju), "저장한 집계 행")
num("saju_dimensions", raw_saju["dimension"].nunique(), "집계 차원")

# ---------- 호출 ----------
calls = pd.read_csv(RES / "calls.csv").groupby("source")["calls"].sum()
check("캘린더 호출 = 2 + 캘린더 수", 2 + n_cal, int(calls["calendar"]))
num("calls_calendar", int(calls["calendar"]), f"캘린더 수집 1회 호출 (캘린더 {n_cal}개)")
num("calls_golf", int(calls["golf"]), "스코어 시트 수집 1회 호출")
num("calls_saju", int(calls["saju"]), "설문 수집 1회 호출")
num("snapshot_date", TODAY.isoformat(), "첫 수집일")

files = sorted(p.relative_to(RAW).as_posix() for p in RAW.glob("*/*.parquet"))
num("raw_files", len(files), "raw parquet 파일 (소스마다 1일치)")

pd.DataFrame(numbers).to_csv(RES / "numbers.csv", index=False)
pd.DataFrame(checks).to_csv(RES / "eval_checks.csv", index=False)
for c in checks:
    print(f"{'OK ' if c['ok'] else 'NG '} {c['check']}: {c['actual']} (기대 {c['expected']})")
print()
for n in numbers:
    print(f"{n['key']:36s} {n['value']}")
assert all(c["ok"] for c in checks), "truth 와 다른 항목이 있음"
