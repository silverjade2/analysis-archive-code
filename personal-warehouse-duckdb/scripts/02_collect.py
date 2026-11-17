"""수집: 가짜 소스 셋을 원래 수집기와 같은 순서로 읽고 schema 를 맞춰 raw parquet 로 저장

외부 호출은 calls.jsonl 에 한 줄씩. 소스 응답 형식만 흉내 내고 호출 횟수와 순서는 원래 수집기 그대로
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import duckdb
import icalendar
import pandas as pd
import recurring_ical_events
from common import (
    CALLS,
    DATA,
    GOLF_TABS,
    LOOKAHEAD_DAYS,
    LOOKBACK_DAYS,
    RES,
    SOURCE,
    TODAY,
    TZ,
    CallLog,
    Schema,
    save_snapshot,
)

CALLS.unlink(missing_ok=True)
calls = CallLog()
RES.mkdir(parents=True, exist_ok=True)
saved = []


def save(source: str, table: str, df: pd.DataFrame, schema: Schema) -> None:
    saved.append({"source": source, "table": table, "rows": save_snapshot(table, df, schema)})


# ---------- 스코어 시트 ----------
def snake(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in name)


def tab_schema(tab: str) -> Schema:
    return Schema({"sheet_row": "BIGINT", **{snake(c): "VARCHAR" for c in GOLF_TABS[tab]}}, ("sheet_row", "id"))


def rows_to_frame(tab: str, rows: list[list[str]]) -> pd.DataFrame:
    """시트 2행부터의 값 -> DataFrame. 끝 빈 칸이 잘려 온 행은 채우고 id 가 빈 행은 버림. 타입 변환은 stg 에서"""
    cols = GOLF_TABS[tab]
    out = []
    for i, r in enumerate(rows):
        r = (list(r) + [""] * len(cols))[: len(cols)]
        if not str(r[0]).strip():
            continue
        out.append({"sheet_row": i + 2, **{snake(c): str(v) for c, v in zip(cols, r)}})
    return pd.DataFrame(out, columns=list(tab_schema(tab).columns))


def read_sheets() -> dict[str, list[list[str]]]:
    """읽기 전용 범위 토큰 1 + 탭 목록 1 + batchGet 1. 탭 4개가 응답 하나를 나눠 씀"""
    doc = json.loads((SOURCE / "golf" / "spreadsheet.json").read_text(encoding="utf-8"))
    with calls.call("golf", "POST oauth2 token"):
        pass
    with calls.call("golf", "GET spreadsheets (탭 목록)"):
        present = doc["tabs"]
    tabs = [t for t in GOLF_TABS if t in present]
    with calls.call("golf", "GET values:batchGet"):
        ranges = [doc["values"][t] for t in tabs]
    return {t: v for t, v in zip(tabs, ranges)}


sheet = read_sheets()
for tab in GOLF_TABS:
    save("golf", f"golf_{tab.lower()}", rows_to_frame(tab, sheet.get(tab, [])), tab_schema(tab))

# ---------- 캘린더 ----------
CAL_SCHEMA = Schema(
    {
        "calendar": "VARCHAR",
        "uid": "VARCHAR",
        "occurrence_id": "VARCHAR",  # uid | 시작 시각
        "title": "VARCHAR",
        "all_day": "BOOLEAN",
        "start_at": "TIMESTAMPTZ",  # 시간 일정만
        "end_at": "TIMESTAMPTZ",
        "start_date": "DATE",  # 종일 일정만. end_date 는 그날을 포함하지 않는 끝
        "end_date": "DATE",
        "recurring": "BOOLEAN",
        "rrule": "VARCHAR",
        "status": "VARCHAR",
        "tzid": "VARCHAR",
        "window_start": "DATE",
        "window_end": "DATE",
    },
    ("calendar", "uid", "occurrence_id", "window_start", "window_end"),
)
WINDOW_SCHEMA = Schema(
    {"window_start": "DATE", "window_end": "DATE", "calendars": "BIGINT", "occurrences": "BIGINT"},
    ("window_start", "window_end"),
)


def as_datetime(value, tz: ZoneInfo) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=tz)  # 시간대 없는 값은 기준 시간대로
    return datetime.combine(value, time(0), tz)


def expand_ics(texts: list[str], calendar: str, start, end, tz: ZoneInfo) -> list[dict]:
    """VCALENDAR 문자열들 -> 조회 기간 [start, end) 안의 회차 1건 = 1행

    반복 여부는 원본 VEVENT 의 RRULE, RDATE, RECURRENCE-ID 로 판정. 전개 결과에는 모든 회차에 RECURRENCE-ID 가 붙어
    처음 판정(회차에 RECURRENCE-ID 가 있으면 반복)은 by_occurrence 로 따로 남김
    """
    rows = []
    for text in texts:
        cal = icalendar.Calendar.from_ical(text)
        masters, overridden = {}, set()
        for comp in cal.walk("VEVENT"):
            uid = str(comp.get("UID", ""))
            if "RRULE" in comp or "RDATE" in comp:
                masters[uid] = comp.get("RRULE").to_ical().decode() if comp.get("RRULE") else "RDATE"
            if "RECURRENCE-ID" in comp:
                overridden.add(uid)
        for occ in recurring_ical_events.of(cal).between(start, end):
            uid = str(occ.get("UID", ""))
            ds = occ["DTSTART"].dt
            de = occ["DTEND"].dt if occ.get("DTEND") else None
            all_day = not isinstance(ds, datetime)
            if de is None:
                dur = occ.get("DURATION")
                de = ds + (dur.dt if dur else (timedelta(days=1) if all_day else timedelta(0)))
            tzid = occ["DTSTART"].params.get("TZID") or (
                str(ds.tzinfo) if isinstance(ds, datetime) and ds.tzinfo else ""
            )
            row = {
                "calendar": calendar,
                "uid": uid,
                "title": str(occ.get("SUMMARY", "")).strip() or "(제목 없음)",
                "all_day": all_day,
                "recurring": uid in masters or uid in overridden,
                "rrule": masters.get(uid),
                "status": str(occ.get("STATUS", "CONFIRMED")).upper(),
                "tzid": tzid,
                "window_start": start,
                "window_end": end,
                "by_occurrence": "RECURRENCE-ID" in occ,
            }
            if all_day:
                row.update(
                    start_date=ds,
                    end_date=de if not isinstance(de, datetime) else de.date(),
                    start_at=None,
                    end_at=None,
                )
                key = ds.isoformat()
            else:
                s, e = as_datetime(ds, tz), as_datetime(de, tz)
                row.update(
                    start_at=s.astimezone(timezone.utc),
                    end_at=e.astimezone(timezone.utc),
                    start_date=None,
                    end_date=None,
                )
                key = s.astimezone(timezone.utc).isoformat()
            row["occurrence_id"] = f"{uid}|{key}"
            rows.append(row)
    return rows


def read_caldav() -> dict[str, list[str]]:
    """사용자 정보 1 + 캘린더 목록 1 + 캘린더마다 조회 1

    가짜 서버는 기간 조건 없이 전부 돌려주고 전개 단계가 기간으로 자름
    """
    doc = json.loads((SOURCE / "calendar" / "server.json").read_text(encoding="utf-8"))
    with calls.call("calendar", "PROPFIND principal"):
        pass
    with calls.call("calendar", "PROPFIND calendars"):
        calendars = doc["calendars"]
    out = {}
    for c in calendars:
        with calls.call("calendar", "REPORT calendar-query"):
            out[c["name"]] = c["objects"]
    return out


tz = ZoneInfo(TZ)
win_start = TODAY - timedelta(days=LOOKBACK_DAYS)
win_end = TODAY + timedelta(days=LOOKAHEAD_DAYS + 1)
by_calendar = read_caldav()
occ = []
for name, texts in by_calendar.items():
    occ += expand_ics(texts, name, win_start, win_end, tz)
occ = pd.DataFrame(occ)
occ[["occurrence_id", "by_occurrence", "recurring"]].rename(columns={"recurring": "by_vevent"}).to_csv(
    DATA / "calendar_rules.csv", index=False
)
save("calendar", "calendar", occ.drop(columns="by_occurrence"), CAL_SCHEMA)
window = {"window_start": win_start, "window_end": win_end, "calendars": len(by_calendar), "occurrences": len(occ)}
save("calendar", "calendar_window", pd.DataFrame([window]), WINDOW_SCHEMA)

# ---------- 설문 응답 ----------
SELECT_COLUMNS = ("created_at", "mbti", "saju_mbti", "day_master", "overall_score", "hour_known", "mapping_version")
SAJU_SCHEMA = Schema(
    {
        # mbti, saju_mbti, mbti_pair, letters_matched, day_master, score_band, hour_known, month
        "dimension": "VARCHAR",
        "value": "VARCHAR",
        "n": "BIGINT",
        "score_sum": "DOUBLE",
        "total": "BIGINT",  # 집계에 들어간 응답 수
        "mapping_versions": "VARCHAR",
    },
    ("dimension", "value", "n", "total"),
)
MIN_CELL = 1  # 이보다 적은 값은 (기타) 로 합침. 1 = 합치지 않음


def read_view() -> list[dict]:
    """DB 쪽 view 가 식별 열과 테스트 응답을 뺌. 접속 1 + 조회 1"""
    con = duckdb.connect()
    types = {c: "VARCHAR" for c in ("created_at", "mbti", "saju_mbti", "day_master", "mapping_version")}
    con.execute(
        "CREATE TABLE submissions AS SELECT * FROM read_csv(?, header = true, types = "
        + "{"
        + ", ".join(f"'{c}': '{t}'" for c, t in types.items())
        + "})",
        [str(SOURCE / "saju" / "submissions.csv")],
    )
    con.execute("CREATE SCHEMA changgo")
    con.execute(
        f"CREATE VIEW changgo.saju_submissions AS SELECT {', '.join(SELECT_COLUMNS)} FROM submissions WHERE NOT is_test"
    )
    with calls.call("saju", "connect"):
        pass
    with calls.call("saju", "SELECT changgo.saju_submissions"):
        cur = con.execute(f"SELECT {', '.join(SELECT_COLUMNS)} FROM changgo.saju_submissions ORDER BY created_at")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]


def letters_matched(a: str, b: str):
    a, b = (a or "").strip().upper(), (b or "").strip().upper()
    if len(a) != 4 or len(b) != 4:
        return None
    return sum(x == y for x, y in zip(a, b))


def score_band(score) -> str:
    try:
        s = int(score)
    except (TypeError, ValueError):
        return "(없음)"
    lo = max(0, min(s // 10 * 10, 90))
    return f"{lo:02d}~{lo + 9:02d}"


def aggregate(records: list[dict], min_cell: int = MIN_CELL) -> list[dict]:
    """행 -> (차원, 값) 건수. 행 단위로는 저장하지 않음"""
    local = ZoneInfo(TZ)
    counts = defaultdict(Counter)
    sums = defaultdict(lambda: defaultdict(float))
    versions = sorted({str(r.get("mapping_version") or "") for r in records} - {""})
    for r in records:
        mbti = (r.get("mbti") or "").strip().upper() or "(없음)"
        saju = (r.get("saju_mbti") or "").strip().upper() or "(없음)"
        score = r.get("overall_score")
        month = ""
        if r.get("created_at"):
            ts = r["created_at"]
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            month = ts.astimezone(local).strftime("%Y-%m")
        matched = letters_matched(mbti, saju)
        values = {
            "mbti": mbti,
            "saju_mbti": saju,
            "mbti_pair": f"{mbti}>{saju}",
            "letters_matched": "(없음)" if matched is None else str(matched),
            "day_master": (r.get("day_master") or "").strip() or "(없음)",
            "score_band": score_band(score),
            "hour_known": "시간 앎" if r.get("hour_known") else "시간 모름",
            "month": month or "(없음)",
        }
        for dim, val in values.items():
            counts[dim][val] += 1
            if isinstance(score, (int, float)):
                sums[dim][val] += float(score)
    rows = []
    for dim, counter in counts.items():
        merged, merged_sum = Counter(), defaultdict(float)
        for val, n in counter.items():
            key = val if n >= min_cell or dim == "month" else "(기타)"
            merged[key] += n
            merged_sum[key] += sums[dim][val]
        for val, n in sorted(merged.items()):
            rows.append(
                {
                    "dimension": dim,
                    "value": val,
                    "n": n,
                    "score_sum": merged_sum[val],
                    "total": len(records),
                    "mapping_versions": ",".join(versions),
                }
            )
    return rows


records = read_view()
for r in records:  # 다른 열이 섞여 와도 집계 전에 버림
    for k in list(r):
        if k not in SELECT_COLUMNS:
            del r[k]
save("saju", "saju_dist", pd.DataFrame(aggregate(records)), SAJU_SCHEMA)

# ---------- 기록 ----------
pd.DataFrame(saved).to_csv(RES / "collect.csv", index=False)
log = pd.read_json(CALLS, lines=True)
log.groupby(["source", "op"], sort=False).size().rename("calls").reset_index().to_csv(RES / "calls.csv", index=False)
rules = pd.read_csv(DATA / "calendar_rules.csv")
pd.DataFrame(
    [
        {"rule": "회차에 RECURRENCE-ID", "recurring": int(rules["by_occurrence"].sum()), "rows": len(rules)},
        {"rule": "원본 VEVENT 에 RRULE 이나 회차 수정", "recurring": int(rules["by_vevent"].sum()), "rows": len(rules)},
    ]
).to_csv(RES / "calendar_rules.csv", index=False)

for r in saved:
    print(f"{r['source']:9s} {r['table']:16s} {r['rows']:5d}행")
print("호출:", {k: int(v) for k, v in log.groupby("source", sort=False).size().items()})
print(
    f"반복 판정: 회차 기준 {rules['by_occurrence'].sum()}, 원본 VEVENT 기준 {rules['by_vevent'].sum()} / {len(rules)}"
)
