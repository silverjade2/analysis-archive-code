"""raw 는 parquet 위의 view 로만 두고 sql/stg -> sql/mart 를 파일 번호 순서로 실행. DAG 없음

mart 는 CSV 로 내보냄. 실행 시간은 같은 DB 파일에 N_TIMING 번 다시 돌려 잼 (sql_seconds.csv 만 실행마다 다름)
"""

import re
import statistics
import time

import duckdb
import pandas as pd
from common import DB, RAW, RES, SQL

RAW_TABLES = ["golf_player", "golf_course", "golf_round", "golf_score", "calendar", "calendar_window", "saju_dist"]
N_TIMING = 5
TARGET_RE = re.compile(r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+([A-Za-z_]+\.[A-Za-z_0-9]+)", re.I)


def quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def refresh_raw(con) -> None:
    """raw.<table> 은 날짜별 파일 전체, raw.<table>_latest 는 가장 최근 파일 하나"""
    for s in ("raw", "stg", "mart"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
    # 캘린더 이름 -> 카테고리 설정표. 매핑이 없으면 빈 표 (카테고리 = 캘린더 이름)
    con.execute(
        "CREATE OR REPLACE TABLE raw.calendar_categories (calendar VARCHAR, category VARCHAR, excluded BOOLEAN)"
    )
    for t in RAW_TABLES:
        files = sorted((RAW / t).glob("????-??-??.parquet"))
        glob = str(RAW / t / "????-??-??.parquet")
        con.execute(
            f"CREATE OR REPLACE VIEW raw.{t} AS SELECT * FROM read_parquet({quote(glob)}, union_by_name = true)"
        )
        con.execute(f"CREATE OR REPLACE VIEW raw.{t}_latest AS SELECT * FROM read_parquet({quote(str(files[-1]))})")


def run_layers(con) -> list[dict]:
    steps = []
    for layer in ("stg", "mart"):
        for path in sorted((SQL / layer).glob("[0-9][0-9][0-9]_*.sql")):
            sql = path.read_text(encoding="utf-8")
            t0 = time.perf_counter()
            con.execute(sql)
            sec = time.perf_counter() - t0
            for t in dict.fromkeys(TARGET_RE.findall(re.sub(r"--[^\n]*", "", sql))):
                steps.append(
                    {
                        "layer": layer,
                        "file": path.name,
                        "table": t,
                        "rows": con.execute(f"SELECT count(*) FROM {t}").fetchone()[0],
                        "seconds": sec,
                    }
                )
    return steps


def one_run() -> tuple[list[dict], float, float]:
    t0 = time.perf_counter()
    con = duckdb.connect(str(DB))
    refresh_raw(con)
    t1 = time.perf_counter()
    steps = run_layers(con)
    t2 = time.perf_counter()
    con.close()
    return steps, t2 - t1, t2 - t0


DB.unlink(missing_ok=True)
steps, *_ = one_run()
pd.DataFrame(steps).drop(columns="seconds").to_csv(RES / "sql_steps.csv", index=False)

timing = []
for i in range(N_TIMING):
    s, sql_sec, total_sec = one_run()
    timing.append(
        {
            "run": i + 1,
            "sql_seconds": round(sql_sec, 3),
            "with_raw_views_seconds": round(total_sec, 3),
            "slowest_file": max(s, key=lambda r: r["seconds"])["file"],
        }
    )
pd.DataFrame(timing).to_csv(RES / "sql_seconds.csv", index=False)

con = duckdb.connect(str(DB), read_only=True)
EXPORT = {
    "golf_rounds": "SELECT * FROM mart.golf_rounds ORDER BY round_date, day_no, round_of_day, player_id",
    "golf_player_trend": "SELECT * FROM mart.golf_player_trend ORDER BY player_id, month",
    "golf_hole_stats": "SELECT * FROM mart.golf_hole_stats ORDER BY player_id, base_course, hole_no",
    "time_by_category_weekly": "SELECT * FROM mart.time_by_category_weekly ORDER BY week_start, category",
    "saju_response_dist": "SELECT * FROM mart.saju_response_dist ORDER BY dimension, value",
}
for name, q in EXPORT.items():
    df = con.execute(q).df()
    for c in df.select_dtypes("float").columns:  # sum() 이 HUGEINT 라 pandas 에서 float 로 옴
        if (df[c].dropna() % 1 == 0).all():
            df[c] = df[c].astype("Int64")
    df.to_csv(RES / f"{name}.csv", index=False)
con.close()

for r in steps:
    print(f"{r['layer']:4s} {r['file']:26s} {r['table']:30s} {r['rows']:5d}행")
med = statistics.median(t["sql_seconds"] for t in timing)
top = max(t["sql_seconds"] for t in timing)
n_files = len({(r["layer"], r["file"]) for r in steps})
print(f"stg, mart SQL {n_files}파일: 중앙값 {med:.3f}초, 최대 {top:.3f}초 ({N_TIMING}번)")
