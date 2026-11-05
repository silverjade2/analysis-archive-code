"""기술별 GitHub 이벤트와 PyPI 다운로드를 ClickHouse 공개 서버에서 주 단위로 집계해 parquet 로 저장

- GitHub: play.clickhouse.com 의 github_events (GH Archive 복제, 정렬 키 event_type, repo_name, created_at)
- PyPI: sql-clickhouse.clickhouse.com 의 pypi.pypi_downloads_per_day
- 기술과 저장소, 패키지 연결: dbt/seeds/skills.csv (strength = none 은 제외)

사용: python scripts/01_fetch_activity.py [--start 2023-01-02] [--end 2026-11-02]
저장: data/{github_weekly,pypi_weekly}.parquet
"""

import argparse
import csv
import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

GH_URL = "https://play.clickhouse.com/?user=play"
PYPI_URL = "https://sql-clickhouse.clickhouse.com/?user=demo"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

# 커밋 수는 2025-10 부터 GH Archive payload 에서 빠졌으므로 쓰지 않는다
GH_EVENTS = ("WatchEvent", "ForkEvent", "PullRequestEvent", "IssuesEvent", "PushEvent", "IssueCommentEvent")


def query(url: str, sql: str) -> pd.DataFrame:
    req = urllib.request.Request(url, data=(sql + " FORMAT CSVWithNames").encode(), method="POST")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return pd.read_csv(io.BytesIO(r.read()), parse_dates=["week"])
        except Exception as e:  # 공개 서버라 가끔 끊김
            if attempt == 4:
                raise
            print(f"  재시도 {attempt + 1}: {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError


def load_map() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = list(csv.DictReader(open(ROOT / "dbt/seeds/skills.csv", encoding="utf-8")))
    gh = [
        (r["skill"], r["strength"], repo)
        for r in rows
        if r["strength"] != "none"
        for repo in r["github_repos"].split("|")
        if repo
    ]
    pp = [
        (r["skill"], r["strength"], pkg)
        for r in rows
        if r["strength"] != "none"
        for pkg in r["pypi_packages"].split("|")
        if pkg
    ]
    return (
        pd.DataFrame(gh, columns=["skill", "strength", "repo_name"]),
        pd.DataFrame(pp, columns=["skill", "strength", "project"]),
    )


def sql_list(values) -> str:
    return ",".join("'" + v.replace("'", "") + "'" for v in sorted(set(values)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-02")  # 월요일
    ap.add_argument("--end", default="2026-11-02")  # 이 날 전 주까지 (끝 주가 덜 찬 주가 되지 않게)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gh_map, pp_map = load_map()

    t0 = time.time()
    gh = query(
        GH_URL,
        f"""
        SELECT toMonday(created_at) AS week, repo_name, toString(event_type) AS event_type,
               count() AS events, uniqExact(actor_login) AS actors
        FROM github_events
        WHERE event_type IN ({sql_list(GH_EVENTS)})
          AND repo_name IN ({sql_list(gh_map.repo_name)})
          AND created_at >= '{args.start}' AND created_at < '{args.end}'
          AND NOT endsWith(actor_login, '[bot]')
        GROUP BY week, repo_name, event_type
        ORDER BY week, repo_name, event_type""",
    )
    gh = gh.merge(gh_map, on="repo_name", how="left")
    gh["week"] = gh["week"].dt.date  # parquet date32: Snowflake 에서 DATE 로 잡히게
    gh.to_parquet(OUT / "github_weekly.parquet", index=False)
    missing = sorted(set(gh_map.repo_name) - set(gh.repo_name))
    print(
        f"GitHub: {len(gh):,}행, 저장소 {gh.repo_name.nunique()}개 ({time.time() - t0:.0f}초)"
        + (f", 활동 없는 저장소: {missing}" if missing else "")
    )

    t0 = time.time()
    pp = query(
        PYPI_URL,
        f"""
        SELECT toMonday(date) AS week, project, sum(count) AS downloads
        FROM pypi.pypi_downloads_per_day
        WHERE project IN ({sql_list(pp_map.project)})
          AND date >= '{args.start}' AND date < '{args.end}'
        GROUP BY week, project
        ORDER BY week, project""",
    )
    pp = pp.merge(pp_map, on="project", how="left")
    pp["week"] = pp["week"].dt.date
    pp.to_parquet(OUT / "pypi_weekly.parquet", index=False)
    missing = sorted(set(pp_map.project) - set(pp.project))
    print(
        f"PyPI: {len(pp):,}행, 패키지 {pp.project.nunique()}개 ({time.time() - t0:.0f}초)"
        + (f", 다운로드 없는 패키지: {missing}" if missing else "")
    )


if __name__ == "__main__":
    main()
