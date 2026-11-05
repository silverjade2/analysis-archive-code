"""사이트 위젯 두 개의 json. dbt 결과 표(csv)와 GitHub 주 단위 집계(parquet)를 읽어 outputs/results/ 에 쓴다

- skill-quadrant.json: 측정 기술 32개의 사분면 값 + 패키지별 주간 다운로드
- event-break.json: 기술별 분기 GitHub 스타와 PyPI 다운로드, 각각 2024년 분기 평균 = 100
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "outputs" / "results"

RECENT_WEEKS = 13
LAG_WEEKS = 156  # 3년 전 같은 13주
BASE_YEAR = 2024

# 사전과 dbt 는 strong. Python 클라이언트 패키지라 글에서는 Docker 와 같이 proxy 로 설명한다
STRENGTH_OVERRIDE = {"Kubernetes": "proxy"}


def write_json(name, obj):
    RES.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    (RES / name).write_text(text + "\n", encoding="utf-8")
    print(f"{name}: {len(text) / 1024:.0f} KB")


def quadrant_data(mart, pypi):
    weeks = sorted(pypi["WEEK"].unique())
    last = pd.Timestamp(weeks[-1])
    recent = (last - pd.Timedelta(days=7 * (RECENT_WEEKS - 1)), last)
    lag = pd.Timedelta(days=7 * LAG_WEEKS)
    past = (recent[0] - lag, last - lag)

    measured = mart[mart["DOWNLOADS_13W"].notna()].copy()
    measured["STRENGTH"] = measured["SKILL"].map(STRENGTH_OVERRIDE).fillna(measured["STRENGTH"])

    wide = pypi.pivot_table(index="WEEK", columns="PACKAGE", values="DOWNLOADS", aggfunc="sum").reindex(weeks)
    packages_by_skill = pypi.groupby("SKILL")["PACKAGE"].unique()

    # 패키지 합으로 다시 낸 13주 합이 mart 값과 같은지 확인
    w = pd.to_datetime(wide.index)
    for _, r in measured.iterrows():
        cols = sorted(packages_by_skill[r["SKILL"]])
        dl = wide.loc[(w >= recent[0]) & (w <= recent[1]), cols].sum().sum()
        assert dl == r["DOWNLOADS_13W"], (r["SKILL"], dl, r["DOWNLOADS_13W"])

    skills = []
    for _, r in measured.sort_values("POSTING_SHARE", ascending=False).iterrows():
        skills.append(
            {
                "skill": r["SKILL"],
                "category": r["CATEGORY"],
                "strength": r["STRENGTH"],
                "share": round(float(r["POSTING_SHARE"]), 6),
                "downloads": int(r["DOWNLOADS_13W"]),
                "growth": None if pd.isna(r["GROWTH_3Y"]) else round(float(r["GROWTH_3Y"]), 4),
                "quadrant": r["QUADRANT"],
                "packages": sorted(packages_by_skill[r["SKILL"]]),
            }
        )

    series = {p: [None if pd.isna(v) else int(v) for v in wide[p]] for p in sorted(wide.columns)}
    meta = {
        "nMeasured": len(measured),
        "strength": measured["STRENGTH"].value_counts().sort_index().to_dict(),
        "medShare": float(measured["POSTING_SHARE"].median()),
        "medDownloads": float(measured["DOWNLOADS_13W"].median()),
        "medGrowth": round(float(measured["GROWTH_3Y"].median()), 4),
        "nGrowth": int(measured["GROWTH_3Y"].notna().sum()),
        "recent": [d.strftime("%Y-%m-%d") for d in recent],
        "past": [d.strftime("%Y-%m-%d") for d in past],
    }
    print("strength", meta["strength"], "medians", meta["medShare"], meta["medDownloads"], meta["medGrowth"])
    return {"meta": meta, "skills": skills, "weeks": weeks, "series": series}


def quarterly(df, week_col, value_col, last_week):
    q = pd.PeriodIndex(pd.to_datetime(df[week_col]), freq="Q")
    out = df.assign(q=q).groupby(["key", "q"])[value_col].sum()
    # 마지막 주가 걸친 분기는 다 차지 않았으므로 뺀다
    full = out.index.get_level_values("q").end_time.normalize() <= last_week + pd.Timedelta(days=6)
    return out[full]


def indexed(s):
    base = s[s.index.year == BASE_YEAR].mean()
    return [round(float(v) / base * 100, 2) for v in s]


def event_break_data(gh, pypi):
    stars = gh[gh["event_type"] == "WatchEvent"]
    assert stars.groupby("repo_name")["skill"].nunique().max() == 1
    last_week = pd.Timestamp(max(stars["week"].max(), pypi["WEEK"].max()))

    gh_all = stars.assign(key="전체")
    pp_all = pypi.assign(key="전체")
    gh_q = quarterly(pd.concat([gh_all, stars.assign(key=stars["skill"])]), "week", "events", last_week)
    pp_q = quarterly(pd.concat([pp_all, pypi.assign(key=pypi["SKILL"])]), "WEEK", "DOWNLOADS", last_week)

    quarters = sorted(set(gh_q.index.get_level_values("q")) | set(pp_q.index.get_level_values("q")))
    repos = stars.groupby("skill")["repo_name"].unique()
    pkgs = pypi.groupby("SKILL")["PACKAGE"].unique()

    keys = ["전체"] + sorted(set(stars["skill"]) | set(pypi["SKILL"]))
    items = []
    for k in keys:
        g = gh_q[k].reindex(quarters) if k in gh_q.index.get_level_values("key") else None
        p = pp_q[k].reindex(quarters) if k in pp_q.index.get_level_values("key") else None
        items.append(
            {
                "key": k,
                "repos": sorted(stars["repo_name"].unique()) if k == "전체" else sorted(repos.get(k, [])),
                "packages": sorted(pypi["PACKAGE"].unique()) if k == "전체" else sorted(pkgs.get(k, [])),
                "stars": None if g is None else [int(v) for v in g.fillna(0)],
                "starsIdx": None if g is None else indexed(g.fillna(0)),
                "downloads": None if p is None else [int(v) for v in p.fillna(0)],
                "downloadsIdx": None if p is None else indexed(p.fillna(0)),
            }
        )

    tot = gh_q["전체"]
    print("스타 합 2024", tot[tot.index.year == BASE_YEAR].min(), "~", tot[tot.index.year == BASE_YEAR].max())
    print("스타 합 2026Q3", tot[pd.Period("2026Q3")], "분기 수", len(quarters))
    return {"quarters": [str(q) for q in quarters], "items": items}


def main():
    mart = pd.read_csv(DATA / "SKILL_DEMAND_VS_ACTIVITY.csv")
    pypi = pd.read_csv(DATA / "STG_PYPI_WEEKLY.csv")
    gh = pd.read_parquet(DATA / "github_weekly.parquet")
    gh["week"] = pd.to_datetime(gh["week"]).dt.strftime("%Y-%m-%d")

    write_json("skill-quadrant.json", quadrant_data(mart, pypi))
    write_json("event-break.json", event_break_data(gh, pypi))


if __name__ == "__main__":
    main()
