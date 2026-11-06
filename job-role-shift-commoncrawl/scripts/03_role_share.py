"""직무군 비중과 외부 통계 대조. 공고 수 변화를 전체 변화와 직무 고유 변화로 나눈다

WDC 회차는 그해 10월 크롤링이라 외부 통계도 10월 값과 비교 (9~11월 평균은 민감도 확인)
"""

import json
import os

import duckdb
import pandas as pd
from common import CROSS as CRAWLS
from common import DATA, NORM, RESULTS, T1, US, WORK_DIR
from roles import ANALYST_ANY, MIN_BODY_CHARS, role_sql

EXT = DATA / "external"
FAMILIES = {  # WDC 제목 규칙, Indeed 직군
    "software": (r"software (engineer|developer)|developer|programmer|개발자", "Software Development"),
    "nurse": (r"(^|[^a-z])nurse|(^|[^a-z])rn([^a-z]|$)|nursing", "Nursing"),
    "sales": (r"(^|[^a-z])sales", "Sales"),
    "accountant": (r"accountant|bookkeep", "Accounting"),
    "teacher": (r"teacher|lecturer|professor", "Education & Instruction"),
    "driver": (r"(^|[^a-z])driver", "Driving"),
}
DATA_ROLES = ("DA", "BA/BI", "DS", "DE", "ML/AI 엔지니어")
JOLTS = {
    "JTS000000000000000JOL": "전체",
    "JTS510000000000000JOL": "정보",
    "JTS540099000000000JOL": "전문, 사업 서비스",
    "JTS620000000000000JOL": "보건, 사회복지",
}


def wdc_table() -> pd.DataFrame:
    con = duckdb.connect()
    cores = os.environ.get("SPARK_CORES", "4")
    con.execute(
        f"SET threads={cores}; SET memory_limit='8GB'; SET temp_directory='{T1}/duck_tmp'; "
        "SET preserve_insertion_order=false;"
    )
    fam = " ".join(f"WHEN regexp_matches(t,'{p}') THEN '{n}'" for n, (p, _) in FAMILIES.items())
    con.execute(f"""CREATE TABLE a AS
      WITH s AS (SELECT crawl, host, country, lower(coalesce(title,'')) t, title, org_name, address_text, description,
          length(coalesce(description,'')) dl, try_cast(left(datePosted,4) AS int) y
          FROM read_parquet('{WORK_DIR}/wdc_jobs/crawl=*/*.parquet', hive_partitioning=true))
      SELECT crawl, host, lower(trim(coalesce(country,''))) IN {US} is_us, {role_sql("t")} rf,
        CASE {fam} ELSE 'x' END fam,
        regexp_matches(t,'{ANALYST_ANY}') any_analyst,
        CASE WHEN dl>={MIN_BODY_CHARS} THEN md5({NORM.format("description")})
             ELSE 'T|'||{NORM.format("title")}||'|'||{NORM.format("org_name")}||'|'
                  ||lower(coalesce(address_text,'')) END k
      FROM s WHERE y IS NULL OR y BETWEEN left(crawl,4)::int-1 AND left(crawl,4)::int""")
    con.execute("""CREATE TABLE d AS SELECT crawl, k, any_value(host) host, bool_or(is_us) is_us, any_value(rf) rf,
      any_value(fam) fam, bool_or(any_analyst) any_analyst FROM a GROUP BY 1,2""")
    con.execute("CREATE TABLE h AS SELECT host FROM d GROUP BY 1 HAVING count(DISTINCT crawl)=3")
    scopes = {
        "전체": "d",
        "미국": "(SELECT * FROM d WHERE is_us)",
        "사이트 고정": "(SELECT d.* FROM d JOIN h USING (host))",
        "미국 사이트 고정": "(SELECT d.* FROM d JOIN h USING (host) WHERE is_us)",
    }
    rows = []
    for scope, src in scopes.items():
        data_in = ",".join(f"'{r}'" for r in DATA_ROLES)
        q = f"""
          SELECT crawl, '전체 공고' g, count(*) n FROM {src} GROUP BY 1
          UNION ALL SELECT crawl, rf, count(*) FROM {src} WHERE rf<>'그 밖' GROUP BY 1,2
          UNION ALL SELECT crawl, '데이터 직무 합계', count(*) FROM {src} WHERE rf IN ({data_in}) GROUP BY 1
          UNION ALL SELECT crawl, fam, count(*) FROM {src} WHERE fam<>'x' GROUP BY 1,2
          UNION ALL SELECT crawl, 'analyst 류 전체', count(*) FROM {src} WHERE any_analyst GROUP BY 1"""
        x = con.execute(q).df()
        x["scope"] = scope
        rows.append(x)
    return pd.concat(rows)


def wdc_summary(t: pd.DataFrame) -> pd.DataFrame:
    p = t.pivot_table(index=["scope", "g"], columns="crawl", values="n").reset_index()
    tot = p[p.g == "전체 공고"].set_index("scope")
    for c in CRAWLS:
        p[f"share_{c}"] = p[c] / p.scope.map(tot[c])
    a, b = CRAWLS[0], CRAWLS[-1]
    p["n_change_pct"] = (p[b] / p[a] - 1) * 100
    p["share_change_pct"] = (p[f"share_{b}"] / p[f"share_{a}"] - 1) * 100
    return p


def indeed_month(months: list[int]) -> pd.DataFrame:
    s = pd.read_csv(EXT / "indeed/job_postings_by_sector_US.csv", parse_dates=["date"])
    s = s[s.variable == "total postings"].rename(columns={"indeed_job_postings_index": "v"})
    a = pd.read_csv(EXT / "indeed/aggregate_job_postings_US.csv", parse_dates=["date"])
    a = a[a.variable == "total postings"].rename(columns={"indeed_job_postings_index_SA": "v"})
    a["display_name"] = "전체 공고"
    x = pd.concat([s[["date", "display_name", "v"]], a[["date", "display_name", "v"]]])
    x = x[x.date.dt.month.isin(months) & x.date.dt.year.isin([2022, 2023, 2024])]
    return x.groupby(["display_name", x.date.dt.year]).v.mean().unstack()


def jolts_month(months: list[int], file: str, suffix: str) -> pd.DataFrame:
    rows = []
    for s in json.loads((EXT / "fred" / file).read_text())["Results"]["series"]:
        for r in s["data"]:
            if int(r["period"][1:]) in months:
                rows.append((JOLTS[s["seriesID"][:-3] + "JOL"] + suffix, int(r["year"]), float(r["value"])))
    return pd.DataFrame(rows, columns=["g", "y", "v"]).groupby(["g", "y"]).v.mean().unstack()


def spearman(a: pd.Series, b: pd.Series) -> float:
    return round(a.rank().corr(b.rank()), 2)  # scipy 없이 순위의 Pearson 상관


def indeed_october_all() -> pd.DataFrame:
    s = pd.read_csv(EXT / "indeed/job_postings_by_sector_US.csv", parse_dates=["date"])
    s = s[s.variable == "total postings"].rename(columns={"indeed_job_postings_index": "v"})
    a = pd.read_csv(EXT / "indeed/aggregate_job_postings_US.csv", parse_dates=["date"])
    a = a[a.variable == "total postings"].rename(columns={"indeed_job_postings_index_SA": "v"})
    a["display_name"] = "전체 공고"
    x = pd.concat([s[["date", "display_name", "v"]], a[["date", "display_name", "v"]]])
    x = x[x.date.dt.month == 10]
    names = ["전체 공고", "Data & Analytics"] + [v for _, v in FAMILIES.values()]
    return x[x.display_name.isin(names)].groupby(["display_name", x.date.dt.year]).v.mean().unstack()


def pct(df: pd.DataFrame) -> pd.Series:
    return (df[2024] / df[2022] - 1) * 100


def main() -> None:
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.1f}".format)
    s = wdc_summary(wdc_table())
    s.to_csv(RESULTS / "task1_role_share.csv", index=False)
    print(s[["scope", "g", *CRAWLS, "n_change_pct", "share_change_pct"]].to_string(index=False))

    compare = []
    for label, months in [("10월", [10]), ("9 ~ 11월", [9, 10, 11])]:
        ind = indeed_month(months)
        jol = pd.concat([jolts_month(months, "bls.json", " 구인"), jolts_month(months, "bls_hires.json", " 채용")])
        print(f"\n## {label} JOLTS (천 건)\n", jol.assign(change_pct=pct(jol)).to_string())
        names = ["전체 공고", "Data & Analytics"] + [v for _, v in FAMILIES.values()]
        print(f"## {label} Indeed 지수\n", ind.loc[names].assign(change_pct=lambda d: pct(d)).to_string())
        base = pct(ind).loc["전체 공고"]
        for wdc_g, ind_g in [("데이터 직무 합계", "Data & Analytics")] + [(k, v) for k, (_, v) in FAMILIES.items()]:
            for scope in ["미국", "미국 사이트 고정"]:
                r = s[(s.scope == scope) & (s.g == wdc_g)].iloc[0]
                ind_rel = ((1 + pct(ind).loc[ind_g] / 100) / (1 + base / 100) - 1) * 100
                compare.append(
                    {
                        "window": label,
                        "scope": scope,
                        "wdc": wdc_g,
                        "indeed": ind_g,
                        "wdc_n_change": r.n_change_pct,
                        "wdc_share_change": r.share_change_pct,
                        "indeed_change": pct(ind).loc[ind_g],
                        "indeed_relative_change": ind_rel,
                    }
                )
    c = pd.DataFrame(compare)
    c.to_csv(RESULTS / "task1_external_compare.csv", index=False)
    print(c[c.window == "10월"].to_string(index=False))
    for (w, sc), g in c.groupby(["window", "scope"]):
        print(
            w,
            sc,
            "비중 변화 Spearman",
            spearman(g.wdc_share_change, g.indeed_relative_change),
            "n",
            len(g),
            "건수 변화 Spearman",
            spearman(g.wdc_n_change, g.indeed_change),
        )
    print("\n## WDC 이후 Indeed 10월 지수 (2020-02 = 100)")
    print(indeed_october_all().to_string())


if __name__ == "__main__":
    main()
