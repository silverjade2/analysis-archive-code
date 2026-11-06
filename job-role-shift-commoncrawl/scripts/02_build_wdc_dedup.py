"""직무군을 붙이고 중복, 오래된 공고를 뺀 WDC 기준 데이터

중복 키: 정규화한 본문의 md5, 본문 200자 미만은 제목 + 회사 + 지역. 회차 안에서만 합친다
게시연도: 회차 연도와 전년만 남기고, 게시일을 읽을 수 없는 공고는 유지
"""

import os

import duckdb
from common import NORM, T1, WORK_DIR
from roles import MIN_BODY_CHARS, role_sql

OUT = T1 / "wdc_jobs_dedup.parquet"


def main() -> None:
    T1.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    cores = os.environ.get("SPARK_CORES", "9")
    con.execute(f"SET threads={cores}; SET memory_limit='10GB'; SET temp_directory='{T1}/duck_tmp';")
    con.execute(f"""
    COPY (
      WITH a AS (
        SELECT *, lower(coalesce(title, '')) AS t, length(coalesce(description, '')) AS body_chars,
               try_cast(left(datePosted, 4) AS int) AS posted_year
        FROM read_parquet('{WORK_DIR}/wdc_jobs/crawl=*/*.parquet', hive_partitioning=true)
      ),
      r AS (SELECT *, {role_sql("t")} AS role_family FROM a),
      f AS (
        SELECT * FROM r
        WHERE role_family <> '그 밖'
          AND (posted_year IS NULL OR posted_year BETWEEN left(crawl, 4)::int - 1 AND left(crawl, 4)::int)
      ),
      k AS (
        SELECT *, CASE WHEN body_chars >= {MIN_BODY_CHARS} THEN md5({NORM.format("description")})
                       ELSE 'T|' || {NORM.format("title")} || '|' || {NORM.format("org_name")} || '|'
                            || lower(coalesce(address_text, '')) END AS dedup_key
        FROM f
      )
      SELECT * EXCLUDE (t) FROM k
      QUALIFY row_number() OVER (PARTITION BY crawl, dedup_key ORDER BY page_url, node) = 1
    ) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(
        con.execute(f"""SELECT role_family, crawl, count(*) n, sum((body_chars >= {MIN_BODY_CHARS})::int) body_ok
                          FROM '{OUT}' GROUP BY 1, 2 ORDER BY 1, 2""")
        .df()
        .to_string()
    )


if __name__ == "__main__":
    main()
