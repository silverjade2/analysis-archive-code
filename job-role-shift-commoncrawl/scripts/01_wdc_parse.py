"""WDC JobPosting N-Quads를 공고 한 행 parquet로

한 줄 = (subject, predicate, object, graph), graph는 공고가 있던 페이지 URL
blank node id는 페이지 안에서만 의미가 있어 (graph, subject)를 노드 키로 쓴다
"""

import argparse
import time
from pathlib import Path

from common import WORK_DIR
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from spark_local import get_spark

# object는 탐욕 매칭으로 마지막 <graph> 앞까지 (literal 안의 공백, > 도 안전)
QUAD_RE = r"^(\S+)\s+<([^>]*)>\s+(.*)\s+<([^>]*)>\s*\.\s*$"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# 공고 노드에서 바로 꺼낼 literal 속성
JOB_PROPS = [
    "title",
    "description",
    "datePosted",
    "validThrough",
    "employmentType",
    "occupationalCategory",
    "skills",
    "industry",
    "qualifications",
    "experienceRequirements",
    "educationRequirements",
    "jobLocationType",
    "responsibilities",
]


def read_quads(spark, path: str) -> DataFrame:
    raw = spark.read.text(path)
    q = raw.select(
        F.regexp_extract("value", QUAD_RE, 1).alias("s"),
        F.regexp_extract("value", QUAD_RE, 2).alias("p_uri"),
        F.regexp_extract("value", QUAD_RE, 3).alias("o_raw"),
        F.regexp_extract("value", QUAD_RE, 4).alias("g"),
    ).where(F.col("g") != "")
    # http, https schema.org를 같은 이름으로
    q = q.withColumn("p", F.regexp_replace("p_uri", r"^https?://schema\.org/", ""))
    is_lit = F.col("o_raw").startswith('"')
    lit_body = F.regexp_extract("o_raw", r'^"(.*)"(?:@[\w-]+|\^\^<[^>]*>)?$', 1)
    # N-Quads 이스케이프와 HTML 엔티티를 JVM 안에서 풀어 Python UDF를 피한다
    java_unescaped = F.call_function(
        "java_method", F.lit("org.apache.commons.lang3.StringEscapeUtils"), F.lit("unescapeJava"), lit_body
    )
    # 0x10FFFF를 넘는 숫자 엔티티는 unescapeHtml4가 예외를 던져 회차 전체가 실패 (2024-12). 풀기 전에 지운다
    too_big = (
        r"(?i)&#x0*([2-9a-f][0-9a-f]{5}|1[1-9a-f][0-9a-f]{4}|[0-9a-f]{7,});"
        r"|&#0*([0-9]{8,}|[2-9][0-9]{6}|1[2-9][0-9]{5});"
    )
    unescaped = F.call_function(
        "java_method",
        F.lit("org.apache.commons.lang3.StringEscapeUtils"),
        F.lit("unescapeHtml4"),
        F.regexp_replace(java_unescaped, too_big, ""),
    )
    return q.select(
        "g",
        "s",
        "p",
        # F.trim은 공백만 지운다. 줄바꿈까지 지우려고 정규식
        F.when(is_lit, F.regexp_replace(unescaped, r"^\s+|\s+$", "")).alias("lit"),
        F.when(is_lit, F.regexp_extract("o_raw", r'"@([\w-]+)$', 1)).alias("lang"),
        F.when(~is_lit, F.regexp_replace("o_raw", r"^<|>$", "")).alias("ref"),
    )


def build_jobs(q: DataFrame) -> DataFrame:
    types = q.where(F.col("p") == RDF_TYPE).select(
        "g", "s", F.regexp_replace("ref", r"^https?://schema\.org/", "").alias("type")
    )
    job_keys = types.where(F.col("type") == "JobPosting").select("g", "s").distinct()

    jq = q.join(job_keys, ["g", "s"])
    # 같은 속성이 여러 번 나오면 가장 긴 값 하나
    lits = (
        jq.where(F.col("p").isin(JOB_PROPS) & F.col("lit").isNotNull())
        .groupBy("g", "s")
        .pivot("p", JOB_PROPS)
        .agg(F.max_by("lit", F.length("lit")))
    )
    lang = jq.where(F.col("lang") != "").groupBy("g", "s").agg(F.first("lang").alias("lang"))

    # 회사 이름: hiringOrganization이 노드면 name, literal이면 그대로
    names = q.where(F.col("p") == "name").groupBy("g", "s").agg(F.max_by("lit", F.length("lit")).alias("name"))
    org_ref = (
        jq.where(F.col("p") == "hiringOrganization")
        .groupBy("g", "s")
        .agg(F.first("ref", ignorenulls=True).alias("org_node"), F.first("lit", ignorenulls=True).alias("org_lit"))
    )
    org = org_ref.join(names.withColumnRenamed("s", "org_node"), ["g", "org_node"], "left").select(
        "g", "s", F.coalesce("name", "org_lit").alias("org_name")
    )

    # 국가: jobLocation, address, addressCountry 순 (literal 또는 Country 노드의 name)
    loc = jq.where(F.col("p") == "jobLocation").select("g", "s", F.col("ref").alias("loc_node"))
    addr = q.where(F.col("p") == "address").select(
        "g", F.col("s").alias("loc_node"), F.col("ref").alias("addr_node"), F.col("lit").alias("addr_lit")
    )
    country = q.where(F.col("p") == "addressCountry").select(
        "g", F.col("s").alias("addr_node"), F.col("ref").alias("c_node"), F.col("lit").alias("c_lit")
    )
    c_named = country.join(
        names.select("g", F.col("s").alias("c_node"), F.col("name").alias("c_name")), ["g", "c_node"], "left"
    )
    loc_country = (
        loc.join(addr, ["g", "loc_node"], "left")
        .join(c_named, ["g", "addr_node"], "left")
        .groupBy("g", "s")
        .agg(
            F.first(F.coalesce("c_lit", "c_name"), ignorenulls=True).alias("country"),
            F.first("addr_lit", ignorenulls=True).alias("address_text"),
        )
    )

    jobs = (
        job_keys.join(lits, ["g", "s"], "left")
        .join(lang, ["g", "s"], "left")
        .join(org, ["g", "s"], "left")
        .join(loc_country, ["g", "s"], "left")
        .withColumn("host", F.lower(F.regexp_extract("g", r"^https?://([^/:]+)", 1)))
        .withColumnRenamed("g", "page_url")
        .withColumnRenamed("s", "node")
    )
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", required=True)
    ap.add_argument("--input")
    ap.add_argument("--out")
    args = ap.parse_args()

    src = args.input or str(WORK_DIR / "wdc" / args.crawl)
    if Path(src).is_dir():
        src = str(Path(src) / "**" / "*.gz") if not args.input else src
    out = args.out or str(WORK_DIR / "wdc_jobs" / f"crawl={args.crawl}")

    t0 = time.time()
    spark = get_spark(f"wdc-{args.crawl}")
    spark.sparkContext.setLogLevel("ERROR")
    q = read_quads(spark, src)
    jobs = build_jobs(q)
    jobs.write.mode("overwrite").parquet(out)
    n = spark.read.parquet(out).count()
    print(f"{args.crawl}: 공고 {n:,}건, {out} ({(time.time() - t0) / 60:.1f}분)")
    spark.stop()


if __name__ == "__main__":
    main()
