"""학습한 업무 유형 모델을 기준 데이터 전체에 적용 (Spark)

적용 대상: 본문 200자 이상, 직무군 '그 밖' 제외 143,762건. 국내 공고는 라벨을 그대로 쓴다
feature와 점수 계산은 model_core.py. 그 소스와 모델 파일 내용을 작업자에 실어 보낸다
"""

import argparse
import time
from pathlib import Path

from common import T1
from spark_local import get_spark

HERE = Path(__file__).resolve().parent

OUT_SCHEMA = (
    "doc_id string, crawl string, page_url string, node string, role_family string, tasks string, "
    "primary_task string, english boolean, probs array<float>"
)


def make_scorer(core_src: str, npz_bytes: bytes, vocab_text: str):
    """partition마다 모델을 한 번 만들어 pandas 조각을 점수로 바꾸는 함수. 클로저에 소스와 모델 내용"""

    def score(batches):
        import json

        import pandas as pd

        ns: dict = {"__name__": "model_core"}
        exec(core_src, ns)  # noqa: S102  model_core.py 소스 그대로
        model = ns["Model"].from_blobs(npz_bytes, vocab_text)
        plain_text, is_english = ns["plain_text"], ns["is_english"]
        for pdf in batches:
            texts = [plain_text(d) for d in pdf["description"]]
            P = model.proba(texts)
            dec = [model.decide(p) for p in P]
            # doc_id는 label_pool과 같은 식 (pandas hash)이라 학습 라벨과 연결된다
            h = pd.util.hash_pandas_object(pdf["page_url"] + pdf["node"], index=False).astype(str)
            yield pd.DataFrame(
                {
                    "doc_id": "wdc-" + pdf["crawl"].astype(str) + "-" + h,
                    "crawl": pdf["crawl"].astype(str),
                    "page_url": pdf["page_url"],
                    "node": pdf["node"],
                    "role_family": pdf["role_family"],
                    "tasks": [json.dumps(t) for t, _ in dec],
                    "primary_task": [pt for _, pt in dec],
                    "english": [is_english(t) for t in texts],
                    "probs": [p.astype("float32").tolist() for p in P],
                }
            )

    return score


def run(
    spark,
    input_path: str,
    model_dir: str,
    out_path: str,
    crawl: str | None = None,
    limit: int | None = None,
    partitions: int = 16,
) -> int:
    """input_path: 기준 데이터 parquet (crawl, page_url, node, role_family, description, body_chars 열)"""
    try:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
    except Exception:  # noqa: BLE001  Databricks serverless는 이 설정을 막는다. 기본 배치 크기로도 돈다
        pass
    m = Path(model_dir)
    scorer = make_scorer(
        (HERE / "model_core.py").read_text(encoding="utf-8"),
        (m / "model.npz").read_bytes(),
        (m / "vocab.json").read_text(encoding="utf-8"),
    )
    df = spark.read.parquet(input_path).where("body_chars >= 200 AND role_family <> '그 밖'")
    if crawl:
        df = df.where(f"crawl = '{crawl}'")
    # 모델이 보는 범위가 앞 4,000자라 Spark에서 먼저 자른다 (긴 본문이 Python 작업자 메모리를 채우는 것을 막음)
    df = df.selectExpr("crawl", "page_url", "node", "role_family", "substring(description, 1, 4000) AS description")
    if limit:
        df = df.limit(limit)
    t0 = time.time()
    df.repartition(partitions).mapInPandas(scorer, OUT_SCHEMA).write.mode("overwrite").parquet(out_path)
    n = spark.read.parquet(out_path).count()
    sec = time.time() - t0
    print(f"적용 {n:,}건, {sec:.0f}초 ({n / sec:,.0f}건/초), {out_path}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(T1 / "wdc_jobs_dedup.parquet"))
    ap.add_argument("--model", default=str(T1 / "model_b/final"))
    ap.add_argument("--out", default=str(T1 / "task_pred_b"))
    ap.add_argument("--crawl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--partitions", type=int, default=16)
    a = ap.parse_args()

    spark = get_spark("task1-b-apply")
    run(spark, a.input, a.model, a.out, a.crawl, a.limit, a.partitions)
    spark.stop()


if __name__ == "__main__":
    main()
