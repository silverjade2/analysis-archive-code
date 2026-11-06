"""경로, 업무 유형, 여러 스크립트가 함께 쓰는 함수"""

import csv
import hashlib
import json
import os
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from model_core import MAX_CHARS, english_ratio, is_english, plain_text  # noqa: F401
from roles import role_family

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "outputs" / "results"
# 공고 원문이 든 중간 파일. 저장소에 넣지 않는다
WORK_DIR = Path(os.environ.get("JOB_ROLE_WORK", ROOT / "work"))
T1 = WORK_DIR / "task1"
WORK = T1 / "labels_work_b"  # 학습용 입력과 라벨
SEAL = T1 / "labels_test_b"  # 평가 표본 입력과 라벨. 학습 코드는 읽지 않는다
MIN_BODY = 200
CROSS = ["2022-12", "2023-12", "2024-12"]
NORM = "regexp_replace(lower(coalesce({}, '')), '[^\\p{{L}}\\p{{N}}]+', '', 'g')"
US = "('us','usa','u.s.','u.s.a.','united states','united states of america')"

with open(DATA / "task_taxonomy.csv", encoding="utf-8") as _f:
    _TAX = list(csv.DictReader(_f))
IDS = [r["task_id"] for r in _TAX]  # 순서 고정
GROUP_OF = {r["task_id"]: r["task_group"] for r in _TAX}
GROUPS = list(dict.fromkeys(GROUP_OF.values()))

DATA_TITLE = re.compile(
    r"data (scien|analy|engineer)|analytics engineer|machine learning|(?<![a-z])ml (engineer|ops)|mlops"
    r"|business intelligence|(?<![a-z])bi (analyst|developer|engineer)|ai engineer|deep learning|nlp engineer"
    r"|데이터|머신러닝|딥러닝|분석가|사이언티스트|ai 엔지니어"
)


def norm_key(s: str | None) -> str:
    """중복 판정용 정규화: 소문자, 글자와 숫자만 (NORM과 같은 뜻)"""
    return re.sub(r"[\W_]+", "", (s or "").lower())


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def wdc_doc_id(df: pd.DataFrame) -> pd.Series:
    h = pd.util.hash_pandas_object(df.page_url + df.node, index=False).astype(str)
    return "wdc-" + df.crawl.astype(str) + "-" + h


def read_jsonl(path: Path) -> list[dict]:
    # 본문에 U+2028이 있어 splitlines()를 쓰지 않는다
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def multi_hot(tasks: pd.Series) -> np.ndarray:
    Y = np.zeros((len(tasks), len(IDS)))
    for i, t in enumerate(tasks.map(json.loads)):
        for x in t:
            Y[i, IDS.index(x)] = 1
    return Y


def primary_hot(p: pd.Series) -> np.ndarray:
    Y = np.zeros((len(p), len(IDS)))
    for i, x in enumerate(p):
        if x != "none":
            Y[i, IDS.index(x)] = 1
    return Y


def load_population() -> pd.DataFrame:
    """적용 대상 143,762건 (본문 200자 이상, 직무군 '그 밖' 제외)"""
    con = duckdb.connect()
    pop = con.execute(f"""
        SELECT page_url, node, crawl, title, description, org_name, host, role_family, dedup_key, lang
        FROM read_parquet('{T1}/wdc_jobs_dedup.parquet') WHERE body_chars >= {MIN_BODY} AND role_family <> '그 밖'
    """).df()
    pop["doc_id"] = wdc_doc_id(pop)
    pop["key2"] = [md5(norm_key(t) + norm_key((d or "")[:MAX_CHARS])) for t, d in zip(pop.title, pop.description)]
    pop["in_title"] = pop.title.fillna("").str.lower().str.contains(DATA_TITLE)
    return pop


def existing_keys(pop: pd.DataFrame, pool: pd.DataFrame) -> tuple[set, pd.DataFrame]:
    """라벨 있는 공고가 기준 데이터의 어느 dedup_key인지 찾는다 (doc_id, 제목 + 본문 앞부분 순)"""
    lab = pd.read_parquet(T1 / "labels_agent.parquet")[["doc_id"]].merge(pool, on="doc_id")
    w = lab[lab.source == "wdc"].copy()
    w["key2"] = [md5(norm_key(t) + norm_key((d or "")[:MAX_CHARS])) for t, d in zip(w.title, w.description)]
    m1 = w.merge(pop[["doc_id", "dedup_key"]], on="doc_id", how="left")
    miss = m1.dedup_key.isna()
    k2 = pop.drop_duplicates("key2").set_index("key2").dedup_key
    m1.loc[miss, "dedup_key"] = m1.loc[miss, "key2"].map(k2)
    m1["new_role"] = m1.title.map(role_family)
    return set(m1.dedup_key.dropna()), m1
