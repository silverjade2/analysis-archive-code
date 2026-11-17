"""경로, 기준일, 소스 설정, raw 저장 규약(schema 선언, 호출 기록). 스크립트 위치 기준이라 어디서 실행해도 됨"""

import json
import math
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "source"  # 가짜 소스: 시트 응답, 캘린더 서버, 설문 DB
RAW = DATA / "raw"
CALLS = DATA / "calls.jsonl"
DB = DATA / "warehouse.duckdb"
SQL = ROOT / "sql"
RES = ROOT / "outputs" / "results"

SEED = 42
TZ = "Asia/Seoul"
TODAY = date(2026, 10, 20)  # 첫 수집일
COLLECTED_AT = datetime(2026, 10, 20, 20, 30, tzinfo=ZoneInfo(TZ))  # 고정해야 parquet 가 실행마다 같음

LOOKBACK_DAYS = 56
LOOKAHEAD_DAYS = 90
DAILY_CALL_LIMIT = {"golf": 20, "calendar": 60, "saju": 20}

# 스코어 앱이 쓰는 시트 탭 4개와 열 (1행 헤더)
GOLF_TABS = {
    "Player": ["id", "name", "updatedAt"],
    "Course": ["id", "name", "pars", "createdAt", "updatedAt", "deletedAt"],
    "Round": ["id", "date", "courseId", "memo", "createdAt", "updatedAt", "deletedAt"],
    "Score": ["id", "roundId", "playerId", "hole", "strokes", "updatedAt"],
}

# 기준 골프장 홀 1~18 의 par, HDCP. sql/stg/020_golf_course_holes.sql 과 같은 값
HOME_PARS = [4, 5, 3, 4, 4, 3, 4, 5, 4, 4, 4, 3, 5, 4, 4, 3, 5, 4]
HOME_HDCP = [7, 3, 15, 1, 11, 17, 5, 9, 13, 8, 12, 16, 2, 10, 4, 18, 6, 14]

ARROW_TYPES = {
    "VARCHAR": pa.string(),
    "BIGINT": pa.int64(),
    "DOUBLE": pa.float64(),
    "BOOLEAN": pa.bool_(),
    "DATE": pa.date32(),
    "TIMESTAMPTZ": pa.timestamp("us", tz="UTC"),
}
META_COLUMNS = {"_snapshot_date": "DATE", "_collected_at": "TIMESTAMPTZ"}


class SchemaError(ValueError):
    pass


class DailyLimitExceeded(RuntimeError):
    pass


def _isnull(v) -> bool:
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v))


def _convert(v, typ: str):
    if typ == "VARCHAR":
        return str(v)
    if typ == "BIGINT":
        return int(v)
    if typ == "DOUBLE":
        return float(v)
    if typ == "BOOLEAN":
        return bool(v)
    if typ == "DATE":
        return v.date() if isinstance(v, datetime) else v if isinstance(v, date) else date.fromisoformat(str(v)[:10])
    d = v if isinstance(v, datetime) else datetime.fromisoformat(str(v))
    return (d if d.tzinfo else d.replace(tzinfo=ZoneInfo(TZ))).astimezone(timezone.utc)


@dataclass(frozen=True)
class Schema:
    """컬럼 이름 -> 타입. required 는 NULL 불가"""

    columns: dict
    required: tuple = ()

    def with_meta(self) -> dict:
        return {**self.columns, **META_COLUMNS}

    def conform(self, df: pd.DataFrame) -> pa.Table:
        """선언한 컬럼만 선언한 타입으로. 선언 밖 컬럼이 오면 저장하지 않고 실패"""
        extra = [c for c in df.columns if c not in self.with_meta()]
        if extra:
            raise SchemaError(f"선언하지 않은 컬럼: {extra}")
        arrays = []
        for col, typ in self.with_meta().items():
            out = []
            for i, v in enumerate(df[col].tolist() if col in df.columns else [None] * len(df)):
                if _isnull(v):
                    if col in self.required:
                        raise SchemaError(f"{col} {i}번째 행이 비어 있음")
                    out.append(None)
                else:
                    out.append(_convert(v, typ))
            arrays.append(pa.array(out, type=ARROW_TYPES[typ]))
        return pa.Table.from_arrays(
            arrays, schema=pa.schema([(c, ARROW_TYPES[t]) for c, t in self.with_meta().items()])
        )


def save_snapshot(table: str, df: pd.DataFrame, schema: Schema) -> int:
    """data/raw/<table>/<yyyy-mm-dd>.parquet. 같은 날 다시 돌리면 덮어씀"""
    df = df.copy()
    df["_snapshot_date"] = TODAY
    df["_collected_at"] = COLLECTED_AT
    data = schema.conform(df)
    path = RAW / table / f"{TODAY.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, path, compression="zstd", compression_level=9)
    return data.num_rows


class CallLog:
    """외부 호출 1건 = calls.jsonl 한 줄. 소스별 하루 상한은 호출 전에 확인"""

    def __init__(self, path: Path = CALLS):
        self.path = path
        self.counts = Counter()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                self.counts[json.loads(line)["source"]] += 1

    @contextmanager
    def call(self, source: str, op: str):
        limit = DAILY_CALL_LIMIT[source]
        if self.counts[source] + 1 > limit:
            raise DailyLimitExceeded(f"{source}: 오늘 외부 호출 {self.counts[source]}회, 상한 {limit}회")
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"source": source, "op": op, "status": status}, ensure_ascii=False) + "\n")
            self.counts[source] += 1
