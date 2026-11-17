"""확인용. 판정 스크립트에 truth_ 없음, 중간 테이블에 정답 컬럼 없음, 02 짝짓기 = 루프 결과"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from common import DATA, DEVICE, PAIR_MAX_SEC, USER  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent / "scripts"
JUDGE = ["02_pairs.py", "03_topics.py", "04_flags.py", "05_retry.py", "06_priority.py", "09_ngram.py"]

bad = [f for f in JUDGE if "truth_" in (SCRIPTS / f).read_text()]
print("truth_ in judge scripts:", bad or "none")

for f in ["08_misfire.py"]:
    src = (SCRIPTS / f).read_text()
    order_ok = src.index('rules["upper"]') < src.index('truth_misfire"])')
    print(f"{f}: rules defined before truth is read:", order_ok)

leaked = []
for name in ["turn_pairs.csv.gz", "topic_match.csv.gz", "turn_flags.csv.gz", "turn_retry.csv.gz"]:
    cols = pd.read_csv(DATA / name, nrows=1).columns
    found = [c for c in cols if c.startswith("truth_")]
    leaked += found
    print(name, "truth columns:", found or "none")

logs = pd.read_csv(DATA / "logs.csv.gz", keep_default_na=False, usecols=["row_id", "session_id", "log_type", "ts"])
logs["ts"] = pd.to_datetime(logs["ts"])
pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", dtype={"resp_row_id": "Int64"}).set_index("turn_id")

# 제어 코드(2)나 음성 참조 행(3)이 낀 세션 3개, 없는 세션 2개, 응답으로 안 붙은 기기 발화가 있는 세션 5개
sess = logs.groupby("session_id")["log_type"].agg(lambda s: (s == 2).any() or (s == 3).any())
picked = list(sess[sess].index[:3]) + list(sess[~sess].index[:2])
late = logs[logs.log_type == DEVICE].merge(pairs[["resp_row_id"]], left_on="row_id", right_on="resp_row_id", how="left")
picked += list(late.loc[late["resp_row_id"].isna(), "session_id"].unique()[:5])

mismatch = 0
for sid in picked:
    rows = logs[(logs.session_id == sid) & logs.log_type.isin([USER, DEVICE, 2])].sort_values(["ts", "row_id"])
    rows = list(rows.itertuples())
    for i, r in enumerate(rows):
        if r.log_type != USER:
            continue
        expect = ""
        for nxt in rows[i + 1 :]:
            if nxt.log_type == 2:
                continue
            if nxt.log_type == DEVICE and (nxt.ts - r.ts).total_seconds() <= PAIR_MAX_SEC:
                expect = str(nxt.row_id)
            break
        got = pairs.loc[r.row_id, "resp_row_id"]
        if ("" if pd.isna(got) else str(got)) != expect:
            mismatch += 1
print(f"pairing checked on {len(picked)} sessions, mismatches {mismatch}")

assert not bad and order_ok and not leaked and mismatch == 0, "verify failed"
