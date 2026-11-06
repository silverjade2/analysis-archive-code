"""라벨 조각 (out/*.jsonl)을 검사해 한 파일로. train이면 학습용, test면 평가 표본"""

import json
import sys

import pandas as pd
from common import IDS, SEAL, WORK, read_jsonl


def main() -> None:
    which = sys.argv[1]
    base, out_name = (WORK, "labels_train_new.parquet") if which == "train" else (SEAL, "labels_test.parquet")
    rows, bad = [], []
    for f in sorted((base / "out").glob("*.jsonl")):
        for n, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad.append((f.name, n, "JSON 아님"))
                continue
            tasks = d.get("tasks") or []
            pt = d.get("primary_task")
            unknown = [t for t in tasks if t not in IDS]
            if unknown or (pt != "none" and pt not in IDS):
                bad.append((f.name, n, f"모르는 id {unknown or pt}"))
                continue
            if pt != "none" and pt not in tasks:
                tasks = tasks + [pt]  # 대표 업무는 목록에도 있어야 한다
            if pt == "none":
                tasks = []
            rows.append({"doc_id": d["doc_id"], "tasks": json.dumps(tasks), "primary_task": pt, "chunk": f.stem})
    df = pd.DataFrame(rows).drop_duplicates("doc_id", keep="last")
    expected = set()
    for f in sorted((base / "in").glob("*.jsonl")):
        if (base / "out" / f.name).exists():
            expected |= {d["doc_id"] for d in read_jsonl(f)}
    missing = expected - set(df.doc_id)
    extra = set(df.doc_id) - expected
    df = df[df.doc_id.isin(expected)]
    df.to_parquet(base / out_name, index=False)
    n_chunks = len(list((base / "out").glob("*.jsonl")))
    print(
        f"{which}: 라벨 {len(df):,}건 저장 ({n_chunks}묶음), 형식 오류 {len(bad)}건, "
        f"빠진 공고 {len(missing)}건, 입력에 없는 doc_id {len(extra)}건"
    )
    if bad:
        print("형식 오류:", bad[:10])
    if missing:
        print("빠진 공고 (예):", sorted(missing)[:10])


if __name__ == "__main__":
    main()
