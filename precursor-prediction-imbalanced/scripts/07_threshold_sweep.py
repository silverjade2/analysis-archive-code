# 임계값 스윕: no_handling 모델의 테스트 구간 점수로 precision / recall / 일평균 경보 건수 계산
# 출력: outputs/results/threshold_sweep.json

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TEST_START_DAY = 150
base = Path(__file__).resolve().parents[1]
res_dir = base / "outputs" / "results"
res_dir.mkdir(parents=True, exist_ok=True)

table = pd.read_csv(base / "data" / "features.csv", parse_dates=["event_date"])
feature_cols = [c for c in table.columns if c not in ("device_id", "event_date", "day", "target")]
test = table[table["day"] >= TEST_START_DAY]
y = test["target"].to_numpy()
n_days = test["day"].nunique()

model = joblib.load(base / "data" / "model_no_handling.joblib")
score = model.predict_proba(test[feature_cols])[:, 1]

# "상위 k건 경보" 로그 격자로 임계값 추출, 소수 경보(고정밀) 구간이 촘촘해짐
sorted_scores = np.sort(score)[::-1]
ks = np.unique(np.geomspace(1, len(score) // 2, 220).astype(int))
thresholds = np.unique(sorted_scores[ks - 1])[::-1]

rows = []
for t in thresholds:
    pred = score >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    if tp + fp == 0:
        continue
    rows.append({
        "threshold": round(float(t), 6),
        "precision": round(tp / (tp + fp), 4),
        "recall": round(tp / (tp + fn), 4),
        "alertsPerDay": round((tp + fp) / n_days, 2),
        "tp": tp,
        "fp": fp,
    })

payload = {
    "meta": {
        "positives": int(y.sum()),
        "rows": int(len(y)),
        "days": int(n_days),
        "baseRate": round(float(y.mean()), 6),
    },
    "points": rows,
}
out = res_dir / "threshold_sweep.json"
out.write_text(json.dumps(payload, ensure_ascii=False))
print(f"points: {len(rows)}, positives: {y.sum()} / {len(y)}, days: {n_days}")
print(f"saved: {out}")
