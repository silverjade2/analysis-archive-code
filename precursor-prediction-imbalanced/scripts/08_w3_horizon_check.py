"""06의 W3 순위 확인용. 이벤트까지 남은 일수별 feature 평균, test 구간 단일 feature AUC"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

TEST_START_DAY = 150
COLS = ["w3_cnt_7d", "w3_cnt_14d", "w3_slope_7d", "u1_std_7d", "u1_std_14d", "u1_mean_7d", "w7_cnt_7d"]
base = Path(__file__).resolve().parents[1]

table = pd.read_csv(base / "data" / "features.csv", parse_dates=["event_date"])
gt = pd.read_csv(base / "data" / "ground_truth.csv", parse_dates=["severe_event_date"])
gt["event_day"] = (gt["severe_event_date"] - table["event_date"].min()).dt.days + table["day"].min()
m = table.merge(gt[["device_id", "group", "event_day"]], on="device_id")
m["days_to_event"] = m["event_day"] - m["day"]

pos = m[m["target"] == 1]
by_h = pos.groupby("days_to_event")[COLS].mean().round(2)
by_h.loc["negative_all"] = m.loc[m["target"] == 0, COLS].mean().round(2)
by_h.loc["negative_confounder"] = m.loc[(m["target"] == 0) & (m["group"] == "confounder"), COLS].mean().round(2)
by_h.index.name = "days_to_event"
by_h.reset_index().to_csv(base / "outputs" / "results" / "w3_by_horizon.csv", index=False)
print(by_h.to_string())

test = m[m["day"] >= TEST_START_DAY]
auc = pd.DataFrame(
    {"feature": COLS, "test_auc_single": [round(roc_auc_score(test["target"], test[c]), 4) for c in COLS]}
)
auc.to_csv(base / "outputs" / "results" / "single_feature_auc.csv", index=False)
print(auc.to_string(index=False))
