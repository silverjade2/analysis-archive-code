"""항목 점수 (원본 / 재설계) + 단계 판정

- 원본: 정상 경계 중심, 원 단위 편차에 sigmoid. lower/upper_limit 안 씀
- 재설계: d = 편차 / (경계~임계치 거리), d=0에서 100이 되게 정규화
"""

import numpy as np
import pandas as pd
from common import ITEMS, K_V2, PARAMS, PENALTY, STAGE_CAUTION, STAGE_STABLE

ORDER = {"안정": 0, "주의": 1, "경고": 2}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def item_score_current(x, p):
    if pd.isna(x) or p["weight"] == 0:
        return np.nan
    if p["normal_low"] <= x <= p["normal_high"]:
        return 100.0 * p["weight"]
    dev = x - p["normal_high"] if x > p["normal_high"] else p["normal_low"] - x  # 원 단위. limit 미사용 (원본 그대로)
    return 100.0 * (1.0 - sigmoid(p["steepness"] * dev)) * p["weight"]


def item_score_v2(x, p, k=K_V2):
    if pd.isna(x) or p["weight"] == 0:
        return np.nan
    if p["normal_low"] <= x <= p["normal_high"]:
        return 100.0 * p["weight"]
    if x > p["normal_high"]:
        span = p["upper_limit"] - p["normal_high"]
        d = (x - p["normal_high"]) / span if span > 0 else np.inf
    else:
        span = p["normal_low"] - p["lower_limit"]
        d = (p["normal_low"] - x) / span if span > 0 else np.inf  # spo2 상방, stress 하방은 span 0 -> 0점
    score = 100.0 * (1.0 - sigmoid(k * (d - 0.5))) / (1.0 - sigmoid(-0.5 * k))
    return score * p["weight"]


def combined(scores: dict) -> float:
    valid = [i for i in ITEMS if not pd.isna(scores[i])]
    if not valid:
        return np.nan
    return round(sum(scores[i] for i in valid) / sum(PARAMS[i]["weight"] for i in valid), 2)


def item_status(x, p):
    if pd.isna(x):
        return "데이터 없음"
    if x < p["lower_limit"] or x > p["upper_limit"]:
        return "경고"
    if x < p["normal_low"] or x > p["normal_high"]:
        return "주의"
    return "안정"


def stage_from_score(score):
    if pd.isna(score):
        return "데이터 없음"
    if score >= STAGE_STABLE:
        return "안정"
    if score >= STAGE_CAUTION:
        return "주의"
    return "경고"


def stage_from_items_original(statuses: list) -> str:
    # 원본 규칙: 결측 1개라도 있으면 데이터 없음
    if "데이터 없음" in statuses:
        return "데이터 없음"
    return max(statuses, key=ORDER.get)


def stage_from_items_v2(statuses: list) -> tuple:
    # 결측 제외 + missing_count
    present = [s for s in statuses if s != "데이터 없음"]
    if not present:
        return "데이터 없음", len(statuses)
    return max(present, key=ORDER.get), len(statuses) - len(present)


def penalty_factor(statuses: list) -> float:
    present = [s for s in statuses if s != "데이터 없음"]
    if not present:
        return np.nan
    return PENALTY[max(present, key=ORDER.get)]


def final_stage(score_stage, item_stage):
    # 단계 캡. 둘 중 나쁜 쪽
    if score_stage == "데이터 없음":
        return item_stage
    if item_stage == "데이터 없음":
        return score_stage
    return max(score_stage, item_stage, key=ORDER.get)


def score_frame(df: pd.DataFrame, k=K_V2) -> pd.DataFrame:
    out = df.copy()
    rows = []
    for _, r in df.iterrows():
        cur = {i: item_score_current(r[i], PARAMS[i]) for i in ITEMS}
        v2 = {i: item_score_v2(r[i], PARAMS[i], k) for i in ITEMS}
        st = [item_status(r[i], PARAMS[i]) for i in ITEMS]
        score_cur = combined(cur)
        score_v2 = combined(v2)
        factor = penalty_factor(st)
        score_pen = np.nan if pd.isna(score_cur) or pd.isna(factor) else round(score_cur * factor, 2)
        item_v2, n_missing = stage_from_items_v2(st)
        rec = {f"status_{i}": s for i, s in zip(ITEMS, st)}
        rec.update({f"score_v2_{i}": v2[i] for i in ITEMS})
        rec.update(
            score_current=score_cur,
            stage_current=stage_from_score(score_cur),
            score_penalized=score_pen,
            stage_penalized=stage_from_score(score_pen),
            score_v2=score_v2,
            stage_v2=stage_from_score(score_v2),
            item_stage_original=stage_from_items_original(st),
            item_stage_v2=item_v2,
            missing_count=n_missing,
            missing_items=",".join(i for i, s in zip(ITEMS, st) if s == "데이터 없음"),
        )
        rec["final_stage"] = final_stage(rec["stage_v2"], item_v2)
        rows.append(rec)
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
