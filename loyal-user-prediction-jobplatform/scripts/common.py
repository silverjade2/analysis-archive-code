"""공통 설정. 경로, 상수, 원본 노트북 feature 목록"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
RES = ROOT / "outputs" / "results"
for p in (DATA, FIG, RES):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
SERVICE_START = pd.Timestamp("2021-05-27")
SNAPSHOT = pd.Timestamp("2023-06-23")
LOGIN_WINDOW_DAYS = 180  # login_counts = 최근 6개월 로그인 수
N_TOTAL = 451_314
N_MODEL = 38_355  # 검사 등급 + 필수 프로필을 채운 유저 (일반 트랙)

# 공채 시즌. 원본 보고서 날짜
SEASONS = [
    ("2021-08-25", "2021-09-07"),
    ("2022-02-07", "2022-03-06"),
    ("2022-10-17", "2022-11-06"),
    ("2023-03-08", "2023-03-28"),
]
SEASONS = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in SEASONS]

# 원본 노트북 feature 32개
NOTEBOOK_FEATURES = [
    "gender",
    "marketing_consent_yn",
    "career_year",
    "age",
    "join_year",
    "join_month",
    "days_since_last_login",
    "extra",
    "final_edu_level",
    "career_type",
    "total_apply_cnt",
    "apply_try_cnt",
    "apply_cnt",
    "test_cnt",
    "complete_cnt",
    "m_total_apply_cnt",
    "m_apply_try_cnt",
    "m_apply_cnt",
    "m_test_cnt",
    "m_complete_cnt",
    "apply_yn",
    "m_apply_yn",
    "login_counts",
    "acca_grade",
    "acca_t_score",
    "mental_health_grade",
    "acc_apply_counts",
    "midas_open_apply_yn",
    "midas_dechams_apply_yn",
    "midas_etc_apply_yn",
    "company_cnt",
    "user_cnt",
]
PREF_FEATURES = ["pref_salary_default_yn", "pref_welfare_cnt"]
TARGET = "matching_use_yn"
CAT_COLS = [
    "gender",
    "marketing_consent_yn",
    "extra",
    "final_edu_level",
    "career_type",
    "apply_yn",
    "m_apply_yn",
    "acca_grade",
    "mental_health_grade",
    "midas_open_apply_yn",
    "midas_dechams_apply_yn",
    "midas_etc_apply_yn",
    "pref_salary_default_yn",
]


def day_index(ts):
    """SERVICE_START 이후 경과 일수"""
    if hasattr(ts, "dt"):
        return (pd.to_datetime(ts) - SERVICE_START).dt.days
    return (ts - SERVICE_START).days


N_DAYS = day_index(SNAPSHOT) + 1
