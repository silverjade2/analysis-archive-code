"""Shared config for the loyal-user-prediction reproduction."""
from pathlib import Path
import numpy as np
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
LOGIN_WINDOW_DAYS = 180          # login_counts = logins in the last 6 months
N_TOTAL = 451_314                # all registered users
N_MODEL = 38_355                 # users with test grade + required profile (general track)

# recruiting seasons (major-company open recruiting) — from the original report
SEASONS = [
    ("2021-08-25", "2021-09-07"),
    ("2022-02-07", "2022-03-06"),
    ("2022-10-17", "2022-11-06"),
    ("2023-03-08", "2023-03-28"),
]
SEASONS = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in SEASONS]

# feature set used in the original notebook (33 columns incl. target)
NOTEBOOK_FEATURES = [
    "gender", "marketing_consent_yn", "career_year", "age", "join_year", "join_month",
    "days_since_last_login", "extra", "final_edu_level", "career_type",
    "total_apply_cnt", "apply_try_cnt", "apply_cnt", "test_cnt", "complete_cnt",
    "m_total_apply_cnt", "m_apply_try_cnt", "m_apply_cnt", "m_test_cnt", "m_complete_cnt",
    "apply_yn", "m_apply_yn", "login_counts", "acca_grade", "acca_t_score",
    "mental_health_grade", "acc_apply_counts", "midas_open_apply_yn",
    "midas_dechams_apply_yn", "midas_etc_apply_yn", "company_cnt", "user_cnt",
]
PREF_FEATURES = ["pref_salary_default_yn", "pref_welfare_cnt"]
TARGET = "matching_use_yn"
CAT_COLS = ["gender", "marketing_consent_yn", "extra", "final_edu_level", "career_type",
            "apply_yn", "m_apply_yn", "acca_grade", "mental_health_grade",
            "midas_open_apply_yn", "midas_dechams_apply_yn", "midas_etc_apply_yn",
            "pref_salary_default_yn"]

def day_index(ts):
    """days since SERVICE_START (int)."""
    if hasattr(ts, "dt"):
        return (pd.to_datetime(ts) - SERVICE_START).dt.days
    return (ts - SERVICE_START).days

N_DAYS = day_index(SNAPSHOT) + 1
