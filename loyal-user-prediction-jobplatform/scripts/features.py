"""v1(snapshot), v2(시간 절단) 공용 feature 생성

build_features(cutoff): 유저별 cutoff 당일까지 이벤트로 feature 32 + 선호 2. 전원 cutoff = snapshot이면 v1
"""

import numpy as np
import pandas as pd
from common import DATA, LOGIN_WINDOW_DAYS, N_DAYS, SERVICE_START


def load():
    users = pd.read_csv(DATA / "users.csv")
    logins = np.load(DATA / "logins.npz")["logins"]
    ev_apply = pd.read_csv(DATA / "events_apply.csv")
    ev_test = pd.read_csv(DATA / "events_test.csv")
    ev_notice = pd.read_csv(DATA / "events_notice.csv")
    return users, logins, ev_apply, ev_test, ev_notice


def _count_upto(ev, cutoff, col=None, mask=None):
    """유저별 day <= cutoff[user] 이벤트 수"""
    m = ev["day"].values <= cutoff[ev["user"].values]
    if mask is not None:
        m &= mask
    w = ev[col].values if col else 1
    return np.bincount(ev["user"].values[m], weights=(w[m] if col else None), minlength=len(cutoff)).astype(int)


def build_features(cutoff, users=None, logins=None, ev_apply=None, ev_test=None, ev_notice=None):
    if users is None:
        users, logins, ev_apply, ev_test, ev_notice = load()
    n = len(users)
    cutoff = np.asarray(cutoff, dtype=int)
    day = np.arange(N_DAYS)[None, :]

    upto = logins.astype(bool) & (day <= cutoff[:, None])
    last = np.where(upto.any(1), N_DAYS - 1 - np.argmax(upto[:, ::-1], axis=1), users["join_day"].values)
    window = upto & (day > (cutoff - LOGIN_WINDOW_DAYS)[:, None])
    f = pd.DataFrame({"user": users["user"].values})
    f["days_since_last_login"] = np.maximum(
        cutoff - last, 0
    )  # 동의일 = 가입일이면 절단일이 가입일 전날 -> 음수. 0으로 clip
    f["login_counts"] = window.sum(1)

    mid = ev_apply["midas"].values == 1
    f["total_apply_cnt"] = _count_upto(ev_apply, cutoff)
    f["apply_try_cnt"] = _count_upto(ev_apply, cutoff, "try")
    f["apply_cnt"] = _count_upto(ev_apply, cutoff, "apply")
    f["test_cnt"] = _count_upto(ev_apply, cutoff, "test")
    f["complete_cnt"] = _count_upto(ev_apply, cutoff, "complete")
    f["m_total_apply_cnt"] = _count_upto(ev_apply, cutoff, mask=mid)
    f["m_apply_try_cnt"] = _count_upto(ev_apply, cutoff, "try", mask=mid)
    f["m_apply_cnt"] = _count_upto(ev_apply, cutoff, "apply", mask=mid)
    f["m_test_cnt"] = _count_upto(ev_apply, cutoff, "test", mask=mid)
    f["m_complete_cnt"] = _count_upto(ev_apply, cutoff, "complete", mask=mid)
    f["apply_yn"] = np.where(f["apply_cnt"] > 0, "Y", "N")
    f["m_apply_yn"] = np.where(f["m_apply_cnt"] > 0, "Y", "N")
    for kind in ("open", "dechams", "etc"):
        c = _count_upto(ev_apply, cutoff, mask=(ev_apply["midas_kind"].values == kind))
        f[f"midas_{kind}_apply_yn"] = np.where(c > 0, "Y", "N")
    m = ev_apply["day"].values <= cutoff[ev_apply["user"].values]
    f["company_cnt"] = ev_apply[m].groupby("user")["company_id"].nunique().reindex(range(n), fill_value=0).values

    f["acc_apply_counts"] = _count_upto(ev_test, cutoff)
    f["user_cnt"] = _count_upto(ev_notice, cutoff)

    static = [
        "gender",
        "marketing_consent_yn",
        "career_year",
        "age",
        "career_type",
        "extra",
        "final_edu_level",
        "acca_grade",
        "acca_t_score",
        "mental_health_grade",
        "pref_salary_default_yn",
        "pref_welfare_cnt",
        "matching_use_yn",
    ]
    for c in static:
        f[c] = users[c].values
    jd = SERVICE_START + pd.to_timedelta(users["join_day"].values, unit="D")
    f["join_year"], f["join_month"] = jd.year, jd.month
    f["cutoff_day"] = cutoff
    return f
