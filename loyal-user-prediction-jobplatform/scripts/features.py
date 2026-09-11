"""v1(snapshot)과 v2(시간 절단)가 공유하는 feature 생성. 각 유저의 cutoff 당일까지의 이벤트만 집계.
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
    """유저별로 day <= cutoff[user]인 이벤트 수를 센다. 플래그 열 / 행 mask는 선택."""
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

    # 로그인
    upto = logins.astype(bool) & (day <= cutoff[:, None])
    last = np.where(upto.any(1), N_DAYS - 1 - np.argmax(upto[:, ::-1], axis=1), users["join_day"].values)
    window = upto & (day > (cutoff - LOGIN_WINDOW_DAYS)[:, None])
    f = pd.DataFrame({"user": users["user"].values})
    f["days_since_last_login"] = np.maximum(cutoff - last, 0)   # 절단일이 가입일 전날이면(동의일 = 가입일) 0
    f["login_counts"] = window.sum(1)

    # 지원
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
    f["company_cnt"] = (ev_apply[m].groupby("user")["company_id"].nunique()
                        .reindex(range(n), fill_value=0).values)

    # 검사 응시, 알림 응답
    f["acc_apply_counts"] = _count_upto(ev_test, cutoff)
    f["user_cnt"] = _count_upto(ev_notice, cutoff)

    # 정적 속성
    static = ["gender", "marketing_consent_yn", "career_year", "age", "career_type", "extra",
              "final_edu_level", "acca_grade", "acca_t_score", "mental_health_grade",
              "pref_salary_default_yn", "pref_welfare_cnt", "matching_use_yn"]
    for c in static:
        f[c] = users[c].values
    jd = SERVICE_START + pd.to_timedelta(users["join_day"].values, unit="D")
    f["join_year"], f["join_month"] = jd.year, jd.month
    f["cutoff_day"] = cutoff
    return f
