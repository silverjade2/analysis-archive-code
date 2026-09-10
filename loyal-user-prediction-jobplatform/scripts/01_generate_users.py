"""01. 가상 유저 생성.

두 층으로 구성한다.
  (a) 전체 가입 유저(N_TOTAL): 저니 상태와 마지막 로그인 날짜를 가진다. 저니맵 절에서만 쓴다.
  (b) 모델링 대상 유저(N_MODEL): 역량 진단 검사를 치르고 필수 프로필을 채운 유저. 이들에게는
      이벤트 타임라인(가입 → 검사 → 프로필 → 동의)과 날짜가 붙은 로그인 / 지원 / 검사 / 알림
      이벤트를 생성해, feature를 snapshot(v1, 원래 노트북 방식)으로도 시점 절단(v2)으로도
      계산할 수 있게 한다.

세 가지 구조를 의도적으로 심는다.
  1. 동의 → 로그인. 동의한 유저는 검사 결과표 확인과 지원 관리를 위해 다시 들어오므로,
     로그인 활동은 타깃이 결정된 이후에 오른다.
  2. 시즌 코호트. 대규모 공채 시즌 직전에 가입한 유저는 "시즌 가입자"인 경우가 많다.
     한 번 지원하고 떠난다. 가입월이 이 동기의 대리 변수다.
  3. 선호 정보 완성이 진짜 동인이다. 연봉 / 복지 선호를 기본값으로 두지 않고 채우면
     동의 확률이 오른다. 저니맵이 찾아낸 신호다.

출력 (data/):
  population.csv      전체 가입 유저, 저니 상태 + 마지막 로그인
  users.csv           모델링 대상 유저, 정적 속성 + 잠재 정답 변수
  logins.npz          모델링 대상 유저 × 일 로그인 행렬 (uint8)
  events_apply.csv    날짜가 붙은 지원 이벤트 (퍼널 플래그, midas 플래그, 회사 id)
  events_test.csv     날짜가 붙은 역량 진단 검사 응시
  events_notice.csv   날짜가 붙은 알림 응답
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA, SEED, SERVICE_START, SNAPSHOT, N_TOTAL, N_MODEL, SEASONS,
                    N_DAYS, day_index)

rng = np.random.default_rng(SEED)
sig = lambda x: 1 / (1 + np.exp(-x))

# ---------------------------------------------------------------- 달력 보조 변수
days = pd.date_range(SERVICE_START, SNAPSHOT, freq="D")
assert len(days) == N_DAYS
season_day = np.zeros(N_DAYS, dtype=bool)          # 시즌 기간 안
pre_season = np.zeros(N_DAYS, dtype=bool)          # 시즌 시작 전 21일 (가입 급증 구간)
for a, b in SEASONS:
    ia, ib = day_index(a), day_index(b)
    season_day[ia:ib + 1] = True
    pre_season[max(0, ia - 21):ia + 1] = True
season_end_after = np.full(N_DAYS, N_DAYS, dtype=int)  # 다음 시즌 종료일의 day index
for a, b in reversed(SEASONS):
    ia, ib = day_index(a), day_index(b)
    season_end_after[:ib + 1] = np.minimum(season_end_after[:ib + 1], ib)


def sample_join_days(n):
    """가입일 분포: 선형 증가 + 시즌 직전 급증."""
    w = np.linspace(0.6, 1.4, N_DAYS)
    w = w * np.where(pre_season, 3.5, 1.0)
    w[-1] = 0  # snapshot 당일 가입은 없다
    w = w / w.sum()
    return rng.choice(N_DAYS, size=n, p=w)


# ================================================================ (a) 전체 가입 유저
print("generating full population ...")
join_pop = sample_join_days(N_TOTAL)
motive_pop = np.where(pre_season[join_pop], rng.random(N_TOTAL) < 0.6, rng.random(N_TOTAL) < 0.25)
# status: 0 가입만, 1 검사만, 2 프로필만, 3 검사+프로필 (모델링 대상 유저)
p_org = np.array([0.72, 0.06, 0.08, 0.14])
p_sea = np.array([0.90, 0.05, 0.035, 0.015])
u = rng.random(N_TOTAL)
cum_org, cum_sea = np.cumsum(p_org), np.cumsum(p_sea)
status_pop = np.where(motive_pop, np.searchsorted(cum_sea, u), np.searchsorted(cum_org, u))
status_pop = np.clip(status_pop, 0, 3)
# 모델링 대상 유저 수를 정확히 N_MODEL로 맞춘다
idx3 = np.flatnonzero(status_pop == 3)
if len(idx3) > N_MODEL:
    status_pop[rng.choice(idx3, len(idx3) - N_MODEL, replace=False)] = 2
elif len(idx3) < N_MODEL:
    cand = np.flatnonzero(status_pop == 2)
    status_pop[rng.choice(cand, N_MODEL - len(idx3), replace=False)] = 3
# 모델링 대상이 아닌 유저의 마지막 로그인: 대부분 가입일, 일부는 나중에 재방문
last_pop = join_pop.copy()
returned = rng.random(N_TOTAL) < np.where(status_pop == 0, 0.15, 0.55)
extra = rng.integers(1, 400, N_TOTAL)
last_pop = np.where(returned, np.minimum(join_pop + extra, N_DAYS - 1), join_pop)
population = pd.DataFrame({
    "join_date": days[join_pop],
    "season_joiner": motive_pop.astype(int),
    "status": pd.Categorical.from_codes(status_pop, ["join_only", "test_only", "profile_only", "test_and_profile"]),
    "last_login_date": days[last_pop],
})
# 45만 행을 그대로 두면 15MB이고 4열 조합의 완전 중복이 대부분이라 조합별 인원(n)으로 집계해 저장한다
population = (population.groupby(["join_date", "season_joiner", "status", "last_login_date"], observed=True)
              .size().reset_index(name="n"))

# ================================================================ (b) 모델링 대상 유저
print("generating modeling population ...")
n = N_MODEL
sel = np.flatnonzero(status_pop == 3)
join = join_pop[sel]
season_joiner = motive_pop[sel]
commit = rng.normal(0, 1, n)                                    # 잠재 성실도

# --- 정적 속성
gender = rng.choice(["남", "여"], n, p=[0.45, 0.55])
age = np.clip(rng.normal(27, 3.5, n).round(), 19, 45).astype(int)
career_year = np.where(rng.random(n) < 0.75, 0, rng.integers(1, 9, n))
career_type = np.where(career_year == 0, "신입", "경력")
extra_lv = rng.choice(["고등학교", "L1", "L2", "L3", "L4", "L5", "L6", "해외대"], n,
                      p=[0.05, 0.06, 0.10, 0.22, 0.24, 0.18, 0.11, 0.04])
final_edu_level = np.select(
    [extra_lv == "고등학교", np.isin(extra_lv, ["L1", "L2"]), rng.random(n) < 0.12],
    ["고졸", "전문대졸", "석사"], default="대졸")
acca_t_score = np.clip(rng.normal(60 + 3 * commit, 12, n).round(), 5, 100).astype(int)
acca_grade = pd.cut(acca_t_score, [-1, 40, 52, 64, 76, 101], labels=["E", "D", "C", "B", "A"]).astype(str)
mental_health_grade = rng.choice(["양호", "주의", "위험"], n, p=[0.80, 0.15, 0.05])

# --- 선호 정보 (진짜 동인)
pref_salary_default = rng.random(n) < sig(0.0 - 0.9 * commit + 0.6 * season_joiner)   # 1 = 기본값 그대로
pref_welfare_cnt = np.clip(rng.poisson(np.exp(1.2 + 0.35 * commit - 0.4 * season_joiner)), 0, 10)
pref_complete = (~pref_salary_default) & (pref_welfare_cnt >= 5)
marketing_consent = rng.random(n) < sig(0.2 + 0.7 * commit)

# --- 타임라인: 가입 → 첫 검사 → 프로필 완성
test1 = join + 1 + rng.geometric(1 / 5, n)
profile = test1 + rng.geometric(1 / 4, n)
profile = np.minimum(profile, N_DAYS - 2)
test1 = np.minimum(test1, profile)

# --- 동의 (타깃): P(동의 | 프로필 & 검사 완료)
z_score = (acca_t_score - 60) / 12
logit = (1.30 * pref_complete + 0.35 * z_score - 0.80 * season_joiner
         + 0.55 * marketing_consent + 0.30 * commit)
b0 = 0.0
for _ in range(60):  # 기저율 81%가 되도록 절편을 푼다
    b0 -= (sig(b0 + logit).mean() - 0.81) * 4
p_true = sig(b0 + logit)
consent = rng.random(n) < p_true
delay = np.rint(np.exp(rng.normal(1.1, 1.0, n))).astype(int)          # 중앙값 약 3일, 긴 꼬리
consent_day = np.where(consent, np.minimum(profile + delay, N_DAYS - 1), -1)
print(f"  consent rate {consent.mean():.4f} (intercept {b0:.3f})")

# --- 로그인 행렬 (유저 × 일)
print("simulating logins ...")
day_grid = np.arange(N_DAYS)[None, :]
base = np.exp(-2.4 + 0.45 * commit)[:, None]                       # 하루 약 0.09회 로그인
lam = np.broadcast_to(base, (n, N_DAYS)).copy()
lam *= np.where(season_day[None, :], 3.0, 1.0)                     # 시즌에는 모두 활동이 는다
# 시즌 가입자는 가입 후 첫 시즌이 끝나면 떠난다
leave_day = season_end_after[join]
lam *= np.where(season_joiner[:, None] & (day_grid > leave_day[:, None]), 0.12, 1.0)
# 동의 → 로그인: 결과표 확인 급증 + 지속적 상승 (지원 관리, 알림)
after = (consent_day[:, None] >= 0) & (day_grid > consent_day[:, None])
lam *= np.where(after, 2.2, 1.0)
lam = np.where(day_grid < join[:, None], 0.0, lam)                  # 가입 전 로그인은 없다
logins = (rng.random((n, N_DAYS)) < (1 - np.exp(-lam))).astype(np.uint8)
rows = np.arange(n)
logins[rows, join] = 1                                             # 가입일은 로그인으로 센다
logins[rows, profile] = 1
pos = np.flatnonzero(consent)
for d in (1, 2):                                                    # 결과표 확인 급증
    dd = np.minimum(consent_day[pos] + d, N_DAYS - 1)
    logins[pos, dd] = np.where(rng.random(len(pos)) < 0.7, 1, logins[pos, dd])

# --- 역량 진단 검사 응시 (날짜 포함)
n_pre = 1 + rng.poisson(0.35, n)                                   # 프로필 완성 전 응시 횟수
n_post = np.where(consent, rng.poisson(0.5, n), rng.poisson(0.15, n))
test_rows = []
for i in range(n):
    d_pre = np.concatenate([[test1[i]], rng.integers(test1[i], profile[i] + 1, n_pre[i] - 1)])
    start = consent_day[i] + 1 if consent[i] else profile[i] + 1
    d_post = rng.integers(start, N_DAYS, n_post[i]) if (n_post[i] > 0 and start < N_DAYS) else np.array([], int)
    for d in np.concatenate([d_pre, d_post]):
        test_rows.append((i, int(d)))
events_test = pd.DataFrame(test_rows, columns=["user", "day"])

# --- 지원 이벤트 (날짜 포함, 퍼널 플래그)
n_apply_pre = rng.poisson(np.exp(-0.2 + 0.5 * commit + 0.6 * season_joiner))
n_apply_post = np.where(consent, rng.poisson(np.exp(0.2 + 0.3 * commit)), rng.poisson(0.3))
n_apply_post = np.where(season_joiner & ~consent, rng.poisson(0.05, n), n_apply_post)
apply_rows = []
for i in range(n):
    k1, k2 = n_apply_pre[i], n_apply_post[i]
    d1 = rng.integers(profile[i], min(N_DAYS, profile[i] + 60), k1) if k1 else np.array([], int)
    start = consent_day[i] + 1 if consent[i] else profile[i] + 1
    d2 = rng.integers(start, N_DAYS, k2) if (k2 and start < N_DAYS) else np.array([], int)
    for d in np.concatenate([d1, d2]):
        is_midas = rng.random() < 0.30
        kind = rng.choice(["open", "dechams", "etc"], p=[0.5, 0.2, 0.3]) if is_midas else "none"
        tried = rng.random() < 0.85
        applied = tried and rng.random() < 0.80
        tested = applied and rng.random() < 0.60
        completed = tested and rng.random() < 0.75
        apply_rows.append((i, int(d), int(is_midas), kind, int(tried), int(applied),
                           int(tested), int(completed), int(rng.integers(1, 400))))
events_apply = pd.DataFrame(apply_rows, columns=["user", "day", "midas", "midas_kind", "try",
                                                 "apply", "test", "complete", "company_id"])

# --- 알림 응답 (날짜 포함): 대부분 추천 풀에 들어간 결과
n_notice = np.where(consent, rng.poisson(0.8, n), rng.poisson(0.15, n))
notice_rows = []
for i in range(n):
    if n_notice[i] == 0:
        continue
    start = consent_day[i] + 1 if consent[i] else profile[i] + 1
    if start >= N_DAYS:
        continue
    for d in rng.integers(start, N_DAYS, n_notice[i]):
        notice_rows.append((i, int(d)))
events_notice = pd.DataFrame(notice_rows, columns=["user", "day"])

users = pd.DataFrame({
    "user": rows,
    "gender": gender, "age": age, "career_year": career_year, "career_type": career_type,
    "extra": extra_lv, "final_edu_level": final_edu_level,
    "acca_grade": acca_grade, "acca_t_score": acca_t_score, "mental_health_grade": mental_health_grade,
    "marketing_consent_yn": np.where(marketing_consent, "Y", "N"),
    "pref_salary_default_yn": np.where(pref_salary_default, "Y", "N"),
    "pref_welfare_cnt": pref_welfare_cnt,
    "join_day": join, "test1_day": test1, "profile_day": profile, "consent_day": consent_day,
    "matching_use_yn": consent.astype(int),
    # 잠재 정답 변수: 재현 결과 채점에만 쓰고 feature로는 쓰지 않는다
    "truth_p_consent": p_true, "truth_commit": commit, "truth_season_joiner": season_joiner.astype(int),
    "truth_pref_complete": pref_complete.astype(int),
})

population.to_csv(DATA / "population.csv", index=False)
users.to_csv(DATA / "users.csv", index=False)
np.savez_compressed(DATA / "logins.npz", logins=logins)
events_test.to_csv(DATA / "events_test.csv", index=False)
events_apply.to_csv(DATA / "events_apply.csv", index=False)
events_notice.to_csv(DATA / "events_notice.csv", index=False)
print(f"saved: population {int(population.n.sum()):,} ({len(population):,} rows), users {len(users):,}, logins {logins.shape}, "
      f"tests {len(events_test):,}, applies {len(events_apply):,}, notices {len(events_notice):,}")
