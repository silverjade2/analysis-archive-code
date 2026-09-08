"""01 — Generate synthetic users.

Two layers:
  (a) the full registered population (N_TOTAL) with a journey status and last-login date,
      used only for the journey-map section;
  (b) the modeling population (N_MODEL): users who took the competency test and filled the
      required profile fields. For these we generate an event timeline (join → test → profile
      → consent) plus dated login / apply / test / notification events, so that features can
      be computed either as a snapshot (v1, as in the original notebook) or cut at a time
      point (v2).

Three structures are planted on purpose:
  1. consent → login. Users who consent come back to check their test report and manage
     applications, so their login activity rises AFTER the target is decided.
  2. season cohort. Users who join right before a major open-recruiting season are more often
     "season joiners": they apply once and leave. Join month is a proxy for this motive.
  3. preference completion is a real driver. Filling salary / welfare preferences (instead of
     leaving defaults) raises the consent probability — the signal the journey map found.

Outputs (data/):
  population.csv      full population, journey status + last login
  users.csv           modeling population, static attributes + latent truth
  logins.npz          modeling population × day login matrix (uint8)
  events_apply.csv    dated apply events (funnel flags, midas flag, company id)
  events_test.csv     dated competency-test attempts
  events_notice.csv   dated notification responses
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

# ---------------------------------------------------------------- calendar helpers
days = pd.date_range(SERVICE_START, SNAPSHOT, freq="D")
assert len(days) == N_DAYS
season_day = np.zeros(N_DAYS, dtype=bool)          # inside a season window
pre_season = np.zeros(N_DAYS, dtype=bool)          # 21 days before a season start (join spike)
for a, b in SEASONS:
    ia, ib = day_index(a), day_index(b)
    season_day[ia:ib + 1] = True
    pre_season[max(0, ia - 21):ia + 1] = True
season_end_after = np.full(N_DAYS, N_DAYS, dtype=int)  # day index of the end of the next season
for a, b in reversed(SEASONS):
    ia, ib = day_index(a), day_index(b)
    season_end_after[:ib + 1] = np.minimum(season_end_after[:ib + 1], ib)


def sample_join_days(n):
    """join-day distribution: linear growth + spikes before seasons."""
    w = np.linspace(0.6, 1.4, N_DAYS)
    w = w * np.where(pre_season, 3.5, 1.0)
    w[-1] = 0  # nobody joins on the snapshot day itself
    w = w / w.sum()
    return rng.choice(N_DAYS, size=n, p=w)


# ================================================================ (a) full population
print("generating full population ...")
join_pop = sample_join_days(N_TOTAL)
motive_pop = np.where(pre_season[join_pop], rng.random(N_TOTAL) < 0.6, rng.random(N_TOTAL) < 0.25)
# status: 0 join-only, 1 test-only, 2 profile-only, 3 test+profile (modeling population)
p_org = np.array([0.72, 0.06, 0.08, 0.14])
p_sea = np.array([0.90, 0.05, 0.035, 0.015])
u = rng.random(N_TOTAL)
cum_org, cum_sea = np.cumsum(p_org), np.cumsum(p_sea)
status_pop = np.where(motive_pop, np.searchsorted(cum_sea, u), np.searchsorted(cum_org, u))
status_pop = np.clip(status_pop, 0, 3)
# force the exact modeling-population size
idx3 = np.flatnonzero(status_pop == 3)
if len(idx3) > N_MODEL:
    status_pop[rng.choice(idx3, len(idx3) - N_MODEL, replace=False)] = 2
elif len(idx3) < N_MODEL:
    cand = np.flatnonzero(status_pop == 2)
    status_pop[rng.choice(cand, N_MODEL - len(idx3), replace=False)] = 3
# last login for the non-modeling population: mostly the join day, some return later
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

# ================================================================ (b) modeling population
print("generating modeling population ...")
n = N_MODEL
sel = np.flatnonzero(status_pop == 3)
join = join_pop[sel]
season_joiner = motive_pop[sel]
commit = rng.normal(0, 1, n)                                    # latent diligence

# --- static attributes
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

# --- preference fields (the true driver)
pref_salary_default = rng.random(n) < sig(0.0 - 0.9 * commit + 0.6 * season_joiner)   # 1 = left default
pref_welfare_cnt = np.clip(rng.poisson(np.exp(1.2 + 0.35 * commit - 0.4 * season_joiner)), 0, 10)
pref_complete = (~pref_salary_default) & (pref_welfare_cnt >= 5)
marketing_consent = rng.random(n) < sig(0.2 + 0.7 * commit)

# --- timeline: join → first test → profile complete
test1 = join + 1 + rng.geometric(1 / 5, n)
profile = test1 + rng.geometric(1 / 4, n)
profile = np.minimum(profile, N_DAYS - 2)
test1 = np.minimum(test1, profile)

# --- consent (target): P(consent | profile & test done)
z_score = (acca_t_score - 60) / 12
logit = (1.30 * pref_complete + 0.35 * z_score - 0.80 * season_joiner
         + 0.55 * marketing_consent + 0.30 * commit)
b0 = 0.0
for _ in range(60):  # solve intercept for an 81% base rate
    b0 -= (sig(b0 + logit).mean() - 0.81) * 4
p_true = sig(b0 + logit)
consent = rng.random(n) < p_true
delay = np.rint(np.exp(rng.normal(1.1, 1.0, n))).astype(int)          # median ~3 days, long tail
consent_day = np.where(consent, np.minimum(profile + delay, N_DAYS - 1), -1)
print(f"  consent rate {consent.mean():.4f} (intercept {b0:.3f})")

# --- login matrix (users × days)
print("simulating logins ...")
day_grid = np.arange(N_DAYS)[None, :]
base = np.exp(-2.4 + 0.45 * commit)[:, None]                       # ~0.09 logins/day
lam = np.broadcast_to(base, (n, N_DAYS)).copy()
lam *= np.where(season_day[None, :], 3.0, 1.0)                     # everyone is busier in season
# season joiners leave after the first season that follows their join
leave_day = season_end_after[join]
lam *= np.where(season_joiner[:, None] & (day_grid > leave_day[:, None]), 0.12, 1.0)
# consent → login: report check burst + sustained lift (application management, notices)
after = (consent_day[:, None] >= 0) & (day_grid > consent_day[:, None])
lam *= np.where(after, 2.2, 1.0)
lam = np.where(day_grid < join[:, None], 0.0, lam)                  # no logins before join
logins = (rng.random((n, N_DAYS)) < (1 - np.exp(-lam))).astype(np.uint8)
rows = np.arange(n)
logins[rows, join] = 1                                             # join day counts as a login
logins[rows, profile] = 1
pos = np.flatnonzero(consent)
for d in (1, 2):                                                    # report-check burst
    dd = np.minimum(consent_day[pos] + d, N_DAYS - 1)
    logins[pos, dd] = np.where(rng.random(len(pos)) < 0.7, 1, logins[pos, dd])

# --- competency test attempts (dated)
n_pre = 1 + rng.poisson(0.35, n)                                   # attempts before profile
n_post = np.where(consent, rng.poisson(0.5, n), rng.poisson(0.15, n))
test_rows = []
for i in range(n):
    d_pre = np.concatenate([[test1[i]], rng.integers(test1[i], profile[i] + 1, n_pre[i] - 1)])
    start = consent_day[i] + 1 if consent[i] else profile[i] + 1
    d_post = rng.integers(start, N_DAYS, n_post[i]) if (n_post[i] > 0 and start < N_DAYS) else np.array([], int)
    for d in np.concatenate([d_pre, d_post]):
        test_rows.append((i, int(d)))
events_test = pd.DataFrame(test_rows, columns=["user", "day"])

# --- apply events (dated; funnel flags)
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

# --- notification responses (dated) — mostly a consequence of being in the pool
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
    # latent truth — used only for scoring the reproduction, never as a feature
    "truth_p_consent": p_true, "truth_commit": commit, "truth_season_joiner": season_joiner.astype(int),
    "truth_pref_complete": pref_complete.astype(int),
})

population.to_csv(DATA / "population.csv", index=False)
users.to_csv(DATA / "users.csv", index=False)
np.savez_compressed(DATA / "logins.npz", logins=logins)
events_test.to_csv(DATA / "events_test.csv", index=False)
events_apply.to_csv(DATA / "events_apply.csv", index=False)
events_notice.to_csv(DATA / "events_notice.csv", index=False)
print(f"saved: population {len(population):,}, users {len(users):,}, logins {logins.shape}, "
      f"tests {len(events_test):,}, applies {len(events_apply):,}, notices {len(events_notice):,}")
