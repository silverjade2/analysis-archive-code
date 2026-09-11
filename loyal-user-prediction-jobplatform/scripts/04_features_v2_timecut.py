"""v2 feature: 타깃 결정 직전에서 절단. 양성은 동의일 - 1, 음성은 양성의 지연 분포로 비슷한 관측 창.
출력: data/features_v2_timecut.csv
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, N_DAYS, SEED
from features import build_features, load

rng = np.random.default_rng(SEED)
users, *rest = load()
pos = users["matching_use_yn"].values == 1
delay = (users.loc[pos, "consent_day"] - users.loc[pos, "profile_day"]).values
# 양성은 자기 지연, 음성은 양성의 지연 분포에서 뽑은 값. 두 클래스 모두 profile + delay - 1
own_delay = np.where(pos, users["consent_day"].values - users["profile_day"].values,
                     rng.choice(delay, size=len(users)))
cutoff = users["profile_day"].values + own_delay - 1
cutoff = np.clip(cutoff, users["profile_day"].values - 1, N_DAYS - 1)   # 동의 당일은 제외
f = build_features(cutoff, users, *rest)   # cutoff 이후 이벤트는 집계에서 빠진다
f.to_csv(DATA / "features_v2_timecut.csv", index=False)
print(f"v2 rows {len(f):,}  median observation window (cutoff - join) pos %d  neg %d days"
      % (np.median(cutoff[pos] - users['join_day'].values[pos]), np.median(cutoff[~pos] - users['join_day'].values[~pos])))
p = f["matching_use_yn"] == 1
print("mean login_counts  pos %.1f  neg %.1f" % (f.loc[p, "login_counts"].mean(), f.loc[~p, "login_counts"].mean()))
print("median days_since_last_login  pos %d  neg %d" % (f.loc[p, "days_since_last_login"].median(), f.loc[~p, "days_since_last_login"].median()))
