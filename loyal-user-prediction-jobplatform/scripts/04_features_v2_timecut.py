"""v2 feature. 타깃이 결정되기 직전에서 자른다.

양성은 동의일 전날이 cutoff다. 음성은 프로필 완성일 전날에 양성의 지연 분포에서 뽑은 값을 더한 날이고, 상한은
snapshot 당일이다. 음성에게 snapshot까지의 전체 기간 대신 양성과 비슷한 관측 창을 주기 위한 설계다. cutoff는
프로필 완성일 전날보다 앞서지 않으므로, 프로필 완성 당일에 동의한 유저(지연 0일)도 동의일보다 엄격히 앞에서
잘린다. feature는 각 유저의 cutoff 당일까지의 이벤트만으로 계산한다.
"""

import numpy as np
from common import DATA, N_DAYS, SEED
from features import build_features, load

rng = np.random.default_rng(SEED)
users, *rest = load()
pos = users["matching_use_yn"].values == 1
delay = (users.loc[pos, "consent_day"] - users.loc[pos, "profile_day"]).values
# 양성은 자기 지연, 음성은 양성의 지연 분포에서 뽑은 값. 두 클래스 모두 profile + delay - 1
own_delay = np.where(pos, users["consent_day"].values - users["profile_day"].values, rng.choice(delay, size=len(users)))
cutoff = users["profile_day"].values + own_delay - 1
cutoff = np.clip(cutoff, users["profile_day"].values - 1, N_DAYS - 1)  # 동의 당일은 제외
f = build_features(cutoff, users, *rest)  # cutoff 이후 이벤트는 집계에서 빠진다
f.to_csv(DATA / "features_v2_timecut.csv", index=False)
print(
    f"v2 rows {len(f):,}  median observation window (cutoff - join) pos %d  neg %d days"
    % (np.median(cutoff[pos] - users["join_day"].values[pos]), np.median(cutoff[~pos] - users["join_day"].values[~pos]))
)
p = f["matching_use_yn"] == 1
print("mean login_counts  pos %.1f  neg %.1f" % (f.loc[p, "login_counts"].mean(), f.loc[~p, "login_counts"].mean()))
print(
    "median days_since_last_login  pos %d  neg %d"
    % (f.loc[p, "days_since_last_login"].median(), f.loc[~p, "days_since_last_login"].median())
)
