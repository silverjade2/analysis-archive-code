"""03. v1 feature: 조회 시점 snapshot (원래 노트북의 구조).

모든 유저의 feature를 snapshot 당일까지의 전체 이벤트로 계산한다. 타깃(동의)이 결정된
이후에 일어난 이벤트도 포함된다.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, N_DAYS
from features import build_features, load

users, *rest = load()
cutoff = np.full(len(users), N_DAYS - 1)
f = build_features(cutoff, users, *rest)
f.to_csv(DATA / "features_v1_snapshot.csv", index=False)
pos = f["matching_use_yn"] == 1
print(f"v1 rows {len(f):,}  positive {pos.mean():.3f}")
print("mean login_counts  pos %.1f  neg %.1f" % (f.loc[pos, "login_counts"].mean(), f.loc[~pos, "login_counts"].mean()))
print("median days_since_last_login  pos %d  neg %d" % (f.loc[pos, "days_since_last_login"].median(), f.loc[~pos, "days_since_last_login"].median()))
