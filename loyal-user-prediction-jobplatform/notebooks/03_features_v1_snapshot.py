"""03 — v1 features: snapshot at the query date (the original notebook's structure).

Every user's features are computed with all events up to the snapshot day, including events
that happened AFTER the target (consent) was decided.
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
