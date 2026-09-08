"""04 — v2 features: cut at the moment before the target is decided.

Positives: cutoff = consent day - 1.
Negatives: cutoff = profile day - 1 + a delay sampled from the positives' (consent - profile)
           distribution, capped at the snapshot.
The cutoff is never earlier than profile day - 1, so a user who consented on the profile day
(delay 0) is still cut strictly before the consent day. This gives negatives a comparable observation
           window instead of the full period up to the snapshot.
Features are then computed from events on or before each user's cutoff only.
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
# same formula for both classes: profile + delay - 1 (positives' delay is their own)
own_delay = np.where(pos, users["consent_day"].values - users["profile_day"].values,
                     rng.choice(delay, size=len(users)))
cutoff = users["profile_day"].values + own_delay - 1
cutoff = np.clip(cutoff, users["profile_day"].values - 1, N_DAYS - 1)   # strictly before consent, even when delay = 0
f = build_features(cutoff, users, *rest)
f.to_csv(DATA / "features_v2_timecut.csv", index=False)
print(f"v2 rows {len(f):,}  median observation window (cutoff - join) pos %d  neg %d days"
      % (np.median(cutoff[pos] - users['join_day'].values[pos]), np.median(cutoff[~pos] - users['join_day'].values[~pos])))
p = f["matching_use_yn"] == 1
print("mean login_counts  pos %.1f  neg %.1f" % (f.loc[p, "login_counts"].mean(), f.loc[~p, "login_counts"].mean()))
print("median days_since_last_login  pos %d  neg %d" % (f.loc[p, "days_since_last_login"].median(), f.loc[~p, "days_since_last_login"].median()))
