"""재시도(같은 세션, RETRY_WINDOW_SEC 안, bigram Jaccard >= RETRY_SIM)와 세션 구조. 정의 sweep 포함

세션을 넘어간 재시도는 기기 식별자가 없어 못 셈
"""

import pandas as pd
from common import DATA, GZ, RES, RETRY_SIM, RETRY_WINDOW_SEC, bigram_sim

pairs = pd.read_csv(DATA / "turn_pairs.csv.gz", keep_default_na=False, parse_dates=["ts"])
flags = pd.read_csv(DATA / "turn_flags.csv.gz", keep_default_na=False)
d = pairs.merge(flags, on="turn_id").sort_values(["session_id", "ts", "turn_id"])

g = d.groupby("session_id")
d["next_text"] = g["text"].shift(-1)
d["next_gap"] = (g["ts"].shift(-1) - d["ts"]).dt.total_seconds()
has_next = d["next_text"].notna() & (d["text"] != "") & (d["next_text"] != "")  # 빈 발화는 비교할 원문이 없음
d["sim"] = 0.0
d.loc[has_next, "sim"] = [bigram_sim(a, b) for a, b in zip(d.loc[has_next, "text"], d.loc[has_next, "next_text"])]


def retry_flag(window: float, sim: float) -> pd.Series:
    return has_next & (d["next_gap"] <= window) & (d["sim"] >= sim)


d["retry"] = retry_flag(RETRY_WINDOW_SEC, RETRY_SIM)
base = d[d["text"] != ""]
retry = base.groupby("state")["retry"].agg(["mean", "sum", "size"]).reset_index()
retry.columns = ["prev_state", "retry_rate", "n_retry", "n_turns"]
retry.to_csv(RES / "retry_by_prev_state.csv", index=False)
d[["turn_id", "retry"]].to_csv(DATA / "turn_retry.csv.gz", index=False, compression=GZ)

sweep = []
for window in [10, 30, 60, 120]:
    for sim in [0.3, 0.5, 0.8]:
        r = retry_flag(window, sim)[d["text"] != ""].groupby(base["state"]).mean()
        sweep.append({"window_sec": window, "sim": sim, **{f"retry_{k}": v for k, v in r.items()}})
sweep = pd.DataFrame(sweep)
sweep["fallback_lowest"] = sweep["retry_fallback"] < sweep[["retry_normal", "retry_none"]].min(axis=1)
sweep.to_csv(RES / "session_sweep.csv", index=False)

s = d.groupby("session_id").agg(n_user=("turn_id", "size"), t0=("ts", "min"), t1=("ts", "max"), retry=("retry", "any"))
session_stats = pd.DataFrame(
    {
        "metric": ["user_turns_per_session", "single_turn_share", "session_len_median_sec", "sessions_with_retry"],
        "value": [
            s["n_user"].mean(),
            (s["n_user"] == 1).mean(),
            (s["t1"] - s["t0"]).dt.total_seconds().median(),
            s["retry"].mean(),
        ],
    }
)
session_stats.to_csv(RES / "session_stats.csv", index=False)
print(retry.round(4).to_string(index=False))
print(session_stats.round(4).to_string(index=False))
print(sweep.round(4).to_string(index=False))
