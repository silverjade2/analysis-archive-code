"""유저 발화 1건 + 기기 응답 1건 짝짓기 -> turn_pairs"""

import pandas as pd
from common import DATA, DEVICE, GZ, PAIR_MAX_SEC, USER

logs = pd.read_csv(
    DATA / "logs.csv.gz", usecols=["row_id", "session_id", "log_type", "ts", "text"], keep_default_na=False
)
logs["ts"] = pd.to_datetime(logs["ts"])
print("log_type share", logs.log_type.value_counts(normalize=True).sort_index().round(4).to_dict())

# 3, 4(음성 참조)와 2(제어 코드)를 빼면 shift(-1)이 곧 다음 응답. 응답 전에 유저가 또 말했으면 앞 발화는 무응답
x = logs[logs.log_type.isin([USER, DEVICE])].sort_values(["session_id", "ts", "row_id"])
nxt = x.groupby("session_id")[["row_id", "log_type", "ts", "text"]].shift(-1)
gap = (nxt["ts"] - x["ts"]).dt.total_seconds()
paired = (x["log_type"] == USER) & (nxt["log_type"] == DEVICE) & (gap <= PAIR_MAX_SEC)

u = x["log_type"] == USER
pairs = pd.DataFrame(
    {
        "turn_id": x.loc[u, "row_id"],
        "session_id": x.loc[u, "session_id"],
        "ts": x.loc[u, "ts"],
        "text": x.loc[u, "text"],
        "resp_row_id": nxt.loc[u, "row_id"].where(paired[u]).astype("Int64"),
        "resp_text": nxt.loc[u, "text"].where(paired[u], ""),
        "latency_sec": gap[u].where(paired[u]).round(1),
    }
)
pairs.to_csv(DATA / "turn_pairs.csv.gz", index=False, compression=GZ)

late = ((x["log_type"] == USER) & (nxt["log_type"] == DEVICE) & (gap > PAIR_MAX_SEC)).sum()
print(f"user turns {len(pairs)}, paired {pairs.resp_row_id.notna().mean():.3f}, cut by {PAIR_MAX_SEC}s {late}")
print("latency median", pairs.latency_sec.median())
