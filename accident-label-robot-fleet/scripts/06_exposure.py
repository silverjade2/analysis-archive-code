"""덜 써서 줄었나: 알림 사건 비율을 달력일, 가동일, 가동 시간으로 나눠 비교

- exposure_monthly.csv: 달별 알림 사건 수, 기기-일, 가동 기기-일, 가동 시간, 세 가지 비율
- exposure_change.csv: 첫 비교 달에서 마지막 달까지 세 비율의 변화율
"""

import numpy as np
import pandas as pd
from common import END, RES, load_devices, load_sessions

COMPARE = ("2025-02", "2025-10")  # 출시 첫 온전한 달, 마지막 달


def main():
    dev = load_devices()
    ses = load_sessions()
    inc = pd.read_csv(RES / "incidents.csv", parse_dates=["ts"])

    months = pd.period_range("2025-01", (END - pd.Timedelta(days=1)).strftime("%Y-%m"), freq="M")
    rows = []
    for m in months:
        m0, m1 = m.start_time, (m + 1).start_time
        lo = dev["install_date"].clip(lower=m0)
        days = float((min(m1, END) - lo).dt.days.clip(lower=0).sum())
        s = ses[(ses["start"] < m1) & (ses["end"] > m0)]
        hrs = ((s["end"].clip(upper=m1) - s["start"].clip(lower=m0)).dt.total_seconds() / 3600).sum()
        active = s.assign(d=s["start"].clip(lower=m0).dt.normalize())[["serial", "d"]].drop_duplicates()
        n = int(((inc["ts"] >= m0) & (inc["ts"] < m1)).sum())
        rows.append((str(m), n, days, len(active), round(hrs, 1)))
    t = pd.DataFrame(rows, columns=["month", "incidents", "device_days", "active_days", "run_hours"])
    t["per_1k_days"] = (t["incidents"] / t["device_days"] * 1000).round(3)
    t["per_1k_active"] = (t["incidents"] / t["active_days"] * 1000).round(3)
    t["per_10k_hours"] = (t["incidents"] / t["run_hours"] * 10000).round(3)
    t["hours_per_day"] = (t["run_hours"] / t["device_days"]).round(3)
    t.to_csv(RES / "exposure_monthly.csv", index=False)

    a, b = (t.set_index("month").loc[m] for m in COMPARE)
    cols = ["per_1k_days", "per_1k_active", "per_10k_hours", "hours_per_day"]
    ch = pd.DataFrame({"measure": cols, "from": [a[c] for c in cols], "to": [b[c] for c in cols]})
    ch["change"] = np.round(ch["to"] / ch["from"] - 1, 3)
    ch.to_csv(RES / "exposure_change.csv", index=False)
    print(t.to_string(index=False))
    print(ch.to_string(index=False))


if __name__ == "__main__":
    main()
