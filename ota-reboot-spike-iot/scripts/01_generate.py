"""가상 세션 로그 생성

- 올타임 기기 25%, 스케줄 기기 5%, 설치 배치 2회, OTA 4회
- truth_* 는 채점 전용
"""

import numpy as np
import pandas as pd
from common import DATA, END, SEED, START

rng = np.random.default_rng(SEED)

DAYS = pd.date_range(START, END, freq="D")
N_BASE = 1700
BATCHES = [("2025-07-08", 150), ("2025-08-19", 150)]
OTA = [("2025-06-24", 14), ("2025-07-17", 14), ("2025-08-06", 14), ("2025-08-28", 14)]
P_ALLTIME = 0.25
P_SCHEDULE = 0.05
P_LMK = 0.18
SCHEDULE_HOUR = 9

OUT = DATA
OUT.mkdir(exist_ok=True)


def human_hour(n):
    comp = rng.choice(3, size=n, p=[0.35, 0.45, 0.20])
    h = np.where(
        comp == 0, rng.normal(8.0, 1.2, n), np.where(comp == 1, rng.normal(19.5, 1.8, n), rng.uniform(10, 17, n))
    )
    return np.clip(h, 0.0, 23.95)


def minutes_lognormal(n, median_min, sigma):
    return np.exp(rng.normal(np.log(median_min), sigma, n))


serials = [f"D{i:05d}" for i in range(1, N_BASE + 1)]
install = list(START - pd.to_timedelta(rng.integers(20, 110, N_BASE), unit="D"))
for d, n in BATCHES:
    k = len(serials)
    serials += [f"D{i:05d}" for i in range(k + 1, k + n + 1)]
    install += [pd.Timestamp(d)] * n
N = len(serials)

role = rng.random(N)
truth_alltime = role < P_ALLTIME
truth_schedule = (role >= P_ALLTIME) & (role < P_ALLTIME + P_SCHEDULE)
devices = pd.DataFrame(
    {
        "serial": serials,
        "install_date": pd.to_datetime(install).normalize(),
        "p_active": np.round(rng.beta(2, 5, N), 3),
        "p_mobile": np.round(rng.beta(2, 3, N), 3),
        "truth_alltime": truth_alltime.astype(int),
        "truth_schedule": truth_schedule.astype(int),
    }
)

rows = []  # serial, kind, start, end, truth_source


def add(serial, kind, start, minutes, src):
    rows.append((serial, kind, start, start + pd.Timedelta(minutes=float(minutes)), src))


for r in devices.itertuples(index=False):
    days = DAYS[DAYS >= r.install_date]
    if len(days) == 0:
        continue
    age = (days - r.install_date).days
    wk = days.dayofweek >= 5
    boost = np.where(age < 3, 2.0, 1.0) * np.where(wk, 1.15, 1.0)

    if r.truth_alltime:
        # 켜짐 20~120h, 꺼짐 평균 12h
        # 워밍업 10일. 관측 첫날에 전부 켜지는 걸 막음
        t = max(r.install_date, START - pd.Timedelta(days=10)) + pd.Timedelta(hours=float(human_hour(1)[0]))
        while t < END + pd.Timedelta(days=1):
            on_h = rng.uniform(20, 120)
            add(r.serial, "fixed", t, on_h * 60, "user")
            t = t + pd.Timedelta(hours=on_h + rng.exponential(12))
        use = rng.random(len(days)) < 0.30
        for d, h in zip(days[use], human_hour(use.sum())):
            add(r.serial, "mobile", d + pd.Timedelta(hours=float(h)), rng.uniform(20, 60), "user")
        continue

    p = np.clip(r.p_active * boost, 0, 0.95)
    active = rng.random(len(days)) < p
    for d in days[active]:
        n_fixed = 1 + int(rng.random() < 0.3)
        for h, m in zip(human_hour(n_fixed), minutes_lognormal(n_fixed, 90, 0.6)):
            add(r.serial, "fixed", d + pd.Timedelta(hours=float(h)), m, "user")
        if rng.random() < r.p_mobile:
            add(r.serial, "mobile", d + pd.Timedelta(hours=float(human_hour(1)[0])), rng.uniform(15, 60), "user")

    if r.truth_schedule:
        for d in days:
            add(
                r.serial,
                "fixed",
                d + pd.Timedelta(hours=SCHEDULE_HOUR, minutes=float(rng.integers(0, 3))),
                120,
                "schedule",
            )

sess = pd.DataFrame(rows, columns=["serial", "kind", "start", "end", "truth_source"])

ota_ts = [pd.Timestamp(d) + pd.Timedelta(hours=h) for d, h in OTA]
new_rows = []
for T in ota_ts:
    ongoing = (sess.kind == "fixed") & (sess.start < T) & (sess.end > T)
    for i in sess.index[ongoing]:
        old_end = sess.at[i, "end"]
        sess.at[i, "end"] = T
        s2 = T + pd.Timedelta(minutes=float(rng.uniform(1.5, 3.0)))
        if old_end > s2:
            new_rows.append((sess.at[i, "serial"], "fixed", s2, old_end, "ota_resume"))
    busy = set(sess.loc[ongoing, "serial"])
    eligible = devices[(devices.install_date < T.normalize()) & (~devices.serial.isin(busy))]
    hit = eligible[rng.random(len(eligible)) < P_LMK]
    for s in hit.serial:
        start = T + pd.Timedelta(minutes=float(rng.uniform(1.0, 6.0)))
        if rng.random() < 0.8:
            m = minutes_lognormal(1, 40, 0.4)[0]  # 자동 종료
        else:
            m = minutes_lognormal(1, 240, 0.5)[0]  # 사용자가 나중에 끔
        new_rows.append((s, "fixed", start, start + pd.Timedelta(minutes=float(m)), "ota_lmk"))

sess = pd.concat([sess, pd.DataFrame(new_rows, columns=sess.columns)], ignore_index=True)
sess = sess[(sess.end > START) & (sess.start < END + pd.Timedelta(days=1))]  # 관측 전 시작분은 자정 분할 집계용
sess["end"] = sess["end"].clip(upper=END + pd.Timedelta(days=1))
sess["start"] = sess["start"].dt.floor("s")
sess["end"] = sess["end"].dt.floor("s")
sess = sess.sort_values(["serial", "start"]).reset_index(drop=True)
sess["session_id"] = np.arange(1, len(sess) + 1)
sess["duration_min"] = ((sess.end - sess.start).dt.total_seconds() / 60).round(1)

devices.to_csv(OUT / "devices.csv", index=False)
sess[["session_id", "serial", "kind", "start", "end", "duration_min", "truth_source"]].to_csv(
    OUT / "sessions.csv", index=False
)
pd.DataFrame({"ota_dt": [pd.Timestamp(d).date() for d, _ in OTA], "ota_hour": [h for _, h in OTA]}).to_csv(
    OUT / "ota_days.csv", index=False
)
pd.DataFrame({"batch_dt": [pd.Timestamp(d).date() for d, _ in BATCHES], "n_devices": [n for _, n in BATCHES]}).to_csv(
    OUT / "install_batches.csv", index=False
)

print(f"devices {N}  sessions {len(sess)}  alltime {truth_alltime.sum()}  schedule {truth_schedule.sum()}")
print(sess.truth_source.value_counts().to_string())
