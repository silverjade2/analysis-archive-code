"""기기 마스터와 이벤트 로그 생성

- 설치가 출시 초기에 몰린 fleet (10개월간 기기 수 2배 이상)
- 주행 기록 4종이 로그 대부분. 기본 스케줄 시각 9/12/15/18시에 몰림
- 사고(전도, 낙하) hazard는 설치 첫 30일 2배, 달력 시간에는 불변
- 사고 뒤 7일은 재발 hazard가 높음 (같은 원인이 안 풀림)
- 사용자가 들어 옮기면 SPLIT 전에는 TIP, 뒤에는 LIFT로 찍힘
- 출고 전 검사 로그가 설치일 전에 일부 남음
"""

import numpy as np
import pandas as pd
from common import DATA, END, LEVEL_SHIFT_1, LEVEL_SHIFT_2, SEED, SPLIT, START

N_DEV = 1200
INSTALL_W = [0.32, 0.20, 0.13, 0.08, 0.07, 0.05, 0.05, 0.04, 0.03, 0.03]  # 월별 설치 비중. 출시 초기에 몰림
DRIVE_RATE = 1.5  # 기기-일당 주행 기록 기대값 (intensity 1, env 1 기준)
DRIVE_P = [0.21, 0.46, 0.13, 0.20]  # N1 N2 N3 N4
HOUR_W = np.array([1, 1, 1, 1, 1, 1, 2, 4, 6, 14, 8, 7, 14, 8, 7, 14, 8, 8, 14, 9, 6, 4, 2, 1], float)
HOUR_W /= HOUR_W.sum()

BASE_H = 0.0021  # 설치 30일 이후 하루 사고 hazard (intensity 1)
EARLY_MULT = 2.2  # 설치 첫 30일
RECUR_MULT, RECUR_TAU = 35.0, 5.0  # 사고 뒤 d일: 1 + RECUR_MULT * exp(-d / RECUR_TAU)
DROP_SHARE = 0.16  # 사고 중 낙하 비중
LIFT_RATE = 0.0014  # 하루 들림 확률 (handling 1). 첫 30일 2배
BURST_SHARE = 0.04  # WIFI 몰림 기기 비중

HW = ["H%02d" % i for i in range(1, 17)]


def level_of(code: str, ts: pd.Series) -> np.ndarray:
    """관제 대응 등급. 전도, 낙하만 운영 중 두 번 바뀜"""
    if code in ("TIP", "DROP"):
        return np.where(ts < LEVEL_SHIFT_1, "4", np.where(ts < LEVEL_SHIFT_2, "3", "1"))
    fixed = {"B2": "1", "T1": "2", "T2": "2", "T3": "2", "T4": "3", "LIFT": "4", "W2": "3"}
    if code in fixed:
        return np.full(len(ts), fixed[code])
    if code.startswith(("H", "M", "W")):
        return np.full(len(ts), "3")
    return np.full(len(ts), "4")


def rand_hours(rng, n):
    h = rng.choice(24, size=n, p=HOUR_W)
    return pd.to_timedelta(h * 60 + rng.uniform(0, 60, n), unit="m")


def make_devices(rng):
    months = pd.date_range(START.to_period("M").to_timestamp(), periods=10, freq="MS")
    m = rng.choice(10, size=N_DEV, p=INSTALL_W)
    day = rng.integers(0, 28, N_DEV)
    install = months[m] + pd.to_timedelta(day, unit="D")
    install = install.where(install >= START, START + pd.to_timedelta(rng.integers(0, 5, N_DEV), unit="D"))
    dev = pd.DataFrame(
        {
            "serial": [f"R{i:05d}" for i in range(N_DEV)],
            "install_date": pd.DatetimeIndex(install).normalize(),
            "truth_intensity": rng.lognormal(0, 0.5, N_DEV).round(3),  # 주행량
            "truth_env": rng.lognormal(0, 0.45, N_DEV).round(3),  # 설치 환경 난이도. 주행 기록을 늘림
            "truth_handling": rng.lognormal(0, 0.5, N_DEV).round(3),  # 들어 옮기는 빈도
            "truth_burst": rng.random(N_DEV) < BURST_SHARE,
        }
    )
    return dev.sort_values("install_date").reset_index(drop=True)


def device_day_grid(dev):
    rows = []
    for d in dev.itertuples(index=False):
        days = pd.date_range(d.install_date, END - pd.Timedelta(days=1), freq="D")
        rows.append(pd.DataFrame({"serial": d.serial, "day": days, "age": np.arange(len(days))}))
    return pd.concat(rows, ignore_index=True)


def gen_drive(rng, grid, dev):
    g = grid.merge(dev[["serial", "truth_intensity", "truth_env"]], on="serial")
    lam = DRIVE_RATE * g["truth_intensity"] * g["truth_env"]
    n = rng.poisson(lam)
    rep = g.loc[g.index.repeat(n)]
    code = rng.choice(["N1", "N2", "N3", "N4"], size=len(rep), p=DRIVE_P)
    ts = rep["day"].to_numpy() + rand_hours(rng, len(rep)).to_numpy()
    return pd.DataFrame({"serial": rep["serial"].to_numpy(), "ts": ts, "code": code, "truth_kind": "drive"})


def gen_misc(rng, grid, dev):
    """고장, 기능 정지, 통신, 상태 기록. 코드마다 coverage와 기기당 빈도가 다름"""
    g = grid.merge(dev[["serial", "truth_intensity", "truth_burst"]], on="serial")
    out = []
    spec = {}  # code: (device coverage prob, daily rate)
    for i, c in enumerate(HW):
        # H14 UV 모듈: 실제 로그에서 부품 고장 중 행이 가장 많던 코드 (소수 기기 반복)
        spec[c] = (rng.uniform(0.03, 0.25), rng.uniform(0.002, 0.03) * (3 if i == 13 else 1))
    spec.update({"M1": (0.6, 0.004), "M2": (0.4, 0.003), "M3": (0.7, 0.006), "M4": (0.3, 0.003)})
    spec.update({"M5": (0.5, 0.004), "M6": (0.5, 0.003), "W2": (0.3, 0.004), "W3": (0.2, 0.003)})
    spec.update({"B1": (0.9, 0.03), "B2": (0.8, 0.02), "U1": (0.6, 0.006), "F1": (0.25, 0.003)})
    spec.update({"T1": (0.004, 0.01), "T2": (0.003, 0.01), "T3": (0.004, 0.01), "T4": (0.2, 0.006)})
    for code, (cov, rate) in spec.items():
        has = dev.loc[rng.random(len(dev)) < cov, "serial"]
        sub = g[g["serial"].isin(has)]
        r = rate * (sub["truth_intensity"] if code.startswith("M") else pd.Series(1.0, index=sub.index))
        if code == "T4":  # 배포 뒤부터 찍히는 코드
            r = r * (sub["day"] >= SPLIT)
        n = rng.poisson(r)
        rep = sub.loc[sub.index.repeat(n)]
        ts = rep["day"].to_numpy() + rand_hours(rng, len(rep)).to_numpy()
        out.append(pd.DataFrame({"serial": rep["serial"].to_numpy(), "ts": ts, "code": code, "truth_kind": "misc"}))
    # WIFI: 몰림 기기 소수가 대부분을 만든다
    rate = np.where(g["truth_burst"], 2.5, 0.004)
    n = rng.poisson(rate)
    rep = g.loc[g.index.repeat(n)]
    ts = rep["day"].to_numpy() + rand_hours(rng, len(rep)).to_numpy()
    out.append(pd.DataFrame({"serial": rep["serial"].to_numpy(), "ts": ts, "code": "W1", "truth_kind": "misc"}))
    return pd.concat(out, ignore_index=True)


def gen_accidents(rng, dev):
    """기기별 하루 단위 hazard 추첨. 사고 뒤 며칠은 재발 hazard가 높아 같은 기기에 몰림"""
    rows, eid = [], 0
    for d in dev.itertuples(index=False):
        days = pd.date_range(d.install_date, END - pd.Timedelta(days=1), freq="D")
        age = np.arange(len(days))
        base = BASE_H * np.sqrt(d.truth_intensity) * np.where(age < 30, EARLY_MULT, 1.0)
        last = None
        u = rng.random(len(days))
        for i, day in enumerate(days):
            mult = 1.0 if last is None else 1 + RECUR_MULT * np.exp(-(i - last) / RECUR_TAU)
            if u[i] < base[i] * mult:
                last = i
                eid += 1
                kind = "accident_drop" if rng.random() < DROP_SHARE else "accident_tip"
                n = 1 + rng.poisson(0.7)
                t0 = day + rand_hours(rng, 1)[0]
                gaps = np.concatenate([[0.0], rng.exponential(12, n - 1)])
                for g in np.cumsum(gaps):
                    rows.append((d.serial, t0 + pd.Timedelta(minutes=float(g)), kind, eid))
    acc = pd.DataFrame(rows, columns=["serial", "ts", "truth_kind", "truth_event"])
    # 배포 전에는 낙하도 전도 코드로 찍혔다
    acc["code"] = np.where((acc["truth_kind"] == "accident_drop") & (acc["ts"] >= SPLIT), "DROP", "TIP")
    return acc


def gen_lifts(rng, grid, dev):
    """사용자가 들어 옮김. 설치 초기에 잦고, 배포 전에는 전도 코드로 섞여 들어감"""
    g = grid.merge(dev[["serial", "truth_handling"]], on="serial")
    p = LIFT_RATE * g["truth_handling"] * np.where(g["age"] < 30, 2.0, 1.0)
    hit = g[rng.random(len(g)) < p]
    n = 1 + rng.poisson(0.3, len(hit))
    rep = hit.loc[hit.index.repeat(n)]
    ts = rep["day"].to_numpy() + rand_hours(rng, len(rep)).to_numpy()
    ts = ts + pd.to_timedelta(rng.exponential(3, len(rep)), unit="m").to_numpy()
    lifts = pd.DataFrame({"serial": rep["serial"].to_numpy(), "ts": ts, "truth_kind": "lift"})
    lifts["code"] = np.where(lifts["ts"] >= SPLIT, "LIFT", "TIP")
    return lifts


def gen_qa(rng, dev):
    """출고 전 검사. 설치 2~10일 전, 일부 기기에 전도가 찍힘"""
    pick = dev[rng.random(len(dev)) < 0.12]
    n = 1 + rng.poisson(0.5, len(pick))
    rep = pick.loc[pick.index.repeat(n)]
    back = rng.integers(2, 11, len(rep))
    ts = (
        rep["install_date"].to_numpy()
        - pd.to_timedelta(back, unit="D").to_numpy()
        + rand_hours(rng, len(rep)).to_numpy()
    )
    return pd.DataFrame({"serial": rep["serial"].to_numpy(), "ts": ts, "code": "TIP", "truth_kind": "qa"})


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    dev = make_devices(rng)
    grid = device_day_grid(dev)
    parts = [
        gen_drive(rng, grid, dev),
        gen_misc(rng, grid, dev),
        gen_accidents(rng, dev),
        gen_lifts(rng, grid, dev),
        gen_qa(rng, dev),
    ]
    ev = pd.concat(parts, ignore_index=True)
    ev["ts"] = pd.to_datetime(ev["ts"]).dt.floor("s")
    ev = ev[(ev["ts"] >= START) & (ev["ts"] < END)]
    ev["truth_event"] = ev["truth_event"].fillna(0).astype(int)
    ev["level"] = ""
    for code in ev["code"].unique():
        m = ev["code"] == code
        ev.loc[m, "level"] = level_of(code, ev.loc[m, "ts"])
    ev = ev.sort_values(["serial", "ts", "code"], kind="stable").reset_index(drop=True)
    ev.insert(0, "id", np.arange(1, len(ev) + 1))
    dev.to_csv(DATA / "devices.csv", index=False)
    ev[["id", "serial", "ts", "code", "level", "truth_kind", "truth_event"]].to_csv(
        DATA / "events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )  # mtime 고정: 다시 돌려도 byte 동일
    print(f"devices {len(dev):,}  events {len(ev):,}  {ev['ts'].min():%Y-%m-%d} ~ {ev['ts'].max():%Y-%m-%d}")
    print(ev["truth_kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
