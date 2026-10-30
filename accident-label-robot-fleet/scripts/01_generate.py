"""기기 마스터, 가동 세션, 이벤트 로그, 상담 기록 생성

- 가동은 하루 한두 번 켜져 몇 시간 도는 세션. 설치 뒤 시간이 갈수록 세션이 짧아짐 (사용 감소)
- 실제 넘어짐(fall)과 주행 실패 뒤 사용자 개입(stuck)은 가동 시간에 비례해 생김
  - 설치 첫 30일 2배, 사건 직후 재발 hazard가 높고, 기기마다 취약성(frailty)이 다름
- 넘어지면 대개 몇 시간 그대로 멈춰 있고, 일부는 곧 사람이 세워 다시 돈다
- stuck은 주행 기록이 10분 남짓 이어지다 사람이 들어 옮기면서 알림이 남음
- 관리 조작(handling)은 가동과 무관하게 사람이 옮긴 것. 일부는 옮긴 뒤 전원을 꺼 둠
- 들림과 낙하는 SPLIT 전에는 전도 코드로 찍힘
- 상담은 알림 일부에 붙고, AUTO_CONSULT 뒤로 많아짐. 넘어졌는지는 거의 적히지 않음
"""

import numpy as np
import pandas as pd
from common import AUTO_CONSULT, DATA, END, SEED, SPLIT, START

N_DEV = 3000
INSTALL_W = [0.32, 0.20, 0.13, 0.08, 0.07, 0.05, 0.05, 0.04, 0.03, 0.03]  # 월별 설치 비중. 출시 초기에 몰림
# 세션 시작 시각. 아침, 점심 뒤, 저녁에 켬
START_W = np.array([0, 0, 0, 0, 0, 0, 2, 6, 9, 6, 3, 2, 3, 5, 4, 2, 2, 3, 6, 5, 2, 1, 0, 0], float)
START_W /= START_W.sum()
HUMAN_W = np.array([1, 1, 1, 1, 1, 1, 2, 4, 6, 7, 7, 7, 8, 8, 7, 7, 7, 8, 9, 9, 7, 5, 3, 2], float)
HUMAN_W /= HUMAN_W.sum()
SESSION_H = 7.5  # 세션 길이 평균 (시간, 설치 직후)
USAGE_FLOOR, USAGE_TAU = 0.6, 120  # 사용 감소: 세션 길이 x (floor + (1 - floor) * exp(-age / tau))

RUN_H = 0.00025  # 가동 1시간당 fall + stuck hazard (frailty 1)
EARLY_MULT = 2.2  # 설치 첫 30일
RECUR_MULT, RECUR_TAU = 6.0, 4.0  # 사건 뒤 d일: 1 + RECUR_MULT * exp(-d / RECUR_TAU)
FRAILTY_K = 0.2  # 감마 shape. 작을수록 기기 차이가 큼
FALL_SHARE = 0.28  # fall + stuck 중 fall
DROP_SHARE = 0.16  # fall 중 낙하
FALL_LEFT_P = 0.72  # 넘어진 뒤 몇 시간 그대로 있음
HANDLE_RATE = 0.0043  # 하루 관리 조작 기대값 (handling 1). 첫 30일 2배
HANDLE_OFF_P = 0.05  # 옮긴 뒤 전원을 꺼 둠
LIFT_CODE_P = 0.7  # SPLIT 뒤 들어 옮긴 알림 중 LIFT로 찍히는 비중
DRIVE_PER_H = 0.05  # 가동 1시간당 평소 주행 기록

CONSULT_P = {"before": 0.15, "after": 0.36}  # 알림 사건 뒤 7일 안 상담
CONSULT_KIND = {"fall": 2.0, "stuck": 1.0, "handling": 1.0}
CONSULT_CODE_OK = 0.9  # 상담에 적힌 에러코드가 알림 코드와 같음
FALL_NOTE_P = 0.003  # 문의, 답변에 넘어짐 서술
OUTCOME = {
    "정상 작동 확인": 0.47,
    "전원, 원격 복구": 0.22,
    "안내, SMS": 0.14,
    "AS 접수": 0.01,
    "기타": 0.16,
}
OTHER_CODES = ["B1", "M3", "W1", "H14"]


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
            "truth_intensity": rng.lognormal(0, 0.4, N_DEV).round(3),  # 하루 세션 수
            "truth_frailty": rng.gamma(FRAILTY_K, 1 / FRAILTY_K, N_DEV).round(3),  # 사고 취약성
            "truth_handling": rng.lognormal(0, 0.6, N_DEV).round(3),  # 들어 옮기는 빈도
            "truth_offuse": rng.beta(1.2, 6, N_DEV).round(3),  # 하루 종일 꺼 두는 날 비중
        }
    )
    return dev.sort_values("install_date").reset_index(drop=True)


def make_sessions(rng, d, n_days):
    """하루 0~3개 세션. 분 단위 [start, end), 설치일 0시 기준"""
    rows = []
    for day in range(n_days):
        if rng.random() < d.truth_offuse:
            continue
        usage = USAGE_FLOOR + (1 - USAGE_FLOOR) * np.exp(-day / USAGE_TAU)
        for h in rng.choice(24, size=1 + rng.poisson(1.0 * d.truth_intensity), p=START_W):
            s = day * 1440 + h * 60 + rng.uniform(0, 60)
            rows.append((s, s + rng.gamma(2.0, SESSION_H * 30 * usage)))
    if not rows:
        return np.empty((0, 2))
    ss = np.array(sorted(rows))
    merged = [list(ss[0])]
    for s, e in ss[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.array(merged)


def cut(sessions, a, b):
    """세션에서 [a, b) 구간을 지움"""
    out = []
    for s, e in sessions:
        if e <= a or s >= b:
            out.append((s, e))
            continue
        if s < a:
            out.append((s, a))
        if e > b:
            out.append((b, e))
    return np.array(out) if out else np.empty((0, 2))


def running_at(sessions, t):
    return bool(((sessions[:, 0] <= t) & (sessions[:, 1] > t)).any())


def pick_run_time(rng, sessions, lo, hi):
    """[lo, hi) 안 가동 시간에서 균등하게 한 시점"""
    s = np.clip(sessions[:, 0], lo, hi)
    e = np.clip(sessions[:, 1], lo, hi)
    w = e - s
    if w.sum() <= 0:
        return None
    i = rng.choice(len(w), p=w / w.sum())
    return s[i] + rng.uniform(0, w[i])


def gen_device(rng, d):
    n_days = (END - d.install_date).days
    ses = make_sessions(rng, d, n_days)
    events = []  # (minute, kind, truth_event)
    cuts = []
    # fall, stuck: 날마다 가동 시간으로 hazard
    run_by_day = np.zeros(n_days)
    for s, e in ses:
        for day in range(int(s // 1440), min(int(e // 1440), n_days - 1) + 1):
            run_by_day[day] += max(0.0, min(e, (day + 1) * 1440) - max(s, day * 1440)) / 60
    last = None
    for day in range(n_days):
        if run_by_day[day] <= 0:
            continue
        mult = d.truth_frailty * (EARLY_MULT if day < 30 else 1.0)
        if last is not None:
            mult *= 1 + RECUR_MULT * np.exp(-(day - last) / RECUR_TAU)
        if rng.random() >= 1 - np.exp(-RUN_H * mult * run_by_day[day]):
            continue
        t = pick_run_time(rng, ses, day * 1440, (day + 1) * 1440)
        if t is None:
            continue
        last = day
        if rng.random() < FALL_SHARE:
            kind = "fall_drop" if rng.random() < DROP_SHARE else "fall_tip"
            stay = 360 + rng.exponential(480) if rng.random() < FALL_LEFT_P else rng.uniform(5, 40)
            cuts.append((t, t + stay))
        else:
            kind = "stuck"
            cuts.append((t, t + rng.uniform(2, 20)))
        events.append((t, kind))
    # handling: 사람 활동 시각에 맞춰 가동과 무관하게
    age = np.arange(n_days)
    lam = HANDLE_RATE * d.truth_handling * np.where(age < 30, 2.0, 1.0)
    for day in np.flatnonzero(rng.random(n_days) < lam):
        t = day * 1440 + rng.choice(24, p=HUMAN_W) * 60 + rng.uniform(0, 60)
        if rng.random() < HANDLE_OFF_P:
            cuts.append((t, t + rng.uniform(120, 600)))
        else:
            cuts.append((t, t + rng.uniform(2, 15)))
        events.append((t, "handling"))
    for a, b in cuts:
        ses = cut(ses, a, b)
    return ses, sorted(events)


def alert_rows(rng, serial, base, t, kind, eid):
    """사건 하나가 남기는 알림 행. stuck은 앞에 주행 기록이 이어짐"""
    rows = []
    ts0 = base + pd.Timedelta(minutes=float(t))
    if kind == "stuck":
        span = rng.lognormal(np.log(11), 0.6)
        for g in np.sort(rng.uniform(0, span, 2 + rng.poisson(3))):
            code = "N1" if rng.random() < 0.6 else "N2"
            rows.append((serial, ts0 - pd.Timedelta(minutes=float(span - g) + 3), code, "stuck_drive", eid))
    if kind == "fall_drop":
        code = "DROP" if ts0 >= SPLIT else "TIP"
    elif kind == "fall_tip":
        code = "TIP"
    else:
        code = "LIFT" if ts0 >= SPLIT and rng.random() < LIFT_CODE_P else "TIP"
    gaps = np.concatenate([[0.0], rng.exponential(10, rng.poisson(0.7))])
    for g in np.cumsum(gaps):
        rows.append((serial, ts0 + pd.Timedelta(minutes=float(g)), code, kind, eid))
    return rows


def gen_consults(rng, first):
    """알림 사건 하나에 상담 하나까지. first는 사건별 첫 알림"""
    after = first["ts"] >= AUTO_CONSULT
    p = np.where(after, CONSULT_P["after"], CONSULT_P["before"])
    p = p * first["truth_kind"].str.split("_").str[0].map(CONSULT_KIND).to_numpy()
    c = first[rng.random(len(first)) < p].copy()
    c["ts"] = c["ts"] + pd.to_timedelta(np.minimum(rng.exponential(1.2, len(c)), 6.9), unit="D")
    ok = rng.random(len(c)) < CONSULT_CODE_OK
    c["error_code"] = np.where(ok, c["code"], rng.choice(OTHER_CODES, len(c)))
    c["outcome"] = rng.choice(list(OUTCOME), size=len(c), p=list(OUTCOME.values()))
    c["fall_note"] = rng.random(len(c)) < FALL_NOTE_P
    c = c[c["ts"] < END].sort_values(["ts", "serial"], kind="stable").reset_index(drop=True)
    c.insert(0, "consult_id", np.arange(1, len(c) + 1))
    c["ts"] = c["ts"].dt.floor("s")
    return c[["consult_id", "serial", "ts", "error_code", "outcome", "fall_note", "truth_event"]]


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    dev = make_devices(rng)
    ses_rows, ev_rows, eid = [], [], 0
    for d in dev.itertuples(index=False):
        ses, events = gen_device(rng, d)
        base = d.install_date
        for s, e in ses:
            ses_rows.append((d.serial, base + pd.Timedelta(minutes=float(s)), base + pd.Timedelta(minutes=float(e))))
        for t, kind in events:
            eid += 1
            ev_rows += alert_rows(rng, d.serial, base, t, kind, eid)
        # 평소 주행 기록: 가동 중에만
        hours = (ses[:, 1] - ses[:, 0]).sum() / 60 if len(ses) else 0
        for _ in range(rng.poisson(DRIVE_PER_H * hours)):
            t = pick_run_time(rng, ses, 0, np.inf)
            code = rng.choice(["N1", "N2", "N3", "N4"], p=[0.21, 0.46, 0.13, 0.20])
            ev_rows.append((d.serial, base + pd.Timedelta(minutes=float(t)), code, "drive", 0))
    ses = pd.DataFrame(ses_rows, columns=["serial", "start", "end"])
    ses = ses[ses["start"] < END]
    ses["end"] = ses["end"].clip(upper=END)
    for c in ["start", "end"]:
        ses[c] = ses[c].dt.floor("s")
    ev = pd.DataFrame(ev_rows, columns=["serial", "ts", "code", "truth_kind", "truth_event"])
    ev["ts"] = ev["ts"].dt.floor("s")
    ev = ev[(ev["ts"] >= START) & (ev["ts"] < END)]
    ev = ev.sort_values(["serial", "ts", "code"], kind="stable").reset_index(drop=True)
    ev.insert(0, "id", np.arange(1, len(ev) + 1))
    alerts = ev[ev["code"].isin(["TIP", "DROP", "LIFT"])]
    first = alerts.groupby("truth_event", sort=True).head(1)
    con = gen_consults(rng, first)

    gz = {"method": "gzip", "mtime": 0}  # mtime 고정: 다시 돌려도 byte 동일
    dev.to_csv(DATA / "devices.csv", index=False)
    ses.to_csv(DATA / "sessions.csv.gz", index=False, compression=gz)
    ev.to_csv(DATA / "events.csv.gz", index=False, compression=gz)
    con.to_csv(DATA / "consults.csv", index=False)
    print(f"devices {len(dev):,}  sessions {len(ses):,}  events {len(ev):,}  consults {len(con):,}")
    print(ev.loc[ev["code"].isin(["TIP", "DROP", "LIFT"]), "truth_kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
