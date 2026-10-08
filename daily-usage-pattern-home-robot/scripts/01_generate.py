"""기기 1,500대 x 평일 20일 사용 이벤트 생성.

가상데이터에 넣은 구조
- 기기 유형 (truth_type): 저활동 50% / 활동 46% / 활동 + 공간 스캔 자동 실행 4%
- 하루 상태 (truth_state): 기기마다 선호 상태가 있고 날마다 일관성 확률로 그 상태, 아니면 다른 상태
  S1 종일 제자리 모드 + 기본 스케줄 / S2 저녁 제자리 모드 + 이동, 센서 호출 / S3 음성 대화만
- 기본 스케줄은 이동 모드 이벤트로 남김 (truth_sched = 1). 02 가 시각과 길이로 다시 가름 (휴리스틱)
- 사용자가 스케줄 시각에 짧게 돌린 이동 모드도 섞여 있어 휴리스틱이 일부 오분류
- 하루의 60% 는 다른 상태 이벤트가 일부 섞인 날 (경계 흐림)
- 제자리 모드는 켜 둔 동안 중간에 껐다 다시 켜기도 함 (실제 쓴 날 하루 시작 4건 남짓)
"""

import numpy as np
import pandas as pd
from common import DATA, DAYS, N_DEVICES, SCHEDULE_HOURS, SEED

rng = np.random.default_rng(SEED)
STATES = ["S1", "S2", "S3"]
BLEND_P = 0.6  # 다른 상태가 섞인 날의 비율 (경계 흐림)
BLEND_P_LOW = 0.15  # 저활동 기기는 쓰는 날에도 음성 위주라 다른 상태가 덜 섞임
BLEND_KEEP = 0.5  # 섞인 상태의 이벤트 중 남기는 비율
FIXED_OFF_RATE = 1.0  # 제자리 모드를 켜 둔 동안 껐다 다시 켜는 횟수 (포아송 평균)


def devices() -> pd.DataFrame:
    t = rng.choice(["low", "active", "active_scan"], size=N_DEVICES, p=[0.50, 0.46, 0.04])
    pref = np.where(t == "low", "S3", np.where(rng.random(N_DEVICES) < 0.48, "S1", "S2"))
    d = pd.DataFrame(
        {
            "serial": [f"D{i:05d}" for i in range(N_DEVICES)],
            "truth_type": t,
            "truth_pref": pref,
            "truth_schedule_on": (pref == "S1") | (rng.random(N_DEVICES) < 0.04),
            "truth_active_p": np.where(t == "low", rng.beta(3, 5, N_DEVICES), rng.beta(18, 1.6, N_DEVICES)),
            "truth_consistency": np.where(t == "low", rng.beta(20, 2, N_DEVICES), rng.beta(17, 3, N_DEVICES)),
            "truth_voice_user": rng.random(N_DEVICES) < np.where(t == "low", 0.9, 0.72),
            "truth_voice_rate": rng.lognormal(np.log(7), 0.5, N_DEVICES),
            "truth_voice_peak": rng.choice(
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                size=N_DEVICES,
                p=[0.03, 0.03, 0.06, 0.12, 0.18, 0.16, 0.13, 0.11, 0.1, 0.08],
            ),
            "truth_camera_user": rng.random(N_DEVICES) < 0.25,
            "truth_follow_user": rng.random(N_DEVICES) < 0.06,
            "truth_vital_user": rng.random(N_DEVICES) < 0.015,
            "truth_sensor": rng.random(N_DEVICES) < 0.42,
            "truth_scan_hour": rng.integers(9, 17, N_DEVICES),
        }
    )
    return d


def ev(out, d, day, state, feat, h, dur_min, sched=0):
    if h < 0 or h >= 24:
        return
    ts = day + pd.Timedelta(minutes=float(h * 60))
    end_cap = (24 - h) * 60
    out.append((d.serial, day, ts, feat, round(min(dur_min, end_cap) * 60), state, sched))


def fixed_on(out, d, day, state, start, end):
    """start ~ end 동안 제자리 모드. 중간에 10~60분 꺼 둔 틈이 생기면 그 뒤는 새로 켠 것"""
    cuts = np.sort(rng.uniform(start, end, rng.poisson(FIXED_OFF_RATE)))
    s = start
    for c in cuts:
        if c - s < 0.25:
            continue
        ev(out, d, day, state, "fixed", s, (c - s) * 60)
        s = c + rng.uniform(10, 60) / 60
    if end - s > 0.1:
        ev(out, d, day, state, "fixed", s, (end - s) * 60)


def day_events(out, d, day, state):
    if state == "S1":
        hrs = []
        if d.truth_schedule_on:
            for h in SCHEDULE_HOURS:
                s = h + rng.uniform(0, 50) / 60
                dur = rng.uniform(3, 9)
                ev(out, d, day, state, "patrol", s, dur, sched=1)
                hrs.append((s, dur))
            for i, (s, dur) in enumerate(hrs):
                start = s + dur / 60
                stop = hrs[i + 1][0] if i + 1 < len(hrs) else 24
                ev(out, d, day, state, "fixed", start, (stop - start) * 60)
        else:
            fixed_on(out, d, day, state, rng.uniform(8, 10), 24)
        if rng.random() < 0.3:
            s = rng.uniform(6.5, 8.5)
            ev(out, d, day, state, "fixed", s, (9 - s) * 60)
        extra = {"patrol": 0.15, "autocall": 0.23 if d.truth_sensor else 0, "voice": 0.19, "camera": 0.04}
    elif state == "S2":
        if rng.random() < 0.5:
            s = rng.uniform(7, 9)
            fixed_on(out, d, day, state, s, s + rng.uniform(2, 5))
        fixed_on(out, d, day, state, rng.uniform(17.5, 20), 24)
        if rng.random() < 0.64:
            for _ in range(1 + rng.poisson(0.7)):
                ev(out, d, day, state, "patrol", rng.uniform(9, 21), rng.uniform(6, 45))
        if d.truth_sensor and rng.random() < 0.85:
            for _ in range(1 + rng.poisson(3)):
                ev(out, d, day, state, "autocall", rng.uniform(8, 22), rng.uniform(10, 40))
        extra = {"voice": 0.18, "camera": 0.07}
    else:
        if rng.random() < 0.10:
            s = rng.uniform(8, 20)
            ev(out, d, day, state, "fixed", s, rng.uniform(1, 3) * 60)
        if d.truth_sensor and rng.random() < 0.3:
            ev(out, d, day, state, "autocall", rng.uniform(7, 9), rng.uniform(10, 30))
        if rng.random() < 0.08:
            ev(out, d, day, state, "patrol", rng.uniform(9, 21), rng.uniform(6, 40))
        extra = {"voice": 0.88 if d.truth_voice_user else 0.25, "camera": 0.06}
    for feat, p in extra.items():
        if rng.random() >= p:
            continue
        if feat == "voice":
            if state != "S3" and not d.truth_voice_user:
                continue
            n = 2 + rng.poisson(d.truth_voice_rate) if state == "S3" else 1 + rng.poisson(2)
            for _ in range(n):
                ev(out, d, day, state, "voice", (d.truth_voice_peak + rng.normal(0, 0.6)) % 24, 0)
        elif feat == "camera" and (d.truth_camera_user or state == "S3"):
            for _ in range(1 + rng.poisson(1)):
                ev(out, d, day, state, "camera", rng.uniform(15, 19.5), rng.uniform(2, 15))
        elif feat in ("patrol", "autocall"):
            ev(out, d, day, state, feat, rng.uniform(9, 21), rng.uniform(6, 40))
    if d.truth_follow_user and rng.random() < 0.05:
        ev(out, d, day, state, "follow", rng.uniform(10, 21), rng.uniform(2, 6))
    if d.truth_vital_user and rng.random() < 0.06:
        ev(out, d, day, state, "vital", rng.uniform(7, 22), 0)
    if d.truth_type == "active_scan" and rng.random() < 0.88:
        ev(out, d, day, state, "spacescan", d.truth_scan_hour + rng.uniform(0, 0.5), 0)


def main():
    dev = devices()
    out = []
    for d in dev.itertuples(index=False):
        for day in DAYS:
            if rng.random() >= d.truth_active_p:
                if rng.random() < 0.3:  # 쓰지 않은 날의 흔적 한 건 (음성은 음성 쓰는 기기만)
                    f = "voice" if d.truth_voice_user and rng.random() < 0.5 else "autocall" if d.truth_sensor else None
                    if f:
                        ev(out, d, day, "idle", f, rng.uniform(8, 22), rng.uniform(5, 20))
                continue
            if rng.random() < d.truth_consistency:
                state = d.truth_pref
            else:
                state = rng.choice([s for s in STATES if s != d.truth_pref])
            day_events(out, d, day, state)
            if rng.random() < (BLEND_P_LOW if d.truth_type == "low" else BLEND_P):  # 다른 상태 하나를 일부만 섞음
                tmp = []
                day_events(tmp, d, day, rng.choice([s for s in STATES if s != state]))
                out.extend((*t[:5], state, t[6]) for t in tmp if rng.random() < BLEND_KEEP)
    e = pd.DataFrame(out, columns=["serial", "dt", "start_ts", "feature", "dur_sec", "truth_state", "truth_sched"])
    e = e.sort_values(["serial", "start_ts", "feature"], kind="stable").reset_index(drop=True)
    e.loc[e["feature"].isin(["voice", "vital", "spacescan"]), "dur_sec"] = 0
    DATA.mkdir(parents=True, exist_ok=True)
    dev.to_csv(DATA / "devices.csv", index=False)
    e.to_csv(
        DATA / "events.csv.gz", index=False, date_format="%Y-%m-%d %H:%M:%S", compression={"method": "gzip", "mtime": 0}
    )
    print(f"devices {len(dev):,}, events {len(e):,}")
    print(e["feature"].value_counts().to_string())


if __name__ == "__main__":
    main()
