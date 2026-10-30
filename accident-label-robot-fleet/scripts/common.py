"""경로, 기간, 사건 규칙 상수. 스크립트 위치 기준이라 어디서 실행해도 됨"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "outputs" / "results"

SEED = 42

START = pd.Timestamp("2025-01-06")  # 출시일. 로그 하한
END = pd.Timestamp("2025-11-01")  # 관측 종료 (이 날 제외)
SPLIT = pd.Timestamp("2025-07-01")  # 펌웨어 배포일. 들림 LIFT, 낙하 DROP 코드가 이날부터 따로 찍힘
AUTO_CONSULT = pd.Timestamp("2025-08-11")  # 전도, 낙하가 관제 등급 1로 올라 상담이 자동으로 열리기 시작한 날

ALERT = ["TIP", "DROP", "LIFT"]  # 주행 불가 알림
DRIVE = ["N1", "N2", "N3", "N4"]

# 사건 규칙. 묶기는 1편의 전도 X(120분)를 들림까지 넓혀 씀
BUNDLE_MIN = 120
PRE_RUN_MIN = 60  # 직전 가동을 보는 창
PRE_DRIVE_MIN = 30  # 직전 주행 기록을 보는 창
POST_WIN = (60, 360)  # 알림 뒤 멈춤을 보는 창 (분)
PRE_DRIVE_GRID = [10, 30, 60]
POST_GRID = [(60, 180), (60, 360)]

K_DAYS = 30
K_GRID = [None, 7, 14, 30, 42]
CURVE_DAYS = 60
LONG_RUN = (31, 60)  # 장기 수준을 보는 경과일

TYPE_ORDER = ["A", "B", "C", "D"]
TYPE_NAME = {
    "A": "A 실제 넘어짐 (상한)",
    "B": "B 주행 실패 뒤 사용자 개입",
    "C": "C 관리 조작",
    "D": "D 판정 불가",
}


def load_sessions() -> pd.DataFrame:
    return pd.read_csv(DATA / "sessions.csv.gz", parse_dates=["start", "end"])


def load_events() -> pd.DataFrame:
    return pd.read_csv(DATA / "events.csv.gz", parse_dates=["ts"])


def load_devices() -> pd.DataFrame:
    return pd.read_csv(DATA / "devices.csv", parse_dates=["install_date"])


BIG = 10**7  # 기기마다 시간축을 이만큼 밀어 한 줄로 이어 붙임 (관측 기간 약 43만 분)


def minutes(ts):
    return (ts - START).dt.total_seconds().to_numpy() / 60


class Timeline:
    """기기별 세션, 주행 기록을 한 축에 놓고 구간 질의"""

    def __init__(self, ses, drive):
        serials = sorted(set(ses["serial"]) | set(drive["serial"]))
        self.key = {s: i * BIG for i, s in enumerate(serials)}
        off = ses["serial"].map(self.key).to_numpy()
        order = np.argsort(minutes(ses["start"]) + off, kind="stable")
        self.s = (minutes(ses["start"]) + off)[order]
        self.e = (minutes(ses["end"]) + off)[order]
        self.d = np.sort(minutes(drive["ts"]) + drive["serial"].map(self.key).to_numpy())

    def offset(self, serial):
        return serial.map(self.key).to_numpy()

    def run_overlap(self, a, b):
        """[a, b)에 걸친 세션이 있나. 세션은 기기 안에서 겹치지 않아 end도 정렬돼 있음"""
        i = np.searchsorted(self.e, a, side="right")
        ok = i < len(self.s)
        return ok & (self.s[np.minimum(i, len(self.s) - 1)] < b)

    def drive_count(self, a, b):
        return np.searchsorted(self.d, b, side="left") - np.searchsorted(self.d, a, side="left")
