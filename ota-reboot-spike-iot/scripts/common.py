"""경로, 상수, 공용 함수. 스크립트 파일 위치 기준이라 어디서 실행해도 된다."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"

SEED = 42
START = pd.Timestamp("2025-06-02")
END = pd.Timestamp("2025-09-14")  # inclusive

MAIN, SUB, G1, G2 = "#9E3D22", "#5B6B7A", "#B4B2A9", "#8A8F98"


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / mad
