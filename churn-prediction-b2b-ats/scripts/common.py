"""공통 상수, 경로, mpl 설정"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
for p in (DATA, RESULTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

REF_DATE = pd.Timestamp("2024-02-15")  # 원본 노트북 실행일
CHURN_GRACE_DAYS = 90
CUTOFF_T = pd.Timestamp("2023-02-15")  # v2 절단 시점 (REF - 12M)
DATA_END = pd.Timestamp("2026-06-30")  # 06 sweep 때문에 미래까지 생성


def rng(offset=0):
    return np.random.default_rng(SEED + offset)


def churn_label(last_end: pd.Series, ref: pd.Timestamp) -> pd.Series:
    # 원본 Churn(): 최종 종료일 + 90일 지나면 1
    return ((ref - last_end).dt.days > CHURN_GRACE_DAYS).astype(int)


GRAY, BLUE, ORANGE, GREEN, RED = "#B4B2A9", "#5B6B7A", "#9E3D22", "#8A8F98", "#9E3D22"


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import glob
    import warnings

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    warnings.filterwarnings("ignore", message="Glyph")
    names = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next(
        (n for n in ["Pretendard", "NanumBarunGothic", "Noto Sans CJK KR", "AppleGothic"] if n in names), None
    )
    if chosen is None:
        for path in glob.glob("/usr/share/fonts/**/NotoSansCJK*Regular*", recursive=True):
            font_manager.fontManager.addfont(path)
        cjk = [f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name]
        chosen = next((n for n in cjk if "KR" in n), cjk[0] if cjk else "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )
    return plt
