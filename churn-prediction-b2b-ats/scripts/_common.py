"""공통 상수·경로·그림 스타일."""
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

REF_DATE = pd.Timestamp("2024-02-15")      # 노트북 실행일 — 라벨의 '오늘'
CHURN_GRACE_DAYS = 90                      # 최종 종료일 + 90일 초과 → 이탈
CUTOFF_T = pd.Timestamp("2023-02-15")      # v2 시간 절단 시점 (REF 12개월 전)
DATA_END = pd.Timestamp("2026-06-30")      # 기준일 스윕용 전체 이력 생성 상한

def rng(offset=0):
    return np.random.default_rng(SEED + offset)

def churn_label(last_end: pd.Series, ref: pd.Timestamp) -> pd.Series:
    """원본 규칙: (ref - 최종종료일).days > 90 이면 이탈(1)."""
    return ((ref - last_end).dt.days > CHURN_GRACE_DAYS).astype(int)

# 그림 스타일
GRAY, BLUE, ORANGE, GREEN, RED = "#8a8f98", "#2f6fd6", "#e8833a", "#3a9d6b", "#c94a4a"
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import glob, warnings
    warnings.filterwarnings("ignore", message="Glyph")
    names = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in ["Pretendard", "NanumBarunGothic", "Noto Sans CJK KR", "AppleGothic"] if n in names), None)
    if chosen is None:
        for path in glob.glob("/usr/share/fonts/**/NotoSansCJK*Regular*", recursive=True):
            font_manager.fontManager.addfont(path)
        cjk = [f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name]
        chosen = next((n for n in cjk if "KR" in n), cjk[0] if cjk else "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams.update({
        "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "savefig.bbox": "tight",
    })
    return plt
