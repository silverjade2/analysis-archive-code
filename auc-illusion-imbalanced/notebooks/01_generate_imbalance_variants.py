# 3호 글용 가상데이터 생성 — 양성 비율만 다른 이진 분류 3버전
#
# 설계 핵심: 클래스 조건부 분포 P(x|y)를 세 버전에서 완전히 동일하게 두고
# 양성 비율 P(y=1)만 3% / 10% / 30%로 바꾼다.
#   → ROC-AUC는 조건부 분포(분리도)만의 함수라 세 버전에서 거의 같아야 하고,
#     PR-AUC·precision@k는 유병률(prevalence)에 민감해 크게 갈려야 한다.
#   이 대비가 "지표가 만드는 착시"의 재료다.
#
# 스키마: imbalance_pos{3,10,30}.csv — f01~f10 (피처 10개) + y
#   - f01~f06: 정보 피처 (y=1이면 평균 +0.9 이동, 표준편차 1)
#   - f07~f10: 노이즈 피처 (양 클래스 동일 분포)
#
# 실행: .venv/bin/python notebooks/01_generate_imbalance_variants.py
# 출력: data/imbalance_pos{3,10,30}.csv
#       outputs/figures/fig1_conditional_dists.png (조건부 분포 동일성 검증)

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
N = 80_000  # 양성 3%에서도 학습에 충분한 양성 수(~1,700개/train)를 확보
POS_RATES = {"pos3": 0.03, "pos10": 0.10, "pos30": 0.30}
N_INFORMATIVE = 6
N_NOISE = 4
SHIFT = 0.9  # 정보 피처의 클래스 간 평균 이동량 → 베이즈 ROC-AUC ≈ 0.94

rng = np.random.default_rng(SEED)
base = Path(__file__).resolve().parents[1]
data_dir = base / "data"
data_dir.mkdir(parents=True, exist_ok=True)
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

feature_cols = [f"f{i:02d}" for i in range(1, N_INFORMATIVE + N_NOISE + 1)]


def generate(pos_rate: float, r: np.random.Generator) -> pd.DataFrame:
    """조건부 분포는 고정, 양성 비율만 다르게 생성."""
    y = (r.random(N) < pos_rate).astype(int)
    X = r.normal(0, 1, size=(N, N_INFORMATIVE + N_NOISE))
    X[:, :N_INFORMATIVE] += SHIFT * y[:, None]  # 정보 피처만 이동
    df = pd.DataFrame(X.round(4), columns=feature_cols)
    df["y"] = y
    return df


dfs = {}
for name, rate in POS_RATES.items():
    df = generate(rate, rng)
    df.to_csv(data_dir / f"imbalance_{name}.csv", index=False)
    dfs[name] = df
    print(f"imbalance_{name}.csv — {len(df)}행, 양성 {df['y'].mean():.3%} ({df['y'].sum()}개)")

# 이론 베이즈 AUC (참고): 정보 피처 6개 합산 시 유효 이동 = SHIFT*sqrt(6)
from scipy.stats import norm

bayes_auc = norm.cdf(SHIFT * np.sqrt(N_INFORMATIVE) / np.sqrt(2))
print(f"\n이론 베이즈 ROC-AUC ≈ {bayes_auc:.3f} (조건부 분포가 같으니 세 버전 공통)")

# ── 검증 플롯: 조건부 분포가 세 버전에서 동일한가 ─────────────────
# 최적 방향 점수(정보 피처 합)를 클래스별로 그려 비교한다.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
for ax, (name, df) in zip(axes, dfs.items()):
    score = df[feature_cols[:N_INFORMATIVE]].sum(axis=1)
    for label, color in [(0, "tab:gray"), (1, "tab:red")]:
        sub = score[df["y"] == label]
        ax.hist(sub, bins=60, density=True, alpha=0.55, color=color,
                label=f"y={label} (n={len(sub):,})")
    ax.set_title(f"양성 {POS_RATES[name]:.0%}")
    ax.set_xlabel("정보 피처 합")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("밀도")
axes[0].legend(fontsize=8)
fig.suptitle("클래스 조건부 분포는 세 버전에서 동일 — 다른 것은 양성 비율뿐 (밀도 기준)")
fig.tight_layout()
fig.savefig(fig_dir / "fig1_conditional_dists.png", dpi=150)
print(f"검증 플롯 저장 — {fig_dir / 'fig1_conditional_dists.png'}")
