# 심은 신호 3종이 실제로 데이터에 들어갔는지 확인하는 검증 플롯
#
#   fig1 — W3: 이벤트 시점 정렬(event-time alignment) 시 7일 전부터 선형 증가하는가
#   fig2 — 사용량 분산: 14일 전부터 기기 간 분산이 커지는가 (평균은 유지)
#   fig3 — W7: 상시 높은 기기군이 존재하고, 그 군은 심각 이벤트와 무관한가
#
# 실행: .venv/bin/python notebooks/02_validate_signals.py
# 출력: outputs/figures/fig1_w3_ramp.png, fig2_usage_variance.png, fig3_w7_confounder.png

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(base / "data" / "device_events.csv", parse_dates=["event_date"])
gt = pd.read_csv(base / "data" / "ground_truth.csv", parse_dates=["severe_event_date"])

start = df["event_date"].min()
n_days = (df["event_date"].max() - start).days + 1
df["day"] = (df["event_date"] - start).dt.days
gt["event_day"] = (gt["severe_event_date"] - start).dt.days

severe = gt[gt["group"] == "severe"]
normal = gt[gt["group"] == "normal"]
confounder = gt[gt["group"] == "confounder"]


def daily_matrix(code: str) -> pd.DataFrame:
    """기기 × 날짜 전체 그리드 (없는 날은 0건)."""
    sub = df[df["event_code"] == code]
    wide = sub.pivot_table(index="device_id", columns="day", values="count", aggfunc="sum")
    return wide.reindex(index=gt["device_id"], columns=range(n_days)).fillna(0)


def align_to_event(wide: pd.DataFrame, devices: pd.DataFrame, rel_range: tuple[int, int]) -> np.ndarray:
    """이벤트일 기준 상대 날짜(rel_range)로 정렬한 (기기 × 상대일) 행렬. 범위 밖은 NaN."""
    rel_days = np.arange(rel_range[0], rel_range[1] + 1)
    out = np.full((len(devices), len(rel_days)), np.nan)
    for r, (_, row) in enumerate(devices.iterrows()):
        t = int(row["event_day"])
        series = wide.loc[row["device_id"]].to_numpy()
        for c, rel in enumerate(rel_days):
            d = t + rel
            if 0 <= d < n_days:
                out[r, c] = series[d]
    return out


# 대조군(normal)에는 가짜 이벤트일을 부여해 같은 방식으로 정렬한다
rng = np.random.default_rng(SEED + 1)
normal = normal.assign(event_day=rng.integers(30, n_days, size=len(normal)))

REL = (-30, 5)
rel_days = np.arange(REL[0], REL[1] + 1)

# ── fig1: W3 선형 램프 ───────────────────────────────────────────
w3 = daily_matrix("W3")
w3_severe = align_to_event(w3, severe, REL)
w3_control = align_to_event(w3, normal, REL)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(rel_days, np.nanmean(w3_severe, axis=0), marker="o", ms=3,
        color="tab:red", label=f"심각 이벤트 기기 (n={len(severe)})")
ax.plot(rel_days, np.nanmean(w3_control, axis=0), marker="o", ms=3,
        color="tab:gray", alpha=0.7, label=f"일반 기기 + 가짜 이벤트일 (n={len(normal)})")
ax.axvline(-7, color="tab:blue", ls="--", lw=1, label="신호 시작 (-7일)")
ax.axvline(0, color="black", ls=":", lw=1)
ax.set_xlabel("심각 이벤트까지 남은 일수")
ax.set_ylabel("W3 일 평균 발생 건수")
ax.set_title("신호 (1) 검증 — W3 경고가 이벤트 7일 전부터 선형 증가")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "fig1_w3_ramp.png", dpi=150)

peak = np.nanmean(w3_severe, axis=0)[rel_days == 0][0]
baseline = np.nanmean(w3_severe[:, rel_days < -7])
print(f"[fig1] W3 심각 기기: 이벤트 당일 평균 {peak:.2f}건 vs 램프 이전 평균 {baseline:.2f}건")

# ── fig2: 사용량 분산 증가 (평균은 유지) ─────────────────────────
u1 = daily_matrix("U1")
# 기기별 베이스라인(이벤트 30일 이전 평균)으로 정규화해 규모 차이를 제거
u1_severe = align_to_event(u1, severe, (-60, 5))
u1_control = align_to_event(u1, normal, (-60, 5))
rel_wide = np.arange(-60, 6)

def normalize(mat: np.ndarray) -> np.ndarray:
    # 신호 구간(-14일~) 밖의 평균으로 정규화. 이벤트일이 이른 기기도 빈 구간이 없도록 -15일 기준.
    base_mean = np.nanmean(mat[:, rel_wide < -15], axis=1, keepdims=True)
    return mat / base_mean

u1_severe_n = normalize(u1_severe)
u1_control_n = normalize(u1_control)

fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
axes[0].plot(rel_wide, np.nanstd(u1_severe_n, axis=0), color="tab:red", label="심각 이벤트 기기")
axes[0].plot(rel_wide, np.nanstd(u1_control_n, axis=0), color="tab:gray", alpha=0.7,
             label="일반 기기 + 가짜 이벤트일")
axes[0].axvline(-14, color="tab:blue", ls="--", lw=1, label="신호 시작 (-14일)")
axes[0].axvline(0, color="black", ls=":", lw=1)
axes[0].set_ylabel("정규화 사용량의 기기 간 표준편차")
axes[0].set_title("신호 (2) 검증 — 이벤트 14일 전부터 사용량 분산 증가, 평균은 유지")
axes[0].legend(fontsize=9)

axes[1].plot(rel_wide, np.nanmean(u1_severe_n, axis=0), color="tab:red", label="심각 이벤트 기기")
axes[1].plot(rel_wide, np.nanmean(u1_control_n, axis=0), color="tab:gray", alpha=0.7,
             label="일반 기기 + 가짜 이벤트일")
axes[1].axvline(-14, color="tab:blue", ls="--", lw=1)
axes[1].axvline(0, color="black", ls=":", lw=1)
axes[1].set_ylim(0.8, 1.2)
axes[1].set_xlabel("심각 이벤트까지 남은 일수")
axes[1].set_ylabel("정규화 사용량 평균")
fig.tight_layout()
fig.savefig(fig_dir / "fig2_usage_variance.png", dpi=150)

std_in = np.nanmean(np.nanstd(u1_severe_n[:, (rel_wide >= -14) & (rel_wide <= 0)], axis=0))
std_out = np.nanmean(np.nanstd(u1_severe_n[:, rel_wide < -14], axis=0))
print(f"[fig2] 정규화 사용량 표준편차: 신호 구간 {std_in:.3f} vs 이전 {std_out:.3f}")

# ── fig3: W7 교란 신호 ───────────────────────────────────────────
w7 = daily_matrix("W7")
w7_daily_mean = w7.mean(axis=1)  # 기기별 W7 일 평균
groups = ["confounder", "severe", "normal"]
labels = {
    "confounder": f"W7 상시 높음 군\n(n={len(confounder)})",
    "severe": f"심각 이벤트 군\n(n={len(severe)})",
    "normal": f"일반 군\n(n={len(normal)})",
}
means = [w7_daily_mean[gt.set_index("device_id")["group"].reindex(w7.index) == g].mean() for g in groups]
severe_rate = [
    (gt[gt["group"] == g]["severe_event_date"].notna().mean()) for g in groups
]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar([labels[g] for g in groups], means,
              color=["tab:orange", "tab:red", "tab:gray"], width=0.55)
for b, rate in zip(bars, severe_rate):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06,
            f"심각 이벤트 경험률 {rate:.0%}", ha="center", fontsize=10)
ax.set_ylabel("W7 일 평균 발생 건수")
ax.set_ylim(0, max(means) * 1.25)
ax.set_title("신호 (3) 검증 — W7이 상시 높은 군은 심각 이벤트와 무관 (교란 신호)")
fig.tight_layout()
fig.savefig(fig_dir / "fig3_w7_confounder.png", dpi=150)

print(f"[fig3] W7 일 평균: confounder={means[0]:.2f}, severe={means[1]:.2f}, normal={means[2]:.2f}")
print(f"[fig3] 심각 이벤트 경험률: confounder={severe_rate[0]:.0%}, severe={severe_rate[1]:.0%}, normal={severe_rate[2]:.0%}")
print(f"saved: {fig_dir}/fig1_w3_ramp.png, fig2_usage_variance.png, fig3_w7_confounder.png")
