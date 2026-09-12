"""그림 5장. 값은 02와 04의 결과 CSV와 03이 쓴 scored_synthetic.csv에서 읽고 다시 계산하지 않는다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import DATA, PARAMS, RES
from sitestyle import BLUE, DARK, GRAY, LIGHT, MUTED, RED, save, setup

setup()
STAGE_COLOR = {"안정": GRAY, "주의": BLUE, "경고": RED}  # ORANGE와 RED가 같은 색이라 3단계는 회색/슬레이트/러스트

cv = pd.read_csv(RES / "item_curves.csv")
t = cv[cv.item == "temperature"]
p = PARAMS["temperature"]

fig, ax = plt.subplots(figsize=(8, 3.9))
ax.axvspan(p["normal_low"], p["normal_high"], color=LIGHT, alpha=0.6, lw=0)
for x in (p["lower_limit"], p["upper_limit"]):
    ax.axvline(x, color=MUTED, ls=":", lw=1)
ax.plot(t.value, t.score_current, color=GRAY, lw=2, label="원본 (경계 중심, 원 단위 k=5)")
ax.plot(t.value, t.score_v2, color=BLUE, lw=2, label="재설계 (정규화 d, k=6)")
steps = pd.read_csv(RES / "boundary_steps.csv")
s = steps[(steps.item == "temperature") & (steps.over_by == 0.01)].iloc[0]
ax.annotate(
    f"37.51℃: {s.score_current:.1f}점",
    xy=(s.value, s.score_current),
    xytext=(38.3, 60),
    fontsize=9,
    color=DARK,
    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
)
s = steps[(steps.item == "temperature") & (steps.over_by == 2.5)].iloc[0]
ax.annotate(
    f"40.0℃ (임계치): {s.score_v2:.1f}점",
    xy=(s.value, s.score_v2),
    xytext=(38.6, 22),
    fontsize=9,
    color=BLUE,
    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
)
ax.text(36.75, 104, "정상범위", ha="center", fontsize=9, color=MUTED)
ax.text(35.0, 104, "하한 임계", ha="center", fontsize=9, color=MUTED)
ax.text(40.0, 104, "상한 임계", ha="center", fontsize=9, color=MUTED)
ax.set_xlim(34.25, 40.75)
ax.set_ylim(0, 112)
ax.set_xlabel("체온 (℃)")
ax.set_ylabel("항목 점수")
ax.set_title("체온 항목 점수, 원본 vs 재설계")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42))
ax.grid(axis="y")
save(fig, "fig1_boundary_cliff")

kc = pd.read_csv(RES / "k_curves.csv")
marks = pd.read_csv(RES / "k_curve_marks.csv").set_index("k")
fig, ax = plt.subplots(figsize=(8, 3.6))
for k, c in zip((4, 6, 8), (LIGHT, BLUE, DARK)):
    d = kc[kc.k == k]
    ax.plot(d.d, d.score, color=c if k != 4 else GRAY, lw=2, label=f"k = {k}")
for x, lab in ((0.5, "경계와 임계치의 중간"), (1.0, "위험 임계치")):
    ax.axvline(x, color=MUTED, ls=":", lw=1)
    ax.text(x + 0.02, 104, lab, fontsize=9, color=MUTED)
for k, c in zip((4, 6, 8), (GRAY, BLUE, DARK)):
    ax.text(1.02, marks.loc[k, "1.0"] + 1.5, f"{marks.loc[k, '1.0']:.1f}", color=c, fontsize=8)
ax.set_xlim(0, 2)
ax.set_ylim(0, 112)
ax.set_xlabel("정규화 편차 d (0 = 정상 경계, 1 = 위험 임계치)")
ax.set_ylabel("항목 점수")
ax.set_title("재설계 점수 곡선 (k별)")
ax.legend(loc="upper right")
ax.grid(axis="y")
save(fig, "fig2_k_curves")

ct = pd.read_csv(RES / "stage_crosstab.csv")
titles = {
    "current": "원본 점수 단계",
    "current_penalized": "원본 보정(×0.75 / ×0.5) 후",
    "v2": "재설계 점수 단계",
    "v2_capped": "재설계 + 단계 캡",
}
fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, (m, title) in zip(axes, titles.items()):
    sub = ct[ct.method == m].set_index("item_stage").reindex(["안정", "주의", "경고"])
    tot = sub[["score_안정", "score_주의", "score_경고"]].sum(axis=1)
    left = np.zeros(3)
    for st in ("안정", "주의", "경고"):
        share = sub[f"score_{st}"].values / tot.values
        ax.barh(sub.index, share, left=left, color=STAGE_COLOR[st], height=0.6, label=st)
        for i, (v, n) in enumerate(zip(share, sub[f"score_{st}"].values)):
            if v > 0.12:
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{n}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=DARK if st == "안정" else "white",
                )
        left += share
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.grid(axis="x")
    ax.invert_yaxis()
axes[0].set_ylabel("항목 단계 (행)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="점수 단계",
    loc="lower center",
    ncol=3,
    fontsize=9,
    title_fontsize=9,
    bbox_to_anchor=(0.5, -0.13),
)
fig.suptitle(
    "항목 단계 x 점수 단계 (6항목 완비 882명)",
    x=0.01,
    ha="left",
    fontsize=12,
    fontweight="semibold",
)
fig.tight_layout()
save(fig, "fig3_stage_crosstab")

df = pd.read_csv(DATA / "scored_synthetic.csv")
status_cols = [c for c in df.columns if c.startswith("status_")]
n_warn = (df[status_cols] == "경고").sum(axis=1)
n_caut = (df[status_cols] == "주의").sum(axis=1)
co = df[(n_warn == 0) & (n_caut >= 1) & (df.missing_count == 0)]
crossed = (co.score_current >= 50) & (co.score_penalized < 50)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axhspan(0, 50, color=LIGHT, alpha=0.5, lw=0)
ax.axvline(200 / 3, color=MUTED, ls=":", lw=1)
ax.axvline(50, color=MUTED, ls=":", lw=1)
ax.scatter(co.score_current[~crossed], co.score_penalized[~crossed], s=14, color=GRAY, alpha=0.7, label="단계 유지")
ax.scatter(
    co.score_current[crossed],
    co.score_penalized[crossed],
    s=16,
    color=RED,
    alpha=0.85,
    label=f"주의에서 경고로 ({crossed.sum()}명)",
)
ax.plot([0, 100], [0, 75], color=MUTED, lw=0.8, ls="--")
ax.text(67.5, 92, "66.7", fontsize=9, color=MUTED)
ax.text(51, 92, "50", fontsize=9, color=MUTED)
ax.text(2, 44, "보정 후 경고 구간", fontsize=9, color=MUTED)
ax.set_xlim(0, 102)
ax.set_ylim(0, 100)
ax.set_xlabel("원본 점수 (보정 전)")
ax.set_ylabel("보정 점수 (×0.75)")
ax.set_title(f"주의 항목만 있는 {len(co)}명, 보정 전후 점수")
ax.legend(loc="upper left")
ax.grid(axis="y")
save(fig, "fig4_penalty_overjudge")

comp = df[df.missing_count == 0]
fig, ax = plt.subplots(figsize=(8, 3.9))
g = np.random.default_rng(0)
for i, st in enumerate(["안정", "주의", "경고"]):
    sub = comp[comp.item_stage_v2 == st]
    jitter = g.uniform(-0.28, 0.28, len(sub))
    ax.scatter(sub.score_v2, i + jitter, s=9, color=STAGE_COLOR[st], alpha=0.45, lw=0)
    q1, med, q3 = sub.score_v2.quantile([0.25, 0.5, 0.75])
    ax.plot([q1, q3], [i - 0.38, i - 0.38], color=DARK, lw=2.2)
    ax.plot([med, med], [i - 0.46, i - 0.30], color=DARK, lw=2.2)
    ax.text(101, i, f"n={len(sub)}", va="center", fontsize=9, color=MUTED)
for x, lab in ((70, "안정 70"), (50, "주의 50")):
    ax.axvline(x, color=MUTED, ls=":", lw=1)
    ax.text(x - 0.6, 2.62, lab, ha="right", fontsize=9, color=MUTED)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["항목 단계 안정", "항목 단계 주의", "항목 단계 경고"])
ax.set_ylim(-0.6, 2.75)
ax.set_xlim(0, 108)
ax.set_xlabel("재설계 종합 점수 (k = 6)")
ax.set_title("항목 단계별 재설계 종합 점수 분포")
ax.grid(axis="x")
save(fig, "fig5_score_by_item_stage")
