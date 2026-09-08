"""08 — Redraw the six figures in the site's visual tone (Pretendard, site palette, webp).

Data and axes are identical to the figures drawn by 02 / 05 / 06; only colors, font, labels
(Korean) and margins change. Everything is read back from data/ and outputs/results/, so this
script never recomputes a number. Output: outputs/figures/site/figN_*.webp (+ .png).
"""
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FIG, RES, SERVICE_START, SNAPSHOT, SEASONS

OUT = FIG / "site"
OUT.mkdir(parents=True, exist_ok=True)

# --- site tone
for f in list(Path.home().glob("Library/Fonts/Pretendard-*.otf")) + list(Path("/Library/Fonts").glob("Pretendard-*.otf")):
    font_manager.fontManager.addfont(str(f))
have_pretendard = any("Pretendard" in f.name for f in font_manager.fontManager.ttflist)
plt.rcParams.update({
    "font.family": "Pretendard" if have_pretendard else "AppleGothic",
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#a1a1aa", "axes.labelcolor": "#3f3f46",
    "xtick.color": "#52525b", "ytick.color": "#52525b",
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlecolor": "#18181b", "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False,
    "grid.color": "#e4e4e7", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
BLUE, ORANGE, GRAY, LIGHT, DARK = "#2563eb", "#ea580c", "#a1a1aa", "#e4e4e7", "#18181b"


def save(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.save(OUT / f"{name}.webp", "WEBP", quality=88, method=6)
    img.save(OUT / f"{name}.png")
    plt.close(fig)
    print(f"saved {name}.webp  {img.size[0]}x{img.size[1]}")


users = pd.read_csv(DATA / "users.csv")

# ---------------------------------------------------------------- fig1 funnel (02)
funnel = pd.read_csv(RES / "journey_funnel.csv")
names = ["가입만", "검사만", "프로필만", "검사 + 프로필", "그중 추천 동의 (핵심 유저)"]
vals = funnel["users"].values.astype(float)
total = vals[:4].sum()
fig, ax = plt.subplots(figsize=(8, 3.8))
colors = [GRAY, GRAY, GRAY, BLUE, ORANGE]
ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.62)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 3000, i, f"{v/1000:,.1f}k  ({v/total:.1%})", va="center", fontsize=9, color=DARK)
ax.set_xlim(0, total * 0.95)
ax.set_xlabel("유저 수")
ax.set_title("가입자 45만 명은 어디에 멈춰 있는가 (조회 시점)")
ax.grid(axis="x")
save(fig, "fig1_journey_funnel")

# ---------------------------------------------------------------- fig2 preference fields (02)
by_consent = pd.read_csv(RES / "journey_preference_by_consent.csv").set_index("matching_use_yn")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
h = users["pref_welfare_cnt"].value_counts().sort_index()
ax.bar(h.index, h.values / h.sum(), color=BLUE, width=0.72)
ax.axvline(4.5, color=ORANGE, ls="--", lw=1)
ax.text(4.7, ax.get_ylim()[1] * 0.9, f"5개 미만: {(users['pref_welfare_cnt'] < 5).mean():.0%}", color=ORANGE)
ax.set_xlabel("선택한 복지 선호 항목 수")
ax.set_ylabel("유저 비율")
ax.set_title("비어 있는 건 선호 정보다")
ax = axes[1]
x = np.arange(2)
w = 0.35
ax.bar(x - w / 2, by_consent["salary_default"], w, label="연봉 기본값 그대로", color=GRAY)
ax.bar(x + w / 2, by_consent["pref_complete"], w, label="선호 정보 완성", color=BLUE)
ax.set_xticks(x, ["비동의", "동의"])
ax.set_ylim(0, 1)
ax.set_ylabel("비율")
ax.legend()
ax.set_title("…그리고 두 집단을 가른다")
for a in axes:
    a.grid(axis="y")
fig.tight_layout()
save(fig, "fig2_preference_fields")

# ---------------------------------------------------------------- fig3 weekly logins (02)
logins = np.load(DATA / "logins.npz")["logins"]
days = pd.date_range(SERVICE_START, SNAPSHOT, freq="D")
weekly = pd.Series(logins.sum(0), index=days).resample("W").sum()[:-1]
pos = users["matching_use_yn"].values == 1
weekly_pos = pd.Series(logins[pos].sum(0), index=days).resample("W").sum()[:-1]
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(weekly.index, weekly.values, color=BLUE, lw=1.6, label="검사+프로필 유저 전체")
ax.plot(weekly_pos.index, weekly_pos.values, color=ORANGE, lw=1.3, label="동의한 유저")
for a, b in SEASONS:
    ax.axvspan(a, b, color=ORANGE, alpha=0.12, lw=0)
ax.text(SEASONS[1][0], ax.get_ylim()[1] * 0.95, "공채 시즌", color=ORANGE, fontsize=9, va="top")
ax.set_ylabel("주간 로그인 수")
ax.set_title("재방문은 공채 시즌에, 그리고 동의한 유저에게서 온다")
ax.legend(loc="upper left")
ax.grid(axis="y")
fig.tight_layout()
save(fig, "fig3_weekly_logins")

# ---------------------------------------------------------------- fig4 AUC by variant (05)
comp = pd.read_csv(RES / "model_comparison.csv")
models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
piv = comp.pivot(index="model", columns="variant", values="AUC").loc[models, ["v1_asis", "v1_pref", "v2_pref"]]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(models))
w = 0.25
for i, (col, lab, c) in enumerate([("v1_asis", "v1 스냅샷", GRAY), ("v1_pref", "v1 스냅샷 + 선호 정보", BLUE),
                                   ("v2_pref", "v2 시간 절단 + 선호 정보", ORANGE)]):
    bars = ax.bar(x + (i - 1) * w, piv[col].values, w, label=lab, color=c)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2, color=DARK)
ax.set_xticks(x, models)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("10-fold CV AUC")
ax.grid(axis="y")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3)
ax.set_title("피처 구성별 AUC — 스냅샷은 모든 모델을 부풀린다")
fig.tight_layout()
save(fig, "fig4_auc_by_variant")

# ---------------------------------------------------------------- fig5 importance v1 vs v2 (05)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, vname, title in zip(axes, ["v1_asis", "v2_pref"], ["v1 — 스냅샷 (당시 구조)", "v2 — 시간 절단 (+ 선호 정보)"]):
    imp = pd.read_csv(RES / f"importance_{vname}.csv", index_col=0)["importance"]
    top = imp.head(10)[::-1]
    ax.hlines(top.index, 0, top.values, color=BLUE, lw=1.5)
    ax.plot(top.values, top.index, "o", color=BLUE)
    ax.set_title(title)
    ax.set_xlabel("정규화 gain 중요도 (LightGBM)")
    ax.grid(axis="x")
    ax.set_xlim(0, max(0.45, top.values.max() * 1.15))
    ax.tick_params(axis="y", labelsize=9)
fig.suptitle("피처 중요도 상위 10 — 모델은 무엇을 배웠나", fontsize=12, x=0.02, ha="left", fontweight="semibold", color=DARK)
fig.tight_layout()
save(fig, "fig5_importance_v1_vs_v2")

# ---------------------------------------------------------------- fig6 nudge lists (06)
out = pd.read_csv(RES / "nudge_list_comparison.csv")
extra = pd.read_csv(RES / "nudge_list_overlap.csv").set_index("metric")["value"]
overlap = extra["overlap between v1 and v2 top-10% lists"]
rand_p = extra["random 10% list: mean true consent propensity"]
metrics = ["logged in ≤ 30d (snapshot)", "preference complete", "salary left at default", "season joiner"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(metrics))
w = 0.26
for i, (lab, col) in enumerate([("v1 리스트", GRAY), ("v2 리스트", BLUE), ("비동의 전체", LIGHT)]):
    axes[0].bar(x + (i - 1) * w, out.iloc[i][metrics].values.astype(float), w, label=lab, color=col)
axes[0].set_xticks(x, ["최근 30일\n로그인", "선호 정보\n완성", "연봉\n기본값", "시즌\n가입자"], fontsize=9)
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].set_title("리스트에 누가 있나")
axes[0].grid(axis="y")
vals = [rand_p, out.iloc[0]["mean true consent propensity"], out.iloc[1]["mean true consent propensity"]]
axes[1].bar(["무작위 10%", "v1 리스트", "v2 리스트"], vals, color=[LIGHT, GRAY, BLUE], width=0.6)
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, color=DARK)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("심어둔 동의 확률의 평균")
axes[1].set_title("정답 확률로 채점하면")
axes[1].grid(axis="y")
fig.suptitle(f"넛지 리스트: 비동의 유저 상위 10% — 두 리스트의 겹침 {overlap:.0%}", x=0.02, ha="left",
             fontsize=12, fontweight="semibold", color=DARK)
fig.tight_layout()
save(fig, "fig6_nudge_lists")
print("font:", plt.rcParams["font.family"])
