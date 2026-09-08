"""10 — Site figures 7–11, drawn from 09's CSVs (never recomputes a number).

  fig7_login_around_consent   daily login rate around the (pseudo) consent day
  fig8_score_plane            non-consented users on the v1 × v2 score plane, colored by planted truth
  fig9_timeline_sample        12 users: events, logins, the v2 cutoff
  fig10_roc_curves            v1 / v2 / planted truth
  fig11_score_deciles         non-consented users by score decile → planted propensity
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES
from sitestyle import setup, save, BLUE, ORANGE, GRAY, LIGHT, DARK, MUTED
GRAY_LINE = MUTED  # #a1a1aa fails 3:1 on white for thin lines; bars keep GRAY

setup()
summary = pd.read_csv(RES / "leakage_anatomy_summary.csv").set_index("metric")["value"]

# ---------------------------------------------------------------- fig7 login rate around the anchor
ev = pd.read_csv(RES / "login_around_consent.csv")
fig, ax = plt.subplots(figsize=(10, 3.9))
ax.axvspan(-0.5, ev["rel_day"].max(), color=LIGHT, alpha=0.45, lw=0)
ax.axvline(-0.5, color=DARK, lw=0.8, ls=":")
ax.plot(ev["rel_day"], ev["consented_rate"], color=ORANGE, lw=1.9, label="동의 유저 (기준일 = 동의일)")
ax.plot(ev["rel_day"], ev["not_consented_rate"], color=GRAY_LINE, lw=1.9, label="비동의 유저 (기준일 = v2 절단 다음 날)")
top = ev[["consented_rate", "not_consented_rate"]].values.max()
d1 = float(ev.loc[ev["rel_day"] == 1, "consented_rate"].iloc[0])
ax.annotate(f"+1일: {d1:.2f}\n결과표 확인 로그인", xy=(1, d1), xytext=(14, d1 * 0.98), fontsize=9, color=ORANGE,
            va="top", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.text(-29, top * 1.02, "v2가 보는 구간", color=MUTED, fontsize=9, va="bottom")
ax.text(29, top * 1.02, "v2 절단 이후 — v1 스냅샷만 보는 구간", color=MUTED, fontsize=9, va="bottom", ha="left")
ax.set_xlim(-30, 60)
ax.set_ylim(0, top * 1.12)
ax.set_xlabel("기준일로부터 일수")
ax.set_ylabel("일별 로그인 확률")
ax.set_title("동의 다음 날 로그인이 튀고, 그 뒤로 내려오지 않는다")
ax.legend(loc="center right")
ax.grid(axis="y")
save(fig, "fig7_login_around_consent")

# ---------------------------------------------------------------- fig8 v1 × v2 score plane (non-consented users)
sc = pd.read_csv(RES / "oof_scores.csv")
neg = sc[sc["matching_use_yn"] == 0]
quad = pd.read_csv(RES / "score_quadrants.csv").set_index("quadrant")
t1 = float(summary["quadrants: v1 list threshold (min OOF score)"])
t2 = float(summary["quadrants: v2 list threshold (min OOF score)"])
k = int(float(summary["quadrants: list size k"]))
BOX = dict(facecolor="white", alpha=0.88, edgecolor="none", pad=3)
fig, ax = plt.subplots(figsize=(7.2, 6.2))
hb = ax.hexbin(neg["score_v1"], neg["score_v2"], C=neg["truth_p_consent"], gridsize=30, cmap="Blues",
               vmin=0.5, vmax=1.0, mincnt=3, linewidths=0.3, edgecolors="white")
cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("칸 안 유저의 심어둔 동의 확률 (평균, 3명 이상)")
cb.outline.set_visible(False)
ax.axvline(t1, color=ORANGE, ls="--", lw=1.2)
ax.axhline(t2, color=ORANGE, ls="--", lw=1.2)
def q(name):
    r = quad.loc[name]
    return f"{int(r['n']):,}명 · 정답 확률 {r['mean_truth_p']:.2f}"
ax.text(0.985, 0.985, "두 리스트 모두\n" + q("both lists"), ha="right", va="top", transform=ax.transAxes, fontsize=9, color=DARK, bbox=BOX)
ax.text(0.985, 0.03, "v1 리스트만\n" + q("v1 list only"), ha="right", va="bottom", transform=ax.transAxes, fontsize=9, color=DARK, bbox=BOX)
ax.text(0.02, 0.985, "v2 리스트만\n" + q("v2 list only"), ha="left", va="top", transform=ax.transAxes, fontsize=9, color=DARK, bbox=BOX)
ax.text(0.02, 0.03, "어느 쪽도 아님\n" + q("neither"), ha="left", va="bottom", transform=ax.transAxes, fontsize=9, color=MUTED, bbox=BOX)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(f"v1 스냅샷 점수 (OOF) — 점선 오른쪽이 상위 {k}명")
ax.set_ylabel(f"v2 시간 절단 점수 (OOF) — 점선 위쪽이 상위 {k}명")
ax.grid()
ax.set_title("비동의 유저 7,457명의 두 점수 — 겹침 28%, 색은 정답")
save(fig, "fig8_score_plane")

# ---------------------------------------------------------------- fig9 timeline sample
tl = pd.read_csv(RES / "timeline_sample.csv")
XMAX = 365
fig, ax = plt.subplots(figsize=(10, 5.4))
for i, r in tl.iterrows():
    yy = len(tl) - 1 - i
    ld = np.array([int(x) for x in str(r["login_days_since_join"]).split(";") if x != ""])
    cut = r["v2_cutoff_day"] + 0.5
    ax.fill_between([cut, XMAX], yy - 0.42, yy + 0.42, color=LIGHT, alpha=0.55, lw=0)
    ax.vlines(ld, yy - 0.26, yy + 0.26, color=DARK, lw=0.8, alpha=0.85)
    ax.plot(r["test_day"], yy, marker="^", color=GRAY, ms=6.5, ls="none", zorder=3)
    ax.plot(r["profile_day"], yy, marker="s", color=GRAY, ms=5.5, ls="none", zorder=3)
    if r["consented"] == 1:
        ax.plot(r["consent_day"], yy, marker="o", color=ORANGE, ms=8, ls="none", zorder=4)
    ax.vlines(cut, yy - 0.46, yy + 0.46, color=BLUE, lw=2, zorder=5)
labels = [("동의" if r["consented"] else "비동의") + f"  #{r['user']}" for _, r in tl.iterrows()]
ax.set_yticks(range(len(tl) - 1, -1, -1), labels)
ax.set_xlim(-3, XMAX)
ax.set_ylim(-0.6, len(tl) - 0.4)
ax.set_xlabel("가입 후 일수")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(axis="x")
handles = [Line2D([], [], color=DARK, lw=0.8, label="로그인"),
           Line2D([], [], marker="^", color=GRAY, ls="none", ms=6.5, label="검사 응시"),
           Line2D([], [], marker="s", color=GRAY, ls="none", ms=5.5, label="프로필 완성"),
           Line2D([], [], marker="o", color=ORANGE, ls="none", ms=8, label="추천 동의 (타깃)"),
           Line2D([], [], color=BLUE, lw=2, label="v2 절단 시점"),
           Patch(color=LIGHT, label="v1 스냅샷만 보는 구간")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=6)
ax.set_title("유저 12명의 첫 1년 — v2는 파란 선 왼쪽만 본다")
save(fig, "fig9_timeline_sample")

# ---------------------------------------------------------------- fig10 ROC
roc = pd.read_csv(RES / "roc_curves.csv")
auc = pd.read_csv(RES / "oof_auc_vs_oracle.csv").set_index("model")["AUC"]
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.plot([0, 1], [0, 1], color=LIGHT, lw=1)
for label, c, ls, kor in [("v1 snapshot", GRAY_LINE, "-", "v1 스냅샷"),
                          ("v2 time cut", BLUE, "-", "v2 시간 절단"),
                          ("oracle (true propensity)", ORANGE, "--", "심어둔 정답 확률")]:
    d = roc[roc["model"] == label]
    ax.plot(d["fpr"], d["tpr"], color=c, ls=ls, lw=1.9, label=f"{kor}   AUC {auc[label]:.3f}")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("위양성률 (FPR)")
ax.set_ylabel("재현율 (TPR)")
ax.set_title("ROC — v1은 정답보다 위에 있다")
ax.legend(loc="lower right")
ax.grid()
save(fig, "fig10_roc_curves")

# ---------------------------------------------------------------- fig11 score deciles
dc = pd.read_csv(RES / "score_deciles.csv")
base = float(summary["deciles: non-consented mean truth propensity"])
n_neg = int(float(summary["deciles: non-consented users"]))
fig, ax = plt.subplots(figsize=(8.5, 3.9))
ax.axhline(base, color=ORANGE, ls="--", lw=1.1)
ax.text(10.55, base + 0.004, f"비동의 전체 평균\n{base:.3f}", color=ORANGE, fontsize=9, va="bottom")
for vname, c, lab in [("v1", GRAY_LINE, "v1 스냅샷 점수"), ("v2", BLUE, "v2 시간 절단 점수")]:
    d = dc[dc["score"] == vname]
    ax.plot(d["decile"], d["mean_truth_p"], marker="o", ms=6, color=c, lw=1.9, label=lab)
    ax.text(0.75, d["mean_truth_p"].iloc[0], f"{d['mean_truth_p'].iloc[0]:.3f}", color=c, ha="right", va="center", fontsize=9)
ax.set_xticks(range(1, 11), ["상위 10%"] + [f"{i}" for i in range(2, 10)] + ["하위 10%"])
ax.set_xlim(0.2, 12.4)
ax.set_xlabel(f"모델 점수 분위 (비동의 유저 {n_neg:,}명)")
ax.set_ylabel("심어둔 동의 확률의 평균")
ax.set_title("점수 분위별 정답 확률 — 상위 분위에서 v2가 앞선다")
ax.legend(loc="lower left")
ax.grid(axis="y")
save(fig, "fig11_score_deciles")
