"""그림 1에서 6까지를 사이트 톤으로 다시 그린다.

데이터와 축은 02, 05, 06이 그린 그림과 같고 색, 폰트, 한국어 라벨, 여백만 바뀐다. 예외는 fig4와 fig5로 여기서
재설계했다. fig4는 심어둔 정답 AUC를 기준선으로 둔 dumbbell, fig5는 lollipop 패널 두 개 대신 순위 bump chart다.
모든 값은 data/와 outputs/results/에서 읽고 어떤 숫자도 다시 계산하지 않는다.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import DATA, RES, SEASONS, SERVICE_START, SNAPSHOT
from matplotlib.lines import Line2D
from sitestyle import BLUE, DARK, GRAY, LIGHT, MUTED, ORANGE, save, setup

# feature id의 한국어 라벨. 글의 위젯과 같은 매핑
NAMES = {
    "login_counts": "로그인 횟수 (6개월)",
    "days_since_last_login": "마지막 로그인 경과일",
    "user_cnt": "알림 응답 횟수",
    "total_apply_cnt": "전체 지원 횟수",
    "apply_try_cnt": "지원 시도 횟수",
    "apply_cnt": "지원 완료 횟수",
    "company_cnt": "지원 기업 수",
    "acc_apply_counts": "검사 응시 횟수",
    "test_cnt": "지원 후 검사 횟수",
    "complete_cnt": "전형 완료 횟수",
    "acca_t_score": "검사 점수",
    "acca_grade": "검사 등급",
    "mental_health_grade": "정서 등급",
    "pref_welfare_cnt": "복지 선호 개수",
    "pref_salary_default_yn": "연봉 기본값 여부",
    "marketing_consent_yn": "마케팅 수신 동의",
    "join_year": "가입연도",
    "join_month": "가입월",
    "age": "나이",
    "career_year": "경력 연수",
    "extra": "학교 등급",
    "final_edu_level": "학력",
    "gender": "성별",
    "career_type": "신입/경력",
}
# 대부분 타깃의 결과인 feature. 동의 이후의 활동
AFTER_TARGET = {
    "login_counts",
    "days_since_last_login",
    "user_cnt",
    "total_apply_cnt",
    "apply_try_cnt",
    "apply_cnt",
    "company_cnt",
    "acc_apply_counts",
    "test_cnt",
    "complete_cnt",
}
# 심어둔 동의의 동인
PLANTED = {"acca_t_score", "pref_welfare_cnt", "pref_salary_default_yn", "marketing_consent_yn", "join_month"}


def name(f):
    return NAMES.get(f, f)


setup()
users = pd.read_csv(DATA / "users.csv")

funnel = pd.read_csv(RES / "journey_funnel.csv")
names = ["가입만", "검사만", "프로필만", "검사 + 프로필", "그중 추천 동의 (핵심 유저)"]
vals = funnel["users"].values.astype(float)
total = vals[:4].sum()
fig, ax = plt.subplots(figsize=(8, 3.8))
colors = [GRAY, GRAY, GRAY, BLUE, ORANGE]
ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.62)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 3000, i, f"{v / 1000:,.1f}k  ({v / total:.1%})", va="center", fontsize=9, color=DARK)
ax.set_xlim(0, total * 0.95)
ax.set_xlabel("유저 수")
ax.set_title("가입자 45만 명은 어디에 멈춰 있는가 (조회 시점)")
ax.grid(axis="x")
save(fig, "fig1_journey_funnel")

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

comp = pd.read_csv(RES / "model_comparison.csv")
oracle = pd.read_csv(RES / "oof_auc_vs_oracle.csv").set_index("model")["AUC"]["oracle (true propensity)"]
models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
piv = comp.pivot(index="model", columns="variant", values="AUC").loc[models]
fig, ax = plt.subplots(figsize=(8.5, 3.7))
yy = np.arange(len(models))[::-1]
for i, m in enumerate(models):
    ax.plot(
        [piv.loc[m, "v2_pref"], piv.loc[m, "v1_pref"]],
        [yy[i]] * 2,
        color=LIGHT,
        lw=3.5,
        zorder=1,
        solid_capstyle="round",
    )
    ax.text(
        piv.loc[m, "v2_pref"] - 0.008,
        yy[i],
        f"{piv.loc[m, 'v2_pref']:.3f}",
        ha="right",
        va="center",
        fontsize=9,
        color=DARK,
    )
    ax.text(
        piv.loc[m, "v1_pref"] + 0.008,
        yy[i],
        f"{piv.loc[m, 'v1_pref']:.3f}",
        ha="left",
        va="center",
        fontsize=9,
        color=DARK,
    )
ax.scatter(piv["v1_asis"], yy, s=75, color=GRAY, zorder=3, label="v1 스냅샷")
ax.scatter(
    piv["v1_pref"], yy, s=75, facecolor="white", edgecolor=MUTED, linewidth=1.8, zorder=4, label="v1 스냅샷 + 선호 정보"
)
ax.scatter(piv["v2_pref"], yy, s=75, color=BLUE, zorder=3, label="v2 시간 절단 + 선호 정보")
ax.axvline(oracle, color=ORANGE, ls="--", lw=1.2, zorder=2)
ax.text(oracle + 0.004, yy.max() + 0.62, f"심어둔 정답 확률의 AUC {oracle:.3f}", color=ORANGE, fontsize=9, va="center")
ax.set_yticks(yy, models)
ax.set_ylim(-0.7, len(models) - 0.1)
ax.set_xlim(0.6, 1.0)
ax.set_xlabel("10-fold CV AUC")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.grid(axis="x")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3)
ax.set_title("feature 구성별 AUC — 스냅샷은 정답보다 잘 맞힌다")
save(fig, "fig4_auc_by_variant")

imp1 = pd.read_csv(RES / "importance_v1_asis.csv", index_col=0)["importance"]
imp2 = pd.read_csv(RES / "importance_v2_pref.csv", index_col=0)["importance"]
TOP_N = 8
feats = list(dict.fromkeys(list(imp1.index[:TOP_N]) + list(imp2.index[:TOP_N])))
rank1 = {f: i + 1 for i, f in enumerate(imp1.index)}
rank2 = {f: i + 1 for i, f in enumerate(imp2.index)}
SHOWN = 13  # 실제 순위는 1에서 13까지 그리고, 그 밖은 "14위 밖" 행, feature에 없으면 "없음" 행


def positions(rank):
    """한쪽의 feature별 y 위치: 실제 순위, 그다음 순위 밖 / 없음 행."""
    pos, over, absent = {}, [], []
    for f in feats:
        if f not in rank:
            absent.append(f)
        elif rank[f] > SHOWN:
            over.append(f)
        else:
            pos[f] = rank[f]
    for i, f in enumerate(sorted(over, key=rank.get)):
        pos[f] = SHOWN + 1 + i
    for i, f in enumerate(absent):
        pos[f] = SHOWN + 1 + len(over) + i
    return pos, len(over), len(absent)


pos1, over1, abs1 = positions(rank1)
pos2, over2, abs2 = positions(rank2)
n_over = max(over1, over2, 1)
n_abs = max(abs1, abs2)
# 두 축이 행을 공유하므로 없음 행은 더 넓은 쪽의 순위 밖 구역 뒤로 민다
for pos, n_o in ((pos1, over1), (pos2, over2)):
    for f, r in pos.items():
        if r > SHOWN + n_o:
            pos[f] = r + (n_over - n_o)
ylabels = (
    [str(i) for i in range(1, SHOWN + 1)]
    + ["14위 밖"]
    + [""] * (n_over - 1)
    + (["없음"] + [""] * (n_abs - 1) if n_abs else [])
)
NROWS = len(ylabels)
fig, ax = plt.subplots(figsize=(10, 6.6))
for f in feats:
    c = ORANGE if f in AFTER_TARGET else BLUE if f in PLANTED else GRAY
    r1, r2 = pos1[f], pos2[f]
    ax.plot([0, 1], [r1, r2], color=c, lw=2.2, alpha=0.9, zorder=2)
    ax.plot([0, 1], [r1, r2], "o", color=c, ms=7, zorder=3)
    left = f"{name(f)}  {imp1[f]:.2f}" if f in rank1 else f"{name(f)}  (feature에 없음)"
    ax.text(-0.04, r1, left, ha="right", va="center", fontsize=9.5, color=DARK)
    ax.text(1.04, r2, f"{imp2[f]:.2f}  {name(f)}", ha="left", va="center", fontsize=9.5, color=DARK)
ax.axhspan(SHOWN + 0.5, NROWS + 0.5, color=LIGHT, alpha=0.35, lw=0)
ax.set_xlim(-0.75, 1.75)
ax.set_ylim(NROWS + 0.6, 0.3)
ax.set_yticks(range(1, NROWS + 1), ylabels)
ax.set_ylabel("중요도 순위 (LightGBM gain)")
ax.set_xticks([0, 1], ["v1 스냅샷 (당시 구조)", "v2 시간 절단 (+ 선호 정보)"])
ax.tick_params(axis="x", labelsize=10.5, length=0)
ax.tick_params(axis="y", length=0)
for s in ("left", "bottom"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y")
handles = [
    Line2D([], [], color=ORANGE, lw=2.2, label="타깃 결정 이후의 행동"),
    Line2D([], [], color=BLUE, lw=2.2, label="심어둔 전환 신호"),
    Line2D([], [], color=GRAY, lw=2.2, label="그 외"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncols=3)
ax.set_title("feature importance 순위 — 시간을 자르면 무엇이 올라오고 무엇이 내려가나")
save(fig, "fig5_importance_v1_vs_v2")

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
fig.suptitle(
    f"넛지 리스트: 비동의 유저 상위 10% — 두 리스트의 겹침 {overlap:.0%}",
    x=0.02,
    ha="left",
    fontsize=12,
    fontweight="semibold",
    color=DARK,
)
fig.tight_layout()
save(fig, "fig6_nudge_lists")
print("font:", plt.rcParams["font.family"])
