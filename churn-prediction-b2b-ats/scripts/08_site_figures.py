"""
08. 그림 8장. 데이터·축은 results CSV 그대로, 색·폰트·여백만 조정.
출력: outputs/figures/fig1~fig8 png
"""
import numpy as np, pandas as pd
from _common import *
plt = setup_mpl()
R = lambda f: pd.read_csv(RESULTS / f)

# fig1 결합 탈락 편향
bb = R("join_bias_by_size.csv"); s = R("join_bias_summary.csv").set_index("metric").value
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(bb.size_band, bb.kept_rate, color=GRAY)
for i, v in enumerate(bb.kept_rate): ax[0].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
ax[0].set_ylim(0, 1.08); ax[0].set_title("외부데이터 결합 후 남은 비율 (재직인원 구간별)"); ax[0].set_ylabel("결합 성공률")
x = np.arange(len(bb)); w = 0.38
ax[1].bar(x - w/2, bb.churn_all, w, color=GRAY, label="결합 전 전체")
ax[1].bar(x + w/2, bb.churn_kept, w, color=BLUE, label="결합 후 남은 회사")
ax[1].set_xticks(x); ax[1].set_xticklabels(bb.size_band); ax[1].set_ylim(0, 0.75)
ax[1].axhline(s.churn_rate_all, color=GRAY, ls="--", lw=1); ax[1].axhline(s.churn_rate_kept, color=BLUE, ls="--", lw=1)
ax[1].text(4.45, s.churn_rate_all + 0.015, f"전체 {s.churn_rate_all:.1%}", color=GRAY, fontsize=8, ha="right")
ax[1].text(4.45, s.churn_rate_kept - 0.045, f"남은 회사 {s.churn_rate_kept:.1%}", color=BLUE, fontsize=8, ha="right")
ax[1].set_title("이탈률: 탈락 전후"); ax[1].legend(fontsize=8, loc="upper right")
fig.savefig(FIGURES / "fig1_join_bias.png"); plt.close(fig)

# fig2 최초계약연도별 이탈률·갱신 기회 횟수
by = R("churn_by_first_year.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(by.first_year.astype(str), by.churn_rate, color=GRAY)
for i, (v, n) in enumerate(zip(by.churn_rate, by.n)): ax.text(i, v + 0.02, f"{v:.0%}\n(n={n})", ha="center", fontsize=8)
ax.set_ylim(0, 1.0); ax.set_ylabel("이탈률 (2024-02 기준)"); ax.set_xlabel("최초 계약 연도")
ax2 = ax.twinx(); ax2.plot(by.first_year.astype(str), by.mean_renewal_opportunities, color=ORANGE, marker="o")
ax2.set_ylabel("평균 갱신 결정 횟수", color=ORANGE); ax2.grid(False); ax2.spines["right"].set_visible(True)
ax.set_title("최초 계약이 오래될수록 이탈률이 높다 — 갱신 결정을 그만큼 여러 번 했기 때문이다")
fig.savefig(FIGURES / "fig2_churn_by_first_year.png"); plt.close(fig)

# fig3 미래 매출 열의 라벨 분리
fl = R("future_revenue_leak.csv"); fl = fl[fl.feature.isin(["rev_2015", "rev_2024", "rev_2025", "n_contracts", "total_revenue", "gubun_신규"])]
lab = {"rev_2015": "2015년 매출 > 0", "rev_2024": "2024년 매출 > 0", "rev_2025": "2025년 매출 > 0",
       "n_contracts": "총계약수 > 중앙값", "total_revenue": "총매출 > 중앙값", "gubun_신규": "구분 = 신규 (재계약 이력 없음)"}
fig, ax = plt.subplots(figsize=(8, 3.8))
y = np.arange(len(fl))
ax.barh(y + 0.2, fl.churn_rate_if_zero, 0.4, color=GRAY, label="조건 불충족")
ax.barh(y - 0.2, fl.churn_rate_if_positive, 0.4, color=RED, label="조건 충족")
for i, (a, b, n) in enumerate(zip(fl.churn_rate_if_positive, fl.churn_rate_if_zero, fl.n_positive)):
    ax.text(a + 0.01, i - 0.2, f"{a:.0%} (n={n})", va="center", fontsize=8, color=RED)
    ax.text(b + 0.01, i + 0.2, f"{b:.0%}", va="center", fontsize=8, color=GRAY)
ax.set_yticks(y); ax.set_yticklabels([lab[f] for f in fl.feature]); ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.set_xlabel("이탈률"); ax.legend(fontsize=8, loc="lower right"); ax.set_title("feature로 남은 열 하나가 라벨을 얼마나 가르는가")
fig.savefig(FIGURES / "fig3_future_revenue_leak.png"); plt.close(fig)

# fig4 AUC 비교
mc = R("model_compare.csv"); cls = mc[mc.task == "classification"]
order = ["v1 snapshot", "v1 minus future/cumulative", "v1 snapshot (688)", "v2 timecut (688)"]
colors = {"v1 snapshot": RED, "v1 minus future/cumulative": ORANGE, "v1 snapshot (688)": RED, "v2 timecut (688)": BLUE}
fig, ax = plt.subplots(figsize=(8.5, 3.8))
markers = {"Logistic Regression": "s", "Random Forest": "^", "XGBoost": "D", "LightGBM": "o"}
for i, v in enumerate(order):
    d = cls[cls.variant == v]
    for _, r in d.iterrows():
        ax.scatter(i, r.auc, color=colors[v], marker=markers[r.model], s=48, zorder=3, label=r.model if i == 0 else None)
    ax.text(i + 0.1, d.auc.max() + 0.006, f"{d.auc.min():.3f}–{d.auc.max():.3f}", fontsize=7, color=colors[v])
ax.legend(fontsize=7, loc="lower left", title="모델", title_fontsize=7)
o688 = mc[(mc.variant == "oracle (688)")].auc.iloc[0]; o1141 = mc[mc.model == "truth_churn_p (exposure-aware)"].auc.iloc[0]
reg = mc[mc.task == "regression"].auc.iloc[0]
ax.hlines(o1141, -0.3, 1.3, color=GREEN, ls="--"); ax.text(1.3, o1141 + 0.004, f"oracle(1,141) {o1141:.3f}", color=GREEN, fontsize=8, ha="right")
ax.hlines(o688, 1.7, 3.3, color=GREEN, ls="--"); ax.text(3.3, o688 + 0.004, f"oracle(688) {o688:.3f}", color=GREEN, fontsize=8, ha="right")
ax.scatter([0], [reg], marker="x", color=RED, s=60, zorder=4); ax.text(-0.08, reg, "RF 회귀(원본)", fontsize=7, ha="right", va="center")
ax.set_xticks(range(4)); ax.set_xticklabels(["v1 스냅샷\n(1,141)", "v1 − 미래·누적 열\n(1,141)", "v1 스냅샷\n(688)", "v2 시간 절단\n(688)"])
ax.set_ylim(0.80, 1.0); ax.set_ylabel("AUC (10-fold CV)"); ax.set_title("정답보다 높은 AUC는 축하할 일이 아니다")
fig.savefig(FIGURES / "fig4_auc_compare.png"); plt.close(fig)

# fig5 SHAP beeswarm v1 vs v2 (상위 10)
def beeswarm(ax, tag, title, X):
    sv = pd.read_csv(DATA / f"shap_values_{tag}.csv"); imp = R(f"shap_importance_{tag}.csv").head(10)
    r = rng(99)
    for i, f in enumerate(imp.feature[::-1]):
        vals = sv[f].values; xv = X[f].values
        rk = (pd.Series(xv).rank(pct=True)).values
        ax.scatter(vals, i + r.normal(0, 0.08, len(vals)), c=rk, cmap="coolwarm", s=5, alpha=0.6, vmin=0, vmax=1)
    ax.set_yticks(range(10)); ax.set_yticklabels(imp.feature[::-1], fontsize=8); ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("SHAP value (이탈 방향 +)"); ax.set_title(title)
v1 = pd.read_csv(DATA / "features_v1.csv"); v2 = pd.read_csv(DATA / "features_v2.csv")
tr = v1.sample(frac=0.95, random_state=786).reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
beeswarm(ax[0], "v1", "v1 스냅샷 — LightGBM OOF SHAP", tr); beeswarm(ax[1], "v2", "v2 시간 절단 — LightGBM OOF SHAP", v2)
fig.text(0.5, -0.02, "점 색: feature 값의 백분위 (파랑 낮음 → 빨강 높음)", ha="center", fontsize=8, color=GRAY)
fig.savefig(FIGURES / "fig5_shap_v1_vs_v2.png"); plt.close(fig)

# fig6 기준일 스윕 (위 이탈률, 아래 라벨 뒤집힘 비율)
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
sw = R("reference_date_sweep.csv"); sw["ref_date"] = pd.to_datetime(sw.ref_date)
live = sw[sw["mode"] == "live"].set_index("ref_date"); frz = sw[sw["mode"] == "frozen"].set_index("ref_date")
post = live.index >= REF_DATE
fig, (a0, a1) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
for a in (a0, a1):
    a.grid(axis="x", visible=False); a.spines["left"].set_color("#cfd3d9"); a.spines["bottom"].set_color("#cfd3d9")
    a.axvspan(live.index.min(), REF_DATE, color="#000", alpha=0.035, lw=0)
    a.axvline(REF_DATE, color="#555", lw=0.9)
    a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); a.tick_params(length=0, labelsize=9)
a0.fill_between(live.index[post], live.churn_rate[post], frz.churn_rate[post], color=RED, alpha=0.08, lw=0)
a0.plot(live.index, frz.churn_rate, color=RED, lw=2.2); a0.plot(live.index, live.churn_rate, color=BLUE, lw=2.2)
a0.scatter([REF_DATE], [live.churn_rate[REF_DATE]], color="#222", s=28, zorder=5)
a0.annotate(f"원본 실행일\n{live.churn_rate[REF_DATE]:.1%}", (REF_DATE, live.churn_rate[REF_DATE]), xytext=(-12, 18),
            textcoords="offset points", fontsize=8.5, ha="right", va="bottom", color="#222")
for months, dy in [(6, -10), (12, -10), (24, -10)]:
    d = REF_DATE + pd.DateOffset(months=months)
    for ser, col in [(frz, RED), (live, BLUE)]:
        a0.scatter([d], [ser.churn_rate[d]], color=col, s=18, zorder=5)
        a0.text(d, ser.churn_rate[d] + (0.028 if col == RED else -0.05), f"{ser.churn_rate[d]:.0%}", fontsize=8, ha="center", color=col)
    a0.text(d, 0.315, f"+{months}개월", fontsize=8, ha="center", color="#777")
a0.text(live.index.max() + pd.Timedelta(days=12), frz.churn_rate.iloc[-1], "스냅샷 고정\n(데이터 갱신 없이 재실행)", fontsize=8.5, color=RED, va="center")
a0.text(live.index.max() + pd.Timedelta(days=12), live.churn_rate.iloc[-1], "데이터 갱신\n(새 계약 반영)", fontsize=8.5, color=BLUE, va="center")
a0.text(live.index.min() + pd.Timedelta(days=10), 0.97, "원본 실행일 이전", fontsize=8, color="#777", va="top")
a0.set_ylim(0.3, 1.0); a0.set_ylabel("이탈률 (1,201사)", fontsize=9)
a1.plot(live.index, frz.flip_rate, color=RED, lw=1.8); a1.plot(live.index, live.flip_rate, color=BLUE, lw=1.8)
a1.fill_between(live.index[post], live.flip_rate[post], frz.flip_rate[post], color=RED, alpha=0.08, lw=0)
a1.set_ylim(0, 0.62); a1.set_ylabel("원본 라벨과\n다른 회사", fontsize=9)
a1.text(live.index.max() + pd.Timedelta(days=12), frz.flip_rate.iloc[-1], f"{frz.flip_rate.iloc[-1]:.0%}", fontsize=8.5, color=RED, va="center")
a1.text(live.index.max() + pd.Timedelta(days=12), live.flip_rate.iloc[-1], f"{live.flip_rate.iloc[-1]:.0%}", fontsize=8.5, color=BLUE, va="center")
a1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 8])); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
a1.set_xlabel("노트북 실행일 (라벨의 today)", fontsize=9)
fig.suptitle("같은 회사, 같은 규칙 — 실행일만 옮겼을 때의 이탈률", x=0.125, ha="left", fontsize=11.5, y=0.955)
fig.text(0.125, 0.905, "최종 종료일 + 90일 초과 → 이탈. 원본 실행일(2024-02) 이후 달라진 라벨은 전부 고객 → 이탈 방향.", fontsize=8.5, color="#777")
fig.subplots_adjust(right=0.82, top=0.86)
fig.savefig(FIGURES / "fig6_reference_date_sweep.png"); plt.close(fig)

# fig7 KM + Cox
import matplotlib.transforms as mtrans
km = R("km_by_product.csv"); cx = R("cox_summary.csv")
fig = plt.figure(figsize=(12.5, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[5, 1.1], wspace=0.48, hspace=0.05)
ax0 = fig.add_subplot(gs[0, 0]); axt = fig.add_subplot(gs[1, 0], sharex=ax0); ax1 = fig.add_subplot(gs[:, 1])
for a in (ax0, ax1):
    a.grid(axis="x", visible=False); a.spines["left"].set_color("#cfd3d9"); a.spines["bottom"].set_color("#cfd3d9"); a.tick_params(length=0, labelsize=9)
XMAX = 60
series = [("both", BLUE, "상품 A+B"), ("diag", "#5b6470", "상품 A만"), ("posting", ORANGE, "상품 B만")]
ends = {}
for prod, col, lab_ in series:
    d = km[(km["product"] == prod) & (km.month <= XMAX)]
    ax0.fill_between(d.month, d.ci_lower, d.ci_upper, step="post", color=col, alpha=0.10, lw=0)
    ax0.step(d.month, d.survival, where="post", color=col, lw=2)
    ends[prod] = (col, lab_, d.iloc[-1].survival)
# 선 끝 라벨, 겹치면 위아래로 벌림
ys = sorted(ends.items(), key=lambda kv: kv[1][2])
placed = []
for prod, (col, lab_, yv) in ys:
    yy = yv
    if placed and yy - placed[-1] < 0.045: yy = placed[-1] + 0.046
    placed.append(yy)
    ax0.text(XMAX + 1.2, yy, f"{lab_}  {yv:.0%}", fontsize=8.5, color=col, va="center")
for m in (12, 24, 36, 48): ax0.axvline(m, color="#cfd3d9", lw=0.7, ls=(0, (2, 3)), zorder=0)
ax0.set_xlim(0, XMAX); ax0.set_ylim(0.25, 1.0); ax0.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax0.set_ylabel("계약 유지 확률 (Kaplan-Meier, 95% CI)", fontsize=9); ax0.tick_params(labelbottom=False)
ax0.set_title("상품 구성별 계약 유지 곡선", loc="left", fontsize=10.5)
ax0.set_xticks([0, 12, 24, 36, 48, 60])
# at-risk 표
axt.set_ylim(-0.5, 2.9); axt.set_yticks([]); axt.grid(False)
for sp in axt.spines.values(): sp.set_visible(False)
axt.tick_params(length=0, labelsize=9); axt.set_xlabel("최초 계약 후 개월", fontsize=9)
axt.text(-1.5, 2.55, "관측 중인 회사 수", fontsize=7.5, color="#777", ha="left", va="center")
for j, (prod, col, lab_) in enumerate(series):
    yrow = 1.8 - 0.85 * j
    axt.text(-1.5, yrow, lab_, fontsize=7.5, color=col, ha="left", va="center")
    for t in [12, 24, 36, 48, 60]:
        n = km[(km["product"] == prod) & (km.month == t)].at_risk.iloc[0]
        axt.text(t, yrow, f"{n}", fontsize=7.5, color=col, ha="center", va="center")
# forest
names = {"headcount_growth": "인원 성장률 (+1 SD)", "log_hiring_rate": "채용 비율 log (+1 SD)", "log_attrition_rate": "퇴사 비율 log (+1 SD)",
         "log_employees": "재직 인원 log (+1 SD)", "firm_age": "업력 (+1 SD)", "product_both": "상품 A+B (vs A만)", "product_posting": "상품 B만 (vs A만)"}
cx = cx.sort_values("exp(coef)", ascending=False).reset_index(drop=True)
tr = mtrans.blended_transform_factory(ax1.transAxes, ax1.transData)
for i, r in cx.iterrows():
    sig = r["p"] < 0.05; col = ("#9aa0a8" if not sig else (BLUE if r["exp(coef)"] < 1 else ORANGE))
    ax1.plot([r["exp(coef) lower 95%"], r["exp(coef) upper 95%"]], [i, i], color=col, lw=1.6, solid_capstyle="round")
    ax1.scatter(r["exp(coef)"], i, color=col, s=42, zorder=5, edgecolor="white", lw=0.8)
    ax1.text(1.03, i, f"{r['exp(coef)']:.2f}  [{r['exp(coef) lower 95%']:.2f}, {r['exp(coef) upper 95%']:.2f}]",
             transform=tr, fontsize=8, va="center", color="#444", family="monospace")
ax1.text(1.03, -0.85, "HR   [95% CI]", transform=tr, fontsize=8, color="#777", family="monospace", va="center")
ax1.axvline(1, color="#555", lw=0.9); ax1.set_xscale("log"); ax1.set_xlim(0.38, 1.75)
ax1.set_xticks([0.5, 0.7, 1.0, 1.4]); ax1.set_xticklabels(["0.5", "0.7", "1.0", "1.4"]); ax1.minorticks_off()
ax1.set_yticks(range(len(cx))); ax1.set_yticklabels([names[f] for f in cx.feature], fontsize=8.5); ax1.set_ylim(len(cx) - 0.4, -1.2)
ax1.set_xlabel("이탈 위험비 (Cox PH) — 1보다 작으면 이탈 위험 감소", fontsize=9)
ax1.set_title(f"Cox 비례위험 회귀 · concordance {cx.concordance.iloc[0]:.3f}", loc="left", fontsize=10.5)
ax1.text(0.0, -0.2, "파랑 위험 감소 · 주황 위험 증가 · 회색 p ≥ 0.05", transform=ax1.transAxes, fontsize=7.5, color="#777")
fig.subplots_adjust(right=0.87, left=0.06, bottom=0.14, top=0.9)
fig.savefig(FIGURES / "fig7_survival.png"); plt.close(fig)

# fig8 캘리브레이션
cal = R("calibration.csv"); pts = cal.dropna(subset=["mean_pred"]); br = cal.dropna(subset=["brier"])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
for tag, col, lab_ in [("v1", RED, "v1 snapshot"), ("v2", BLUE, "v2 time-cut")]:
    d = pts[pts.variant == tag]; b = br[br.variant == tag].brier.iloc[0]
    ax.plot(d.mean_pred, d.frac_pos, marker="o", color=col, label=f"{lab_} (Brier {b:.3f})")
ax.set_xlabel("Predicted churn probability (LightGBM OOF, 8 bins)"); ax.set_ylabel("Observed churn rate"); ax.legend(fontsize=8); ax.set_title("Calibration")
fig.savefig(FIGURES / "fig8_calibration.png"); plt.close(fig)
print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))
