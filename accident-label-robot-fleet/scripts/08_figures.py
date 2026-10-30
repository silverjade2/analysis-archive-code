"""그림 6장. 값은 outputs/results 의 CSV에서 읽기만 함"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sitestyle as ss
from common import K_DAYS, RES, TYPE_NAME, TYPE_ORDER

TYPE_COLOR = {"A": ss.ORANGE, "B": ss.LIGHT_ORANGE, "C": ss.BLUE, "D": ss.GRAY}


def fig1():
    share = pd.read_csv(RES / "type_share.csv").set_index("type")["share"]
    per = pd.read_csv(RES / "type_by_period.csv").set_index("period")
    sens = pd.read_csv(RES / "type_sensitivity.csv")
    rows = [("전체", share), ("코드 분리 전", per.loc["분리 전"]), ("코드 분리 뒤", per.loc["분리 뒤"])]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, (label, s) in enumerate(rows):
        left = 0.0
        for t in TYPE_ORDER:
            v = float(s[t])
            ax.barh(i, v, left=left, color=TYPE_COLOR[t], edgecolor="white", height=0.62)
            ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=10, color=ss.DARK)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("알림 사건 중 비중")
    ax.set_title(f"알림 사건의 유형 구성. 창을 바꿔도 A는 {sens['A'].min():.0%}~{sens['A'].max():.0%}")
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR[t]) for t in TYPE_ORDER],
        [TYPE_NAME[t] for t in TYPE_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
    )
    ss.save(fig, "fig1_type_share")


def fig2():
    m = pd.read_csv(RES / "a_signal_monthly.csv")
    base = pd.read_csv(RES / "stop_baseline.csv")
    hours = base.groupby("hour")["stop_rate"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    x = np.arange(len(m))
    cols = [ss.ORANGE if r >= 2 else ss.GRAY for r in m["ratio"]]
    ax.bar(x, m["ratio"], color=cols, width=0.65)
    ax.axhline(2, color=ss.DARK, lw=1, ls="--")
    ax.text(len(m) - 0.5, 2.05, "판단 기준 2배", ha="right", va="bottom", fontsize=9)
    for xi, r in zip(x, m["ratio"]):
        ax.text(xi, r + 0.04, f"{r:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[5:] + "월" for s in m["month"]], fontsize=8)
    ax.set_ylabel("알림 뒤 멈춤 ÷ 시각대 기대")
    ax.set_title("달별 A 신호 배수")
    ax = axes[1]
    ax.plot(hours.index, hours.to_numpy(), color=ss.BLUE, marker="o", ms=3)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("시각")
    ax.set_ylabel("평소 멈춤 비율")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("가동 중 아무 시점에서 1~6시간 뒤 멈춰 있을 비율")
    ax.grid(axis="y")
    ss.save(fig, "fig2_a_signal")


def fig3():
    c = pd.read_csv(RES / "recurrence_curve.csv")
    s = pd.read_csv(RES / "recurrence_summary.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.scatter(c["day"], c["rate"].clip(upper=40), s=10, color=ss.GRAY, label="경과일별 재발률")
    ax.plot(c["day"], c["rate_7d"], color=ss.ORANGE, lw=2, label="그날부터 7일 평균")
    ax.axhline(s["base_rate"], color=ss.BLUE, lw=1.2, ls="--")
    ax.axhline(s["long_run_rate"], color=ss.ORANGE, lw=1, ls=":")
    ax.text(60, s["base_rate"] - 0.4, f"평소 발생률 {s['base_rate']:.1f}", ha="right", va="top", color=ss.BLUE)
    ax.text(60, s["long_run_rate"] + 0.4, f"경과 31~60일 수준 {s['long_run_rate']:.1f}", ha="right", va="bottom")
    ax.set_ylim(0, 40)
    ax.axvline(K_DAYS, color=ss.DARK, lw=1)
    ax.text(K_DAYS + 0.5, 37, f"K{K_DAYS}", fontsize=10)
    ax.set_xlim(0, 61)
    ax.set_xlabel("직전 A, B 사건 뒤 경과일")
    ax.set_ylabel("1,000 기기-일당")
    ax.set_title("재발 위험은 평소 수준으로 내려오지 않는다")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8))
    ss.save(fig, "fig3_recurrence")


def fig4():
    f = pd.read_csv(RES / "funnel.csv")
    k = pd.read_csv(RES / "k_anchors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(f))
    ax.barh(y, f["n"], color=[ss.GRAY, ss.LIGHT_ORANGE, ss.ORANGE], height=0.6)
    for yi, r in zip(y, f.itertuples(index=False)):
        tail = "" if pd.isna(r.share_of_prev) else f"  (앞 단계의 {r.share_of_prev:.0%})"
        ax.text(r.n, yi, f" {r.n:,}{tail}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(f["step"])
    ax.invert_yaxis()
    ax.set_xlim(0, f["n"].max() * 1.45)
    ax.set_xlabel("건수")
    ax.set_title("알림 사건에서 예측 기준 시점까지")
    ax = axes[1]
    x = np.arange(len(k))
    ax.bar(x, k["ab"], color=[ss.ORANGE if str(v) == str(K_DAYS) else ss.LIGHT_ORANGE for v in k["k"]], width=0.6)
    for xi, v in zip(x, k["ab"]):
        ax.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K{v}" if v != "없음" else "K 없음" for v in k["k"]])
    ax.set_ylabel("A, B 기준 시점 수")
    ax.set_title("K별 기준 시점")
    ss.save(fig, "fig4_funnel")


def fig5():
    t = pd.read_csv(RES / "consult_link.csv")
    r = pd.read_csv(RES / "consult_a_over_c.csv").iloc[0]
    periods = ["자동 상담 전", "자동 상담 뒤", "전체"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.2
    for j, typ in enumerate(TYPE_ORDER):
        v = [float(t[(t["period"] == p) & (t["type"] == typ)]["link_rate"].iloc[0]) for p in periods]
        ax.bar(np.arange(3) + (j - 1.5) * w, v, width=w, color=TYPE_COLOR[typ], label=TYPE_NAME[typ])
    ax.set_xticks(range(3))
    ax.set_xticklabels(periods)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("7일 안 상담이 붙은 사건")
    ax.set_title(f"유형별 상담 연결률. A ÷ C = {r['a_over_c']:.2f} (기준 {r['threshold']:.1f})")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="y")
    ss.save(fig, "fig5_consult_link")


def fig6():
    m = pd.read_csv(RES / "exposure_monthly.csv")
    m = m[m["month"] >= "2025-02"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(m))
    for col, label, color in [
        ("per_1k_days", "달력 기기-일당", ss.GRAY),
        ("per_1k_active", "가동 기기-일당", ss.BLUE),
        ("per_10k_hours", "가동 시간당", ss.ORANGE),
    ]:
        v = m[col] / m[col].iloc[0]
        ax.plot(x, v, marker="o", ms=4, color=color, label=label)
        ax.text(x[-1] + 0.15, v.iloc[-1], f"{v.iloc[-1] - 1:+.0%}", va="center", color=color, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([s[5:] + "월" for s in m["month"]])
    ax.set_ylabel("2월 = 1")
    ax.set_title("알림 사건 비율: 분모를 가동으로 바꿔도 줄어든다")
    ax.legend(loc="lower left")
    ax.grid(axis="y")
    ss.save(fig, "fig6_exposure")


def main():
    ss.setup()
    for f in [fig1, fig2, fig3, fig4, fig5, fig6]:
        f()


if __name__ == "__main__":
    main()
