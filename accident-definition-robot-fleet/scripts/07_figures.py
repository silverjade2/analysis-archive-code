"""그림 8장. 값은 outputs/results 의 CSV에서 읽기만 함"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sitestyle as ss
from common import K_DAYS, RES, SPLIT, TIER_ORDER, X_MIN

TIER_COLOR = {TIER_ORDER[0]: ss.ORANGE, TIER_ORDER[1]: ss.BLUE, TIER_ORDER[2]: ss.GRAY}


def fig1():
    cat = pd.read_csv(RES / "category_share.csv")
    tier = pd.read_csv(RES / "tier_share.csv").set_index("tier").reindex(TIER_ORDER)
    prof = pd.read_csv(RES / "code_profile.csv")
    tier_of = prof.drop_duplicates("category").set_index("category")["tier"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    cat = cat.sort_values("rows")
    y = np.arange(len(cat))
    cols = [TIER_COLOR[tier_of[c]] for c in cat["category"]]
    ax.barh(y + 0.2, cat["codes"] / cat["codes"].sum(), height=0.38, color=cols, alpha=0.35)
    ax.barh(y - 0.2, cat["log_share"], height=0.38, color=cols)
    for yi, r in zip(y, cat.itertuples(index=False)):
        ax.text(r.codes / cat["codes"].sum(), yi + 0.2, f" {int(r.codes)}종", va="center", fontsize=8, color=ss.MUTED)
        ax.text(r.log_share, yi - 0.2, f" {r.log_share:.1%}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(cat["category"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("비중")
    ax.set_title("범주별 코드 수 (연한 막대)와 로그 양 (진한 막대)")
    ax = axes[1]
    for bi, col in enumerate(["log_share", "code_share"]):
        left = 0.0
        for t in TIER_ORDER:
            v = float(tier.loc[t, col])
            ax.barh(bi, v, left=left, color=TIER_COLOR[t], edgecolor="white", height=0.6)
            if v >= 0.06:
                ax.text(left + v / 2, bi, f"{v:.0%}", ha="center", va="center", fontsize=10, color="white")
            elif v > 0:
                ax.annotate(
                    f"A {v:.1%}",
                    xy=(left + v / 2, bi + 0.3),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=9,
                    color=TIER_COLOR[t],
                )
            left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["로그 행", "코드 수"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("비중")
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in TIER_ORDER],
        TIER_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
    )
    ax.set_title("3단계별 코드 수와 로그 행 비중")
    fig.tight_layout()
    ss.save(fig, "fig1_code_vs_log_share")


def fig2():
    lv = pd.read_csv(RES / "level_timeline.csv", parse_dates=["month"]).set_index("month")
    cmp = pd.read_csv(RES / "definition_compare.csv", parse_dates=["month"]).set_index("month")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bottom = np.zeros(len(lv))
    for col, c, lab in [
        ("level_4_share", ss.GRAY, "등급 4"),
        ("level_3_share", ss.BLUE, "등급 3"),
        ("level_1_share", ss.ORANGE, "등급 1"),
    ]:
        ax.bar(lv.index, lv[col], bottom=bottom, width=22, color=c, label=lab)
        bottom += lv[col].to_numpy()
    ax.set_ylim(0, 1)
    ax.set_ylabel("비중")
    ax.set_title("전도, 낙하 로그의 관제 등급 구성")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax = axes[1]
    ax.plot(cmp.index, cmp["by_code_rows"], color=ss.ORANGE, lw=2.2, marker="o", ms=4, label="코드 기준 (전도, 낙하)")
    ax.plot(
        cmp.index,
        cmp["by_level1_rows"],
        color=ss.BLUE,
        lw=2.2,
        marker="o",
        ms=4,
        label="등급 1 기준 (배터리 경고 제외)",
    )
    ax.set_ylabel("월별 로그 행")
    ax.set_title("같은 기기, 두 정의의 월별 추이")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    for a in axes:
        a.tick_params(axis="x", rotation=30)
        a.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig2_level_vs_code_definition")


def fig3():
    g = pd.read_csv(RES / "gap_bins.csv")
    d = g.dropna(subset=["density"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d["bin_no"], d["density"], color=ss.ORANGE, lw=2.2, marker="o", ms=4)
    x_bin = int(d.loc[d["hi_min"] == X_MIN["TIP"], "bin_no"].iloc[0]) + 0.5
    ax.axvline(x_bin, color=ss.DARK, ls="--", lw=1)
    ax.text(x_bin + 0.15, ax.get_ylim()[1] * 0.92, f"X = {X_MIN['TIP']}분", fontsize=9)
    ticks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18]
    ax.set_xticks(ticks)
    ax.set_xticklabels([g.loc[g["bin_no"] == t, "bin_label"].iloc[0] for t in ticks], rotation=30, ha="right")
    ax.set_ylabel("밀도 (비중 / log10 구간 폭)")
    ax.set_xlabel("같은 기기의 직전 전도 로그와의 간격")
    ax.set_title("전도 로그 간격 분포: 수 분 안 연속 기록과 며칠 뒤 재발 사이")
    ax.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig3_gap_density")


def fig4():
    k = pd.read_csv(RES / "k_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    labels = ["K 없음" if v == "none" else f"K{v}" for v in k["k_days"]]
    cols = [ss.ORANGE if str(v) == str(K_DAYS) else ss.GRAY for v in k["k_days"]]
    ax.bar(labels, k["accidents"], color=cols)
    for i, v in enumerate(k["accidents"]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("사고 건수")
    ax.set_title("재발을 접는 기간 K에 따른 사고 건수")
    ax = axes[1]
    ax.plot(labels, k["top10pct_share"], color=ss.ORANGE, lw=2.2, marker="o", ms=5, label="상위 10% 기기의 몫")
    ax.set_ylim(0, max(0.5, k["top10pct_share"].max() * 1.2))
    ax.set_ylabel("비중")
    ax2 = ax.twinx()
    ax2.plot(labels, k["max_per_device"], color=ss.BLUE, lw=1.8, marker="s", ms=4, ls="--", label="한 기기 최다 건수")
    ax2.set_ylabel("건")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("K별 기기 쏠림 (상위 10% 몫, 한 기기 최다)")
    for a in axes:
        a.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig4_k_sensitivity")


def fig5():
    mon = pd.read_csv(RES / "monthly_rate.csv", parse_dates=["month"]).set_index("month")
    ten = pd.read_csv(RES / "tenure_rate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    full = ~mon["partial"]
    ax.plot(
        mon.index[full],
        mon.loc[full, "rate_lift"],
        color=ss.BLUE,
        lw=1.8,
        ls="--",
        marker="o",
        ms=4,
        label="들림 포함 (상한)",
    )
    ax.plot(
        mon.index[full], mon.loc[full, "rate"], color=ss.ORANGE, lw=2.4, marker="o", ms=4, label="기본 (전도, 낙하)"
    )
    ax.plot(mon.index[~full], mon.loc[~full, "rate"], color=ss.GRAY, marker="o", ms=4, ls="none", label="부분 달")
    ax.axvline(SPLIT, color=ss.DARK, ls=":", lw=1)
    ax.text(SPLIT, ax.get_ylim()[0] + 0.2, " 들림, 낙하 코드 분리", fontsize=8.5, va="bottom")
    ax.set_ylabel("기기 1,000대당 하루 건수")
    ax.set_title("월별 사고율: 두 판")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    overall = ten["accidents"].sum() / ten["device_days"].sum() * 1000
    cols = [ss.ORANGE if i == 0 else ss.GRAY for i in range(len(ten))]
    ax.bar(ten["age_days"], ten["rate"], color=cols)
    ax.axhline(overall, color=ss.DARK, ls="--", lw=1)
    ax.text(len(ten) - 0.5, overall, f" 평균 {overall:.2f}", fontsize=8.5, va="bottom", ha="right")
    for i, v in enumerate(ten["rate"]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xlabel("설치 후 경과일")
    ax.set_ylabel("기기 1,000대당 하루 건수")
    ax.set_title("설치 후 경과별 사고율")
    ax.tick_params(axis="x", rotation=30)
    for a in axes:
        a.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig5_monthly_and_tenure")


def fig6():
    s = pd.read_csv(RES / "sir.csv", parse_dates=["month"]).set_index("month")
    t = pd.read_csv(RES / "eval_truth_sir.csv", parse_dates=["month"]).set_index("month")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    full = ~s["partial"]
    ax.plot(s.index[full], s.loc[full, "sir_basic"], color=ss.ORANGE, lw=2.2, marker="o", ms=4, label="기본")
    ax.plot(
        s.index[full],
        s.loc[full, "sir_lift_included"],
        color=ss.BLUE,
        lw=1.8,
        ls="--",
        marker="o",
        ms=4,
        label="들림 포함",
    )
    ax.plot(
        t.index[full],
        t.loc[full, "sir_truth"],
        color=ss.GRAY,
        lw=1.6,
        marker="s",
        ms=3.5,
        label="ground truth 사고",
    )
    ax.axhline(1, color=ss.DARK, lw=0.9, ls=":")
    ax.axvline(SPLIT, color=ss.DARK, ls=":", lw=1)
    ax.set_ylabel("관측 ÷ 기대 (SIR)")
    ax.set_title("설치 후 경과 구성을 맞춘 기대 대비 관측")
    ax.legend(loc="lower left")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig6_sir")


def fig7():
    """코드 하나가 점 하나. 가로 coverage, 세로 기기당 행 수(log), 크기 행 수"""
    prof = pd.read_csv(RES / "code_profile.csv")
    prof = prof[prof["rows"] > 0]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    groups = [
        (prof["category"] == "주행", ss.BLUE, "주행 판단 기록"),
        (prof["code"].isin(["TIP", "DROP"]), ss.ORANGE, "사고 (전도, 낙하)"),
        (~prof["category"].isin(["주행"]) & ~prof["code"].isin(["TIP", "DROP"]), ss.GRAY, "그 밖의 코드"),
    ]
    for m, c, lab in groups:
        d = prof[m]
        ax.scatter(
            d["coverage"],
            d["rows_per_device"],
            s=np.sqrt(d["rows"]) * 1.6 + 10,
            color=c,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.6,
            label=lab,
        )
    shift = {"N1": (8, 9), "N4": (8, -11), "N2": (-8, -16)}  # 오른쪽 위에 몰린 주행 코드 라벨이 겹치지 않게
    for r in prof.nlargest(6, "rows").itertuples():
        dx, dy = shift.get(r.code, (5, 3))
        ax.annotate(
            r.name,
            (r.coverage, r.rows_per_device),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color=ss.MUTED,
            ha="right" if dx < 0 else "left",
        )
    for code in ["TIP", "DROP"]:
        r = prof[prof["code"] == code].iloc[0]
        ax.annotate(
            r["name"],
            (r["coverage"], r["rows_per_device"]),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=8.5,
            color=ss.ORANGE,
        )
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.08)
    ax.axvline(0.5, color=ss.LIGHT, lw=0.8, ls=":")
    ax.text(
        0.98,
        0.55,
        "모든 기기에서 많이: 일상 운영 기록",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=ss.MUTED,
    )
    ax.text(
        0.02,
        0.95,
        "소수 기기에서 많이: 몰림",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=ss.MUTED,
    )
    ax.set_xlabel("그 코드를 남긴 기기 비율 (coverage)")
    ax.set_ylabel("기기당 로그 행 (log)")
    ax.set_title("코드별 coverage와 기기당 로그 수 (점 크기 = 로그 행 수)")
    ax.legend(loc="lower right")
    ax.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig7_code_bubble")


def fig8():
    """전달용 한 장. 여섯 칸 모두 앞 그림과 같은 CSV"""
    cat = pd.read_csv(RES / "category_share.csv").sort_values("rows")
    fun = pd.read_csv(RES / "funnel.csv")
    mon = pd.read_csv(RES / "monthly_rate.csv", parse_dates=["month"]).set_index("month")
    ten = pd.read_csv(RES / "tenure_rate.csv")
    g = pd.read_csv(RES / "gap_bins.csv")
    k = pd.read_csv(RES / "k_sensitivity.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))

    ax = axes[0, 0]
    ax.barh(cat["category"], cat["log_share"], color=[ss.BLUE if c == "주행" else ss.GRAY for c in cat["category"]])
    for i, v in enumerate(cat["log_share"]):
        ax.text(v, i, f" {v:.1%}", va="center", fontsize=8.5)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("전체 로그 대비 비중")
    ax.set_title("1. 에러 로그의 정체: 범주별 비중")

    ax = axes[0, 1]
    cols = [ss.GRAY, ss.GRAY, ss.LIGHT_BLUE, ss.ORANGE]
    ax.bar(fun["step"], fun["count"], color=cols)
    for i, v in enumerate(fun["count"]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("건")
    ax.set_title("2. 로그 행에서 사고 건수로")
    ax.tick_params(axis="x", labelsize=8.5)

    ax = axes[0, 2]
    full = ~mon["partial"]
    ax.plot(
        mon.index[full],
        mon.loc[full, "rate_lift"],
        color=ss.BLUE,
        lw=1.6,
        ls="--",
        marker="o",
        ms=3.5,
        label="들림 포함 (상한)",
    )
    ax.plot(mon.index[full], mon.loc[full, "rate"], color=ss.ORANGE, lw=2.2, marker="o", ms=3.5, label="기본")
    ax.axvline(SPLIT, color=ss.DARK, ls=":", lw=1)
    ax.set_ylabel("기기 1,000대당 하루 건수")
    ax.set_title("3. 월별 사고율 (점선 = 코드 분리)")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 0]
    overall = ten["accidents"].sum() / ten["device_days"].sum() * 1000
    ax.bar(ten["age_days"], ten["rate"], color=[ss.ORANGE if i == 0 else ss.GRAY for i in range(len(ten))])
    ax.axhline(overall, color=ss.DARK, ls="--", lw=1)
    ax.text(len(ten) - 0.5, overall, f" 평균 {overall:.2f}", fontsize=8.5, va="bottom", ha="right")
    ax.set_xlabel("설치 후 경과일")
    ax.set_ylabel("기기 1,000대당 하루 건수")
    ax.set_title("4. 설치 후 경과별 사고율")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    d = g.dropna(subset=["density"])
    ax.plot(d["bin_no"], d["density"], color=ss.ORANGE, lw=2, marker="o", ms=3.5)
    x_bin = int(d.loc[d["hi_min"] == X_MIN["TIP"], "bin_no"].iloc[0]) + 0.5
    ax.axvline(x_bin, color=ss.DARK, ls="--", lw=1)
    ax.text(x_bin + 0.2, ax.get_ylim()[1] * 0.9, f"X = {X_MIN['TIP']}분", fontsize=8.5)
    ticks = [1, 5, 9, 13, 15, 17]
    ax.set_xticks(ticks)
    ax.set_xticklabels([g.loc[g["bin_no"] == t, "bin_label"].iloc[0] for t in ticks], rotation=30, ha="right")
    ax.set_ylabel("밀도")
    ax.set_title("5. 전도 로그 간격 (묶는 간격 X의 근거)")

    ax = axes[1, 2]
    labels = ["K 없음" if v == "none" else f"K{v}" for v in k["k_days"]]
    ax.bar(labels, k["accidents"], color=[ss.ORANGE if str(v) == str(K_DAYS) else ss.GRAY for v in k["k_days"]])
    for i, v in enumerate(k["accidents"]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("사고 건수")
    ax.set_title("6. 재발을 접는 기간 K에 따른 사고 건수")

    for a in axes.flat:
        a.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig8_summary")


if __name__ == "__main__":
    ss.setup()
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
        f()
