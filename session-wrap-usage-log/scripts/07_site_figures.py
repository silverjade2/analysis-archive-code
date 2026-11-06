"""사이트 그림 5장. 값은 outputs/results 의 CSV 에서 읽기만 함"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sitestyle as ss
from common import RES

A_CLUSTERS = {"commit", "wrapup"}
REGEX_METHODS = ["sum_overlap", "union_overlap", "first", "first_ctx"]
P_R_LABELS = {
    "commit": ((-8, 4), "right"),
    "wrapup": ((-8, -11), "right"),
    "scrape": ((-8, 0), "right"),
    "notebook": ((8, 0), "left"),
    "ui": ((8, 0), "left"),
    "relay": ((-8, 0), "right"),
}


def fig1():
    f = pd.read_csv(RES / "funnel.csv").set_index("step")
    total = f["remaining"].iloc[0]
    human = f["remaining"].iloc[-1]
    tool = f.loc["tool_result", "removed"]
    ide = f.loc["IDE 자동 삽입", "removed"]
    bg = f.loc["다른 세션이 쓴 첫 지시문", "removed"]
    second = ide + bg
    other = total - human - tool - second

    fig, ax = plt.subplots(figsize=(8, 2.5))
    left = 0
    for n, col in [(human, ss.BLUE), (second, ss.ORANGE), (other, ss.GRAY), (tool, ss.LIGHT)]:
        ax.barh(0, n, left=left, color=col, height=0.5, edgecolor="white", lw=1.5)
        left += n

    share = f["remaining_share"].iloc[-1]
    ax.text(0, 0.34, f"사람이 입력\n{human:,} ({share:.1%})", va="bottom", fontsize=9)
    ax.text(human + second, 0.34, f"그 밖 1차 거르기\n{other:,}", va="bottom", fontsize=9)
    ax.text(human + second + other + tool / 2, 0, f"tool_result {tool:,}", ha="center", va="center", fontsize=9.5)
    ax.annotate(
        f"2차에서 뺀 것: IDE 자동 삽입 {ide}, 다른 세션이 쓴 첫 지시문 {bg}",
        xy=(human + second / 2, -0.25),
        xytext=(human + second / 2, -0.62),
        ha="left",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": ss.ORANGE, "lw": 1, "relpos": (0, 1)},
    )
    ax.set_xlim(0, total)
    ax.set_ylim(-0.8, 0.72)
    ax.axis("off")
    ax.set_title(f"user 내용 조각 {total:,}개 중 사람이 입력한 것은 13%다", pad=10)
    ss.save(fig, "fig1_composition")


def fig2():
    c = pd.read_csv(RES / "cluster_counts.csv")
    c = c[c["cluster"] != "other"].sort_values("overlap_n").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(c))
    col = [ss.ORANGE if k in A_CLUSTERS else ss.BLUE for k in c["cluster"]]
    ax.hlines(y, c["perm_min"], c["perm_max"], color=ss.LIGHT, lw=6, label="규칙 순서 200가지의 첫 일치 범위")
    ax.scatter(c["overlap_n"], y, facecolors="white", edgecolors=col, s=40, zorder=3, label="겹침 허용")
    ax.scatter(c["first_n"], y, color=col, s=40, zorder=3, label="첫 일치 (기본 순서)")
    ax.set_yticks(y)
    ax.set_yticklabels(c["label"])
    ax.set_xlabel("메시지 수")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    ax.set_title("규칙 순서만 바꿔도 군집 크기가 움직인다")
    ss.save(fig, "fig2_cluster_counts")


def fig3():
    e = pd.read_csv(RES / "eval_clusters.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    col = [ss.ORANGE if k in A_CLUSTERS else ss.BLUE for k in e["cluster"]]
    ax.scatter(e["first_precision"], e["first_recall"], color=col, s=48, zorder=3, edgecolors="white", lw=1)
    for r in e[e["cluster"].isin(P_R_LABELS)].itertuples():
        offset, ha = P_R_LABELS[r.cluster]
        ax.annotate(
            r.label,
            (r.first_precision, r.first_recall),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9,
        )
    ax.text(0.98, 0.42, "잡은 것은 맞지만\n절반 가까이 놓친다", ha="right", va="center", fontsize=10, color=ss.MUTED)
    ax.text(0.42, 0.74, "다른 의도를\n끌어온다", ha="left", va="center", fontsize=10, color=ss.MUTED)
    ax.set_xlim(0.4, 1.03)
    ax.set_ylim(0.2, 0.85)
    ax.set_xlabel("precision (잡은 것 중 맞은 비율)")
    ax.set_ylabel("recall (실제 그 의도 중 잡은 비율)")
    ax.grid()
    ax.set_title("규칙의 오차는 놓치는 군집과 끌어오는 군집으로 갈린다")
    ss.save(fig, "fig3_rule_precision_recall")


def fig4():
    e = pd.read_csv(RES / "eval_clusters.csv")
    e["error"] = e["ctx_n"] - e["truth_n"]
    e = e.sort_values("error").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(e))
    col = [ss.ORANGE if k in A_CLUSTERS else ss.BLUE for k in e["cluster"]]
    ax.barh(y, e["error"], color=col, height=0.62)
    ax.axvline(0, color=ss.DARK, lw=1)
    for yi, v in zip(y, e["error"]):
        label = f"{v:+d}" if v else "0"
        ax.text(v + (2 if v >= 0 else -2), yi, label, ha="left" if v >= 0 else "right", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(e["label"])
    ax.set_xlim(-115, 100)
    ax.set_xlabel("붙인 뒤 메시지 수 - ground truth (왼쪽은 작게, 오른쪽은 크게 셈)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("짧은 응답을 붙여도 세션 마감은 작고 UI 시안은 크다")
    ss.save(fig, "fig4_context_truth")


def fig5():
    c = pd.read_csv(RES / "eval_candidates.csv").sort_values("truth").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(c))
    lo = c[REGEX_METHODS].min(axis=1)
    hi = c[REGEX_METHODS].max(axis=1)
    col = [ss.ORANGE if k == "A" else ss.BLUE for k in c["candidate"]]
    ax.hlines(y, lo, hi, color=col, lw=9, label="정규식으로 센 네 방법의 범위")
    ax.scatter(c["truth"], y, marker="|", color=ss.DARK, s=260, lw=2, zorder=4, label="ground truth")
    for yi, a, b, t in zip(y, lo, hi, c["truth"]):
        ax.text(a, yi + 0.27, f"{a}", ha="center", fontsize=8.5)
        ax.text(b, yi + 0.27, f"{b}", ha="center", fontsize=8.5)
        ax.text(t, yi - 0.42, f"{t}", ha="center", fontsize=8.5, fontweight="semibold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.candidate} {r.name}" for r in c.itertuples()])
    ax.set_ylim(-0.7, len(c) - 0.4)
    ax.set_xlim(0, 820)
    ax.set_xlabel("덮는 메시지 수")
    ax.grid(axis="x")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.45))
    ax.set_title("어떻게 세도 A가 가장 크고, C와 D는 ground truth에서 뒤집힌다")
    ss.save(fig, "fig5_candidates")


ss.setup()
for f in [fig1, fig2, fig3, fig4, fig5]:
    f()
