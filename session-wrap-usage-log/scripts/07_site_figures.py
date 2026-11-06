"""사이트 그림 5장. 값은 outputs/results 의 CSV 에서 읽기만 함"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sitestyle as ss
from common import RES

A_CLUSTERS = {"commit", "wrapup"}


def fig1():
    f = pd.read_csv(RES / "funnel.csv")
    f = f[(f["removed"] > 0) | (f.index == 0)].reset_index(drop=True)
    second = f["step"].isin(["IDE 자동 삽입", "다른 세션이 쓴 첫 지시문"])
    labels = ["user 내용 조각 전체"] + [f"{s} 제외" for s in f["step"].iloc[1:]]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(f))[::-1]
    cols = [ss.ORANGE if s else ss.GRAY for s in second]
    cols[0] = ss.BLUE
    ax.barh(y, f["remaining"], color=cols, height=0.62)
    for yi, r, n in zip(y, f["remaining"], f["removed"]):
        ax.text(r, yi, f"  {r:,}" + (f"  (-{n:,})" if n else ""), va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, f["remaining"].max() * 1.25)
    ax.set_xlabel("남은 내용 조각 수")
    ax.set_title("user 기록에서 사람이 입력한 메시지까지 (회색 1차, 주황 2차)")
    ss.save(fig, "fig1_funnel")


def fig2():
    c = pd.read_csv(RES / "cluster_counts.csv")
    c = c[c["cluster"] != "other"].sort_values("overlap_n").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(c))
    col = [ss.ORANGE if k in A_CLUSTERS else ss.BLUE for k in c["cluster"]]
    ax.hlines(y, c["perm_min"], c["perm_max"], color=ss.LIGHT, lw=6, label="규칙 순서 200가지의 첫 일치 범위")
    ax.scatter(c["overlap_n"], y, facecolors="white", edgecolors=col, s=40, zorder=3, label="겹침 허용")
    ax.scatter(c["first_n"], y, color=col, s=40, zorder=3, label="첫 일치 (기본 순서)")
    moved = c[(c["first_n_before_2nd"] - c["first_n"]) > 5]
    ax.scatter(
        moved["first_n_before_2nd"],
        moved.index,
        marker="x",
        color=ss.MUTED,
        s=30,
        zorder=3,
        label="2차 거르기 전 첫 일치",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(c["label"])
    ax.set_xlabel("메시지 수")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    ax.set_title("군집별 메시지 수: 겹침 허용과 첫 일치")
    ss.save(fig, "fig2_cluster_counts")


def fig3():
    e = pd.read_csv(RES / "eval_clusters.csv").sort_values(["first_precision", "first_recall"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(e))
    col = [ss.ORANGE if k in A_CLUSTERS else ss.BLUE for k in e["cluster"]]
    ax.hlines(y, e["first_recall"], e["first_precision"], color=ss.LIGHT, lw=2)
    ax.scatter(e["first_precision"], y, color=col, s=36, zorder=3, label="precision (잡은 것 중 맞은 비율)")
    ax.scatter(
        e["first_recall"],
        y,
        facecolors="white",
        edgecolors=col,
        s=36,
        zorder=3,
        label="recall (실제 그 의도 중 잡은 비율)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(e["label"])
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("비율")
    ax.grid(axis="x")
    ax.legend(loc="upper left")
    ax.set_title("첫 일치 규칙의 군집별 precision과 recall (ground truth 기준)")
    ss.save(fig, "fig3_rule_precision_recall")


def fig4():
    e = pd.read_csv(RES / "eval_clusters.csv").sort_values("truth_n").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(e))
    ax.hlines(y, e["first_n"], e["ctx_n"], color=ss.LIGHT, lw=2)
    ax.scatter(e["first_n"], y, color=ss.GRAY, s=36, zorder=3, label="첫 일치")
    ax.scatter(e["ctx_n"], y, color=ss.ORANGE, s=36, zorder=3, label="짧은 응답을 앞 턴에 붙인 뒤")
    ax.scatter(e["truth_n"], y, marker="|", color=ss.DARK, s=160, lw=1.6, zorder=4, label="ground truth")
    ax.set_yticks(y)
    ax.set_yticklabels(e["label"])
    ax.set_xlabel("메시지 수")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    ax.set_title("군집별 메시지 수: 첫 일치, 짧은 응답을 붙인 뒤, ground truth")
    ss.save(fig, "fig4_context_truth")


def fig5():
    c = pd.read_csv(RES / "eval_candidates.csv")
    methods = [
        ("sum_overlap", "겹침 허용 합"),
        ("union_overlap", "합집합"),
        ("first", "첫 일치"),
        ("first_ctx", "짧은 응답 포함"),
        ("truth", "ground truth"),
    ]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    color = {"A": ss.ORANGE, "B": ss.GRAY, "C": ss.GRAY, "D": ss.BLUE}
    ends = c.set_index("candidate")["truth"].sort_values()
    label_y = {}
    for k, v in ends.items():
        label_y[k] = max(v, max(label_y.values(), default=-1e9) + 30)  # 끝 라벨 겹침 방지
    for r in c.itertuples():
        v = [getattr(r, m) for m, _ in methods]
        a = r.candidate == "A"
        ax.plot(
            x,
            v,
            marker="o",
            color=color[r.candidate],
            lw=2.2 if a else 1.4,
            ms=5,
            ls="--" if r.candidate == "C" else "-",
        )
        ax.text(x[-1] + 0.1, label_y[r.candidate], f"{r.candidate} {r.name}", va="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_xlim(-0.2, len(methods) - 1 + 1.1)
    ax.set_ylabel("덮는 메시지 수")
    ax.grid(axis="y")
    ax.set_title("후보별로 덮는 메시지 수: 세는 방법 다섯 가지")
    ss.save(fig, "fig5_candidates")


ss.setup()
for f in [fig1, fig2, fig3, fig4, fig5]:
    f()
