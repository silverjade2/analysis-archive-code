"""사이트용 그림 5장. 값은 outputs/results 의 CSV에서 읽기만 함"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sitestyle as ss

RES = Path(__file__).resolve().parents[1] / "outputs" / "results"

GROUPS = [
    ("recent_career", "최근 경력 신입", ss.ORANGE),
    ("newgrad", "경력 없는 신입", ss.LIGHT_ORANGE),
    ("mover", "이직 준비", ss.GRAY),
    ("explorer", "탐색형", ss.LIGHT),
    ("hired", "비활성: 취업 완료", ss.BLUE),
    ("lapsed", "비활성: 이탈", ss.LIGHT_BLUE),
]
# seed 42 기준 K-means 군집 번호. 구성(js_cluster_composition.csv)을 보고 붙인 이름
CLUSTER_NAMES = {0: "활동 신입", 1: "입력 빈약, 저활동", 2: "탐색, 이직", 3: "비활성", 4: "입력 빈약, 활동"}


def fig1():
    sw = pd.read_csv(RES / "js_k_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axhspan(sw["silhouette"].min(), sw["silhouette"].max(), color=ss.LIGHT, zorder=0)
    ax.plot(sw["k"], sw["silhouette"], color=ss.BLUE, marker="o", ms=5)
    k5 = sw.set_index("k").loc[5, "silhouette"]
    ax.plot([5], [k5], marker="o", ms=9, color=ss.ORANGE)
    ax.annotate("k=5 채택", (5, k5), xytext=(8, -16), textcoords="offset points", color=ss.ORANGE, fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.set_xticks(sw["k"])
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.grid(axis="y")
    ax.set_title(f"k별 silhouette: {sw['silhouette'].min():.3f}~{sw['silhouette'].max():.3f}")
    ss.save(fig, "fig1_k_sweep")


def fig2():
    comp = pd.read_csv(RES / "js_cluster_composition.csv", index_col="cluster")
    order = comp.index[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    left = np.zeros(len(order))
    for key, label, color in GROUPS:
        v = comp.loc[order, key].to_numpy()
        ax.barh(range(len(order)), v, left=left, color=color, edgecolor="white", label=label)
        left += v
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([CLUSTER_NAMES[c] for c in order])
    ax.set_xlabel("구직자 수")
    ax.set_title("k=5 군집별 잠재 유형 구성")
    ax.legend(loc="lower right", ncol=2)
    ss.save(fig, "fig2_cluster_composition")


def fig3():
    prof = pd.read_csv(RES / "js_dormant_profile.csv", index_col="truth_outcome")
    known = pd.read_csv(RES / "js_dormant_known.csv").iloc[0]
    metrics = [
        ("logins_180d", "로그인"),
        ("views_180d", "공고 조회"),
        ("applies_180d", "지원"),
        ("profile_completeness", "프로필 완성도"),
    ]
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.3, 3.2])
    for i, (col, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        vals = [prof.loc["hired", col], prof.loc["lapsed", col]]
        ax.bar([0, 1], vals, color=[ss.BLUE, ss.LIGHT_BLUE], width=0.7)
        for x, v in zip([0, 1], vals):
            ax.text(x, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["취업", "이탈"], fontsize=8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(label, fontsize=10, loc="center")
    fig.text(0.02, 0.97, "(a) 비활성 군집의 취업 완료와 이탈, 집계 feature 평균", fontsize=12, weight="semibold")

    ax = fig.add_subplot(gs[0, 5])
    hired_unknown = int(known["hired"] - known["hired_by_offer_event"])
    parts = [
        ("확인됨", int(known["hired_by_offer_event"]), ss.BLUE),
        ("확인 불가\n(실제 취업)", hired_unknown, "#9AA6B2"),
        ("확인 불가\n(실제 이탈)", int(known["lapsed"]), ss.LIGHT_BLUE),
    ]
    left = 0
    for label, n, color in parts:
        ax.barh(0, n, left=left, color=color, edgecolor="white", height=0.5)
        txt = "white" if color == ss.BLUE else ss.DARK
        ax.text(left + n / 2, 0, f"{label}\n{n}", ha="center", va="center", fontsize=9, color=txt)
        left += n
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("구직자 수")
    ax.set_title(f"(b) {int(known['hired_or_lapsed'])}명 중 채용 진행 상태로 확인 불가 {known['unknown_share']:.0%}")
    ss.save(fig, "fig3_dormant_split")


def fig4():
    t = pd.read_csv(RES / "js_target_rule.csv").iloc[0]
    hit = int(t["rule_hit_true"])
    fig, ax = plt.subplots(figsize=(9, 2.8))
    rows = [
        (
            "실제 최근 경력 신입",
            [(hit, "규칙이 잡음", ss.ORANGE), (int(t["missed_sparse"]), "놓침: 경력 칸 빈칸", ss.LIGHT_ORANGE)],
        ),
        ("규칙이 고른 사람", [(hit, "맞음", ss.ORANGE), (int(t["rule_hit"]) - hit, "다른 유형", ss.GRAY)]),
    ]
    for y, (_, segs) in enumerate(rows):
        left = 0
        for n, label, color in segs:
            ax.barh(y, n, left=left, color=color, edgecolor="white", height=0.55)
            ax.text(
                left + n / 2,
                y,
                f"{label} {n:,}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if color == ss.ORANGE else ss.DARK,
            )
            left += n
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("구직자 수")
    ax.set_title(f"규칙(경력 6~30개월, 퇴사 12개월 안): recall {t['recall']:.3f}, precision {t['precision']:.3f}")
    ss.save(fig, "fig4_target_rule")


def fig5():
    rc = pd.read_csv(RES / "recovery_comparison.csv")
    names = ["기준선 K-means", "GMM", "DBSCAN (eps=10)", "버스트 feature 7개"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(rc))
    ax.barh(y, rc["missed_total"], color=ss.LIGHT, height=0.55)
    ax.barh(y, rc["recovered"], color=ss.ORANGE, height=0.55)
    for yi, r in zip(y, rc.itertuples(index=False)):
        ax.text(r.missed_total + 0.4, yi, f"{r.recovered}/{r.missed_total}  (ARI {r.ari:.3f})", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.set_xlabel("미회수 28대 중 되찾은 기기 수")
    ax.set_title("기기 데이터: 방법별 미회수 28대 회수")
    ss.save(fig, "fig5_device_recovery")


if __name__ == "__main__":
    ss.setup()
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
