# 구직자 군집 그림. 값은 outputs/results/js_*.csv 에서 읽기만 함
# 출력: outputs/figures/fig4_jobseeker_clusters.png

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

base = Path(__file__).resolve().parents[1]
res = base / "outputs" / "results"
fig_dir = base / "outputs" / "figures"

GROUPS = [
    ("recent_career", "최근 경력 신입", "#d9480f"),
    ("newgrad", "경력 없는 신입", "#f4a261"),
    ("mover", "이직 준비", "#8d99ae"),
    ("explorer", "탐색형", "#c9ccd1"),
    ("hired", "비활성: 취업 완료", "#2b6cb0"),
    ("lapsed", "비활성: 이탈", "#90cdf4"),
]
CLUSTER_NAMES = {0: "활동 신입", 1: "입력 빈약, 저활동", 2: "탐색, 이직", 3: "비활성", 4: "입력 빈약, 활동"}

comp = pd.read_csv(res / "js_cluster_composition.csv", index_col="cluster")
known = pd.read_csv(res / "js_dormant_known.csv").iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})

ax = axes[0]
order = comp.index[::-1]
left = pd.Series(0, index=order)
for key, label, color in GROUPS:
    ax.barh([CLUSTER_NAMES[c] for c in order], comp.loc[order, key], left=left, color=color, label=label)
    left = left + comp.loc[order, key]
ax.set_xlabel("구직자 수")
ax.set_title("(a) k=5 군집별 잠재 유형 구성")
ax.legend(fontsize=8, frameon=False, loc="lower right")

ax = axes[1]
hired_unknown = int(known["hired"] - known["hired_by_offer_event"])
parts = [
    ("채용 진행 상태로 확인된 취업", int(known["hired_by_offer_event"]), "#2b6cb0"),
    ("확인 불가: 실제 취업", hired_unknown, "#7fa7d6"),
    ("확인 불가: 실제 이탈", int(known["lapsed"]), "#90cdf4"),
]
left = 0
for label, n, color in parts:
    ax.barh(["비활성 군집\n(취업 완료 + 이탈)"], n, left=left, color=color, label=f"{label} {n}")
    left += n
ax.set_xlabel("구직자 수")
ax.set_title(f"(b) 비활성 군집 {int(known['hired_or_lapsed'])}명 중 확인 불가 {known['unknown_share']:.0%}")
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(fig_dir / "fig4_jobseeker_clusters.png", dpi=150, bbox_inches="tight")
