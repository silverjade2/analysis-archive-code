"""그림 4장. 1~3은 운영 로그 분석 기록(data/record_rates.csv), 4는 가상 로그 결과. 다시 계산하지 않는다."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import DATA, RES, UNMATCHED
from sitestyle import BLUE, GRAY, LIGHT, MUTED, ORANGE, save, setup

setup()

STATE_KO = {"normal": "정상", "fallback": "fallback", "none": "무응답"}
STATE_COLOR = {"normal": GRAY, "fallback": ORANGE, "none": LIGHT}

rec = pd.read_csv(DATA / "record_rates.csv")

f1 = rec[rec["figure"] == "fig1"]
fig, ax = plt.subplots(figsize=(7.2, 2.4))
left = 0.0
seg_color = {"1위 문구": GRAY, "2위 문구": "#8A8F98", "3~15위 인식 실패 문구": LIGHT, "상위 15개 밖": "#f4f4f5"}
for r in f1.itertuples():
    ax.barh([0], [r.value_pct], left=left, color=seg_color.get(r.item, ORANGE), height=0.6, edgecolor="white")
    if r.value_pct >= 15:
        ax.text(left + r.value_pct / 2, 0, f"{r.item}\n{r.value_pct:g}%", ha="center", va="center", fontsize=8.5)
    elif r.item != "정보 부재 문구":
        ax.annotate(
            f"{r.item} {r.value_pct:g}%",
            xy=(left + r.value_pct / 2, 0.3),
            xytext=(left + r.value_pct / 2, 0.75),
            ha="center",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )
    left += r.value_pct
info = f1[f1["item"] == "정보 부재 문구"].iloc[0]
ax.annotate(
    f"정보 부재 문구 {info.value_pct:g}%",
    xy=(100 - info.value_pct / 2, 0.3),
    xytext=(88, 0.75),
    ha="center",
    fontsize=8.5,
    color=ORANGE,
    arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.8},
)
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 1.0)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("fallback 중 비중 (%)")
ax.set_title("fallback 문구 구성")
save(fig, "fig1_fallback_phrases")

f2 = rec[rec["figure"] == "fig2"].pivot(index="group", columns="item", values="value_pct")
f2 = f2.loc[["최신성 요구 있음", "최신성 요구 없음"]]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
left = np.zeros(len(f2))
for st in ["normal", "fallback", "none"]:
    v = f2[st].to_numpy()
    ax.barh(f2.index, v, left=left, color=STATE_COLOR[st], label=STATE_KO[st], height=0.6)
    left += v
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("응답 상태 구성 (%)")
ax.set_title("최신성 요구 여부별 응답 상태")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35))
save(fig, "fig2_state_by_fresh")

f3 = rec[rec["figure"] == "fig3"].set_index("item").loc[["fallback", "normal", "none"]]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.bar([STATE_KO[s] + " 직후" for s in f3.index], f3["value_pct"], color=[ORANGE, GRAY, GRAY], width=0.6)
ax.set_ylabel("재시도율 (%)")
ax.set_title("직전 응답 상태별 재시도율")
ax.grid(axis="y")
save(fig, "fig3_retry_by_prev_state")

pr = pd.read_csv(RES / "priority_all.csv")
pr = pr[pr["topic"] != UNMATCHED]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.scatter(pr["validity_h"], pr["fail_rate"] * 100, s=pr["n"] * 3, color=BLUE, alpha=0.55, edgecolor="white")
for r in pr.itertuples():
    ax.annotate(
        r.topic,
        (r.validity_h, r.fail_rate * 100),
        xytext=(0, 14 + np.sqrt(r.n) * 0.9),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
ax.set_xscale("log")
ax.set_xticks([3, 24, 72, 336, 720], ["3시간", "1일", "3일", "2주", "1개월"])
ax.minorticks_off()
ax.set_xlim(1.5, 1500)
lo, hi = pr["fail_rate"].min() * 100, pr["fail_rate"].max() * 100
ax.set_ylim(lo - 6, hi + 8)
ax.set_xlabel("답의 유효 기간 (로그 축)")
ax.set_ylabel("실패율 (%)")
ax.set_title("주제별 유효 기간과 실패율 (원 크기는 발화 수)")
ax.grid(True)
save(fig, "fig4_validity_vs_fail")
