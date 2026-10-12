"""그림 15장. 값은 outputs/results 의 CSV 에서 읽기만 함"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sitestyle as ss
from common import FEAT, NAME, OCC, RES, SCHEDULE_HOURS
from matplotlib.colors import LinearSegmentedColormap

SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#8a9aa9", "#2f3e4c"])
PAT = {1: ss.BLUE, 2: ss.ORANGE, 3: "#B8913A"}
FNAME = [NAME[f] for f in FEAT]


def pat_name():
    """패턴 번호에 짧은 이름 붙이기 (사용률 표 기준)"""
    r = pd.read_csv(RES / "pattern_feature_use.csv", index_col=0).drop(index="전체")
    out = {}
    for p, row in r.iterrows():
        p = int(p)
        if row["voice"] > 0.9 and row["fixed"] < 0.5:
            out[p] = f"패턴 {p} 음성 중심"
        elif row["schedule"] > 0.9:
            out[p] = f"패턴 {p} 종일 제자리 + 스케줄"
        else:
            out[p] = f"패턴 {p} 저녁 제자리 + 이동/호출"
    return out


def hour_grid(df, feats, pattern=None):
    d = df if pattern is None else df[df["pattern"] == pattern]
    g = d.pivot_table(index="feature", columns="hr", values="share", aggfunc="sum")
    return g.reindex(index=feats, columns=range(24)).fillna(0)


def heat(ax, m, vmax_row, labels, title, show_y=True):
    norm = np.divide(m, vmax_row[:, None], out=np.zeros_like(m), where=vmax_row[:, None] > 0)
    norm = (
        norm * np.where(vmax_row < 0.01, 0.3, 1.0)[:, None]
    )  # 최대 1% 미만 기능은 흐리게 (표본이 작아 진하게 보이는 착시 방지)
    ax.imshow(norm, aspect="auto", cmap=SEQ, vmin=0, vmax=1)
    for h in SCHEDULE_HOURS:
        ax.axvline(h - 0.5, color=ss.ORANGE, lw=0.8, ls=":", alpha=0.8)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("시각")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if show_y else [""] * len(labels))
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)


def fig1():
    s = pd.read_csv(RES / "feature_scale.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    y = np.arange(len(s))[::-1]
    for ax, col, lab, fmt in [
        (axes[0], "device_share", "한 번이라도 쓴 기기 비율", "{:.0%}"),
        (axes[1], "device_days", "쓴 기기-일 수", "{:,.0f}"),
        (axes[2], "occ_hours_per_used_day", "쓴 날 평균 점유 시간 (h)", "{:.1f}"),
    ]:
        v = s[col].to_numpy()
        cols = [ss.BLUE if f in OCC else ss.GRAY for f in s["feature"]]
        ax.barh(y, v, color=cols, height=0.65)
        for yi, vi, f in zip(y, v, s["feature"]):
            if col == "occ_hours_per_used_day" and f not in OCC:
                ax.text(0.1, yi, "순간 기능", va="center", fontsize=8, color=ss.MUTED)
            else:
                ax.text(vi, yi, " " + fmt.format(vi), va="center", fontsize=8)
        ax.set_title(lab)
        ax.set_xlim(0, v.max() * 1.25)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s["name"])
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=ss.BLUE), plt.Rectangle((0, 0), 1, 1, color=ss.GRAY)],
        ["켜 두는 기능 (점유 시간)", "순간 기능 (시작만)"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ss.save(fig, "fig1_feature_scale")


def fig2():
    hs = pd.read_csv(RES / "overall_hour_share.csv", index_col=0).reindex(FEAT)
    use = pd.read_csv(RES / "overall_use_rate.csv", index_col=0)["use_rate"].reindex(FEAT)
    m = hs.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [4.2, 1]}, sharey=True)
    labels = [f"{NAME[f]} ({m[i].max():.0%})" for i, f in enumerate(FEAT)]
    heat(axes[0], m, m.max(1), labels, "기능별 시각 사용 비율 (기능마다 따로 칠함, 괄호 = 최대)")
    ax = axes[1]
    ax.barh(range(len(FEAT)), use.to_numpy(), color=ss.BLUE, height=0.6)
    for i, v in enumerate(use.to_numpy()):
        ax.text(v, i, f" {v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("그날 쓴 기기-일 비율")
    fig.tight_layout()
    ss.save(fig, "fig2_overall_hour")


def fig3():
    v = pd.read_csv(RES / "feature_variance_share.csv")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    cols = [ss.BLUE, ss.ORANGE, ss.GRAY, "#B8913A", "#8a9aa9", "#d4d4d8", "#2f3e4c", "#e4e4e7", "#c2410c"]
    for bi, (col, lab) in enumerate([("norm_share", "정규화 뒤"), ("raw_share", "정규화 전")]):
        left = 0.0
        for i, r in v.iterrows():
            ax.barh(bi, r[col], left=left, color=cols[i], edgecolor="white", height=0.62)
            if r[col] >= 0.07:
                ax.text(
                    left + r[col] / 2,
                    bi,
                    f"{r['name']}\n{r[col]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cols[i] not in ("#d4d4d8", "#e4e4e7") else ss.DARK,
                )
            left += r[col]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["정규화 뒤", "정규화 전"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("기기-일 행렬의 분산에서 각 기능이 차지하는 몫")
    ax.set_title("켜 두는 기능이 거리를 독차지하는가")
    fig.tight_layout()
    ss.save(fig, "fig3_variance_share")


def fig4():
    sw = pd.read_csv(RES / "k_sweep.csv")
    raw = pd.read_csv(RES / "k_sweep_raw.csv")
    k = int(pd.read_csv(RES / "k_choice.csv")["k"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    top = sw["silhouette"].max()
    ax.axhspan(top - 0.03, top, color=ss.LIGHT, label="최대값에서 0.03 안")
    ax.plot(sw["k"], sw["silhouette"], color=ss.BLUE, marker="o", lw=2, label="정규화 뒤")
    ax.plot(raw["k"], raw["silhouette"], color=ss.GRAY, marker="o", lw=1.5, ls="--", label="정규화 전")
    ax.scatter(
        [k], [sw.set_index("k").loc[k, "silhouette"]], s=140, facecolor="none", edgecolor=ss.ORANGE, lw=2, zorder=5
    )
    ax.annotate(
        f"선택 k={k}",
        (k, sw.set_index("k").loc[k, "silhouette"]),
        xytext=(10, -18),
        textcoords="offset points",
        color=ss.ORANGE,
        fontsize=9,
    )
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette")
    ax.set_ylim(0, max(0.4, raw["silhouette"].max() + 0.05))
    ax.set_title("silhouette: 모든 k 에서 0.25 아래")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(sw["k"], sw["inertia"] / 1000, color=ss.BLUE, marker="o", lw=2)
    ax.set_xlabel("k")
    ax.set_ylabel("inertia (천)")
    ax.set_title("inertia: elbow 없이 일정하게 감소")
    ax.grid(axis="y")
    fig.tight_layout()
    ss.save(fig, "fig4_k_sweep")


def fig5():
    hs = pd.read_csv(RES / "pattern_hour_share.csv")
    names = pat_name()
    pats = sorted(hs["pattern"].unique())
    grids = [hour_grid(hs, FEAT, p).to_numpy() for p in pats]
    vmax = np.max(np.stack(grids), axis=0).max(1)
    fig, axes = plt.subplots(1, len(pats), figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    for i, (p, m) in enumerate(zip(pats, grids)):
        heat(axes[i], m, vmax, FNAME, names[p], show_y=i == 0)
        if i:
            axes[i].set_yticks([])
        axes[i].title.set_color(PAT[p])
    fig.suptitle(
        "패턴별 시각 사용 비율 (기능마다 세 패턴 중 최대가 가장 진함, 점선 = 기본 스케줄 시각)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=ss.MUTED,
    )
    fig.tight_layout()
    ss.save(fig, "fig5_pattern_heatmaps")


def fig6():
    r = pd.read_csv(RES / "pattern_feature_use.csv", index_col=0)
    names = pat_name()
    rows = [names[int(p)] for p in r.index[:-1]] + ["전체"]
    m = r.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.imshow(m, aspect="auto", cmap=SEQ, vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j,
                i,
                f"{m[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if m[i, j] > 0.55 else ss.DARK,
            )
    ax.set_xticks(range(len(FEAT)))
    ax.set_xticklabels(FNAME, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.axhline(len(rows) - 1.5, color="white", lw=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("패턴별 기능 사용률 (그날 그 기능을 한 번이라도 쓴 기기-일 비율)")
    fig.tight_layout()
    ss.save(fig, "fig6_pattern_feature_use")


def fig7():
    st = pd.read_csv(RES / "seq_steps.csv", index_col=0)
    pr = pd.read_csv(RES / "seq_profile.csv", index_col=0)
    names = pat_name()
    pats = [c for c in st.columns if c != "전체"]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    x = st.index.to_numpy()
    for p in pats:
        ax.plot(x, st[p], color=PAT[int(p)], marker="o", lw=2, label=names[int(p)])
    ax.axvspan(4.5, 10.5, color=ss.LIGHT, zorder=0)
    ax.text(
        4.6,
        ax.get_ylim()[1] * 0.92,
        f"앞 4개로 자르면 잘리는 날 {pr.loc['전체', 'cut_by_trunc_share']:.1%}",
        fontsize=9,
        color=ss.MUTED,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) if v < 10 else "10+" for v in x])
    ax.set_xlabel("하루 사용 횟수 (기능을 새로 시작할 때마다 1회)")
    ax.set_ylabel("기기-일 비율")
    ax.set_title(
        f"하루 사용 횟수: 중앙값 {pr.loc['전체', 'uses_median']:.0f}회, 5회 넘는 날 {pr.loc['전체', 'over5_share']:.1%}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    ss.save(fig, "fig7_daily_uses")


def fig8():
    q = pd.read_csv(RES / "seq_top.csv")
    names = pat_name()
    pats = sorted(q["pattern"].unique())
    fig, axes = plt.subplots(len(pats), 1, figsize=(12, 2.0 * len(pats) + 0.6))
    for ax, p in zip(axes, pats):
        d = q[q["pattern"] == p].iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d["share"], color=PAT[p], height=0.6)
        for yi, (v, sq) in enumerate(zip(d["share"], d["seq_text"])):
            ax.text(v, yi, f" {v:.1%}  {sq}", va="center", fontsize=8.5)
        ax.set_yticks([])
        ax.set_xlim(0, 1.4)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{names[p]}: 하루 순서 상위 5 (자르지 않음, 같은 기능 연속은 ×n)", fontsize=10, color=PAT[p])
    fig.tight_layout()
    ss.save(fig, "fig8_day_sequences")


def fig9():
    t = pd.read_csv(RES / "seq_transitions.csv")
    names = pat_name()
    pats = sorted(t["pattern"].unique())
    fig, axes = plt.subplots(1, len(pats), figsize=(14, 3.6))
    for ax, p in zip(axes, pats):
        d = t[t["pattern"] == p].iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d["share"], color=PAT[p], height=0.6)
        for yi, v in zip(y, d["share"]):
            ax.text(v, yi, f" {v:.0%}", va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a} -> {b}" for a, b in zip(d["from"], d["to"])], fontsize=8.5)
        ax.set_xlim(0, d["share"].max() * 1.3)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(names[p], fontsize=10, color=PAT[p])
    fig.suptitle("다음에 쓰는 기능 (사용 1회 뒤의 사용 1회, 패턴 안 전이 중 비중)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    ss.save(fig, "fig9_transitions")


def fig10():
    s = pd.read_csv(RES / "pattern_share.csv").set_index("pattern")
    names = pat_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(s))
    w = 0.38
    ax.bar(x - w / 2, s["day_share"], w, color=[PAT[p] for p in s.index], label="기기-일 기준")
    ax.bar(x + w / 2, s["device_share"], w, color=[PAT[p] for p in s.index], alpha=0.45, label="기기 기준 (주 패턴)")
    for xi, (a, b) in enumerate(zip(s["day_share"], s["device_share"])):
        ax.text(xi - w / 2, a, f"{a:.0%}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, b, f"{b:.0%}", ha="center", va="bottom", fontsize=9, color=ss.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [" ".join(names[p].split(" ")[:2]) + "\n" + names[p].split(" ", 2)[2] for p in s.index], fontsize=8.5
    )
    ax.set_ylabel("비중")
    ax.set_ylim(0, max(s["day_share"].max(), s["device_share"].max()) * 1.25)
    ax.legend(loc="upper right")
    ax.set_title("기기-일 기준과 기기 기준의 순위가 다르다")
    ax = axes[1]
    ax.barh(x[::-1], s["days_per_device"], color=[PAT[p] for p in s.index], height=0.55)
    for yi, v, c in zip(x[::-1], s["days_per_device"], s["consistency"]):
        ax.text(v, yi, f" {v:.1f}일, 일관성 {c:.2f}", va="center", fontsize=9)
    ax.set_yticks(x[::-1])
    ax.set_yticklabels([f"패턴 {p}" for p in s.index])
    ax.set_xlim(0, s["days_per_device"].max() * 1.8)
    ax.set_xlabel("주 패턴 기기 1대당 기기-일 (평일 20일 중)")
    ax.set_title("기기당 날 수와 일관성")
    fig.tight_layout()
    ss.save(fig, "fig10_pattern_share")


def fig11():
    sm = pd.read_csv(RES / "seq_similarity.csv").set_index("pattern")
    names = pat_name()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(sm))
    w = 0.36
    ax.bar(x - w / 2, sm["within_device"], w, color=[PAT[p] for p in sm.index], label="같은 기기의 두 날")
    ax.bar(x + w / 2, sm["between_device"], w, color=ss.GRAY, label="같은 패턴 다른 기기의 두 날")
    for xi, (a, b) in enumerate(zip(sm["within_device"], sm["between_device"])):
        ax.text(xi - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9, color=ss.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([names[p] for p in sm.index], fontsize=9)
    ax.set_ylabel("전이 집합 Jaccard")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left")
    ax.set_title("하루 순서는 같은 기기끼리 조금 더 닮았을 뿐이다")
    fig.tight_layout()
    ss.save(fig, "fig11_sequence_similarity")


def fig12():
    sil = pd.read_csv(RES / "silhouette_by_unit.csv")
    xt = pd.read_csv(RES / "device_type_x_pattern.csv", index_col=0)
    names = pat_name()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    ax.bar(sil["unit"], sil["silhouette"], color=[ss.GRAY, ss.BLUE], width=0.55)
    for xi, (v, k) in enumerate(zip(sil["silhouette"], sil["k"])):
        ax.text(xi, v, f"{v:.2f} (k={k})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("silhouette")
    ax.set_title("분석 단위를 기기로 바꾸면 묶임이 뚜렷해진다")
    ax = axes[1]
    y = np.arange(len(xt))[::-1]
    left = np.zeros(len(xt))
    for c in xt.columns:
        p = int(c)
        color = ss.LIGHT if p == 0 else PAT[p]
        v = xt[c].to_numpy()
        ax.barh(
            y,
            v,
            left=left,
            color=color,
            height=0.6,
            edgecolor="white",
            label="군집 대상 날 없음" if p == 0 else names[p],
        )
        for yi, l0, vi in zip(y, left, v):
            if vi >= 0.08:
                ax.text(
                    l0 + vi / 2,
                    yi,
                    f"{vi:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if p else ss.DARK,
                )
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"유형 {i}" for i in xt.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("유형 안 기기의 주 패턴 구성")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    ax.set_title("기기 유형 x 기기-일 주 패턴")
    fig.tight_layout()
    ss.save(fig, "fig12_unit_compare")


def fig13():
    ph = pd.read_csv(RES / "personal_hours.csv")
    vh = pd.read_csv(RES / "personal_voice_hour.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(ph))[::-1]
    ax.barh(y + 0.18, ph["within_sd_h"], 0.34, color=ss.BLUE, label="기기 안 (날마다 흔들림)")
    ax.barh(y - 0.18, ph["between_sd_h"], 0.34, color=ss.GRAY, label="기기 사이 (기기마다 다름)")
    for yi, a, b in zip(y, ph["within_sd_h"], ph["between_sd_h"]):
        ax.text(a, yi + 0.18, f" {a:.1f}", va="center", fontsize=8.5)
        ax.text(b, yi - 0.18, f" {b:.1f}", va="center", fontsize=8.5, color=ss.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(ph["name"])
    ax.set_xlabel("하루 첫 사용 시각의 표준편차 (시간)")
    ax.set_xlim(0, ph[["within_sd_h", "between_sd_h"]].to_numpy().max() * 1.3)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("기능별 첫 사용 시각 (주 패턴 날만)")
    ax = axes[1]
    ax.bar(vh["voice_median_block"], vh["devices"], width=1.6, color=ss.ORANGE)
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("기기의 음성 대화 첫 사용 시각 (중앙값, 2시간 구간)")
    ax.set_ylabel("기기 수")
    ax.set_title("음성을 쓰는 시각은 기기마다 다르다")
    fig.tight_layout()
    ss.save(fig, "fig13_personal_hours")


def fig14():
    s = pd.read_csv(RES / "eval_schedule_heuristic.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    rows = [
        ("실제로 스케줄을 켠 기기", s["devices_truth_on"], ss.ORANGE),
        ("휴리스틱: 20일 중 한 번이라도 잡힌 기기", s["devices_any_schedule"], ss.GRAY),
        ("휴리스틱: 20일 중 10일 이상 잡힌 기기", s["devices_schedule_half_days"], ss.BLUE),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c) in zip(y, rows):
        ax.barh(yi, v, color=c, height=0.55)
        ax.text(v, yi, f" {v:.1%}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 0.6)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"기본 스케줄 휴리스틱: 이벤트 정밀도 {s['precision']:.1%}여도 기기 비율은 부푼다")
    fig.tight_layout()
    ss.save(fig, "fig14_schedule_heuristic")


def fig15():
    ev = pd.read_csv(RES / "eval_summary.csv")
    a = ev[ev["what"].str.startswith("기기")].copy()
    a["ari"] = a["ari"].astype(float)
    norm = pd.read_csv(RES / "eval_pattern_x_state.csv", index_col=0)
    raw = pd.read_csv(RES / "eval_raw_pattern_x_state.csv", index_col=0)
    st = {
        "S1": ("종일 제자리 + 스케줄", ss.ORANGE),
        "S2": ("저녁 제자리 + 이동/호출", ss.BLUE),
        "S3": ("음성 중심", "#B8913A"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.35})
    ax = axes[0]
    y = np.arange(len(a))[::-1]
    ax.barh(y, a["ari"], color=[ss.BLUE, ss.GRAY, ss.BLUE, ss.BLUE][: len(a)], height=0.55)
    for yi, v in zip(y, a["ari"]):
        ax.text(v, yi, f" {v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([w.replace(" vs ", "\nvs ") for w in a["what"]], fontsize=8.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("ARI (가상데이터에 넣은 구조와 일치도)")
    ax.set_title("ground truth 를 얼마나 되찾나")
    for ax, t, title in [(axes[1], norm, "정규화 뒤 패턴의 실제 상태"), (axes[2], raw, "정규화 전 패턴의 실제 상태")]:
        yy = np.arange(len(t))[::-1]
        left = np.zeros(len(t))
        for s in ["S1", "S2", "S3"]:
            v = t[s].to_numpy()
            ax.barh(yy, v, left=left, color=st[s][1], height=0.6, edgecolor="white", label=st[s][0])
            for yi, l0, vi in zip(yy, left, v):
                if vi >= 0.1:
                    ax.text(l0 + vi / 2, yi, f"{vi:.0%}", ha="center", va="center", fontsize=8.5, color="white")
            left += v
        ax.set_yticks(yy)
        ax.set_yticklabels([f"패턴 {p}" for p in t.index])
        ax.set_xlim(0, 1)
        ax.set_title(title)
    axes[2].legend(loc="upper center", bbox_to_anchor=(-0.1, -0.12), ncol=3, fontsize=8.5)
    ss.save(fig, "fig15_eval")


if __name__ == "__main__":
    ss.setup()
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15]:
        f()
