"""EDA. 출처별 문체 통계, 상담 의도 분포, 감정 라벨 분포, 화자 편중, 음성 feature PCA."""

import pandas as pd
from common import (
    AUDIO_COLS,
    C_DARK,
    C_GRAY,
    C_MM,
    C_NEG,
    DATA,
    EMO_COLORS,
    EMOTIONS,
    FIGURES,
    RESULTS,
    SEED,
    setup_mpl,
)
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt = setup_mpl()
df = pd.read_csv(DATA / "utterances.csv", low_memory=False)

df["n_char"] = df.text.str.len()
df["polite_ending"] = df.text.str.endswith(("요", "습니다")).astype(int)
style = (
    df.groupby("source")
    .agg(
        n=("utt_id", "size"),
        mean_char=("n_char", "mean"),
        polite_rate=("polite_ending", "mean"),
        truth_depressed_rate=("truth_depressed", "mean"),
    )
    .round(3)
)
style.to_csv(RESULTS / "eda_source_style.csv")

intent = df[df.source == "counsel"].intent.value_counts().rename_axis("intent").reset_index(name="n")
intent["is_symptom"] = intent.intent.str.startswith("정신증상").astype(int)
intent.to_csv(RESULTS / "eda_counsel_intent.csv", index=False)

daily = df[df.source == "daily"]
emo = daily.emotion_label.value_counts(normalize=True).reindex(EMOTIONS).rename("share").round(4)
emo.to_csv(RESULTS / "eda_emotion_share.csv")

spk = daily.groupby("speaker_id").emotion_label.agg(
    n="size", top_share=lambda s: s.value_counts(normalize=True).iloc[0]
)
spk_summary = pd.DataFrame(
    {
        "n_speakers": [spk.shape[0]],
        "utt_per_speaker_median": [spk.n.median()],
        "top_emotion_share_median": [spk.top_share.median().round(3)],
        "top_emotion_share_p90": [spk.top_share.quantile(0.9).round(3)],
    }
)
spk_summary.to_csv(RESULTS / "eda_speaker_bias.csv", index=False)

# fig1. 원 프로젝트의 학습 데이터에서 어느 칸이 비어 있었는지
counsel = df[df.source == "counsel"]
n_sym = int((counsel.intent.str.startswith("정신증상")).sum())
n_norm = len(counsel) - n_sym
cells = pd.DataFrame(
    [
        dict(source="counsel", label="1 (우울)", n=n_sym, used_in_training=1),
        dict(source="counsel", label="0 (우울 아님)", n=n_norm, used_in_training=0),
        dict(source="daily", label="1 (우울)", n=0, used_in_training=0),
        dict(source="daily", label="0 (우울 아님)", n=len(daily), used_in_training=1),
    ]
)
cells.to_csv(RESULTS / "eda_label_source_cells.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
W = {"counsel": len(counsel), "daily": len(daily)}
total = sum(W.values())
x = 0
for src in ["counsel", "daily"]:
    w = W[src] / total
    sub = cells[cells.source == src]
    y = 0
    for _, r in sub.iterrows():
        h = r.n / W[src]
        if r.n == 0:
            continue
        col = C_MM if r.used_in_training else C_NEG
        ax.add_patch(plt.Rectangle((x, y), w - 0.02, h - 0.02, color=col, alpha=0.9))
        ax.text(
            x + w / 2 - 0.01,
            y + h / 2 - 0.01,
            f"{r.label}\n{r.n:,}건" + ("\n학습에서 제외" if not r.used_in_training else ""),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
        y += h
    if src == "daily":
        ax.add_patch(plt.Rectangle((x, 0.985), w - 0.02, 0.0, color="none"))
        ax.text(
            x + w / 2 - 0.01,
            1.03,
            "라벨 1 · 0건\n(이 칸이 비어 있다)",
            ha="center",
            va="bottom",
            color=C_DARK,
            fontsize=9,
        )
    ax.text(
        x + w / 2 - 0.01,
        -0.06,
        "상담 스크립트" if src == "counsel" else "일상 대화",
        ha="center",
        va="top",
        fontsize=10,
    )
    x += w
ax.set_xlim(0, 1)
ax.set_ylim(-0.12, 1.16)
ax.axis("off")
ax.set_title("원 프로젝트의 학습 데이터: 라벨 × 출처")

ax = axes[1]
order = emo.sort_values(ascending=True)
ax.hlines(order.index, 0, order.values, color=C_GRAY, lw=2)
ax.scatter(order.values, order.index, color=[EMO_COLORS[e] for e in order.index], s=120, zorder=3)
for e, v in order.items():
    ax.text(v + 0.006, e, f"{v:.1%}", va="center", fontsize=9)
ax.set_xlim(0, 0.32)
ax.set_xlabel("share")
ax.set_title("일상 대화 감정 라벨 비중 (가상데이터)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "fig1_data_structure.png")
plt.close()

# fig7. 같은 점을 화자 색과 감정 색으로 두 번 그린다. 음성 feature가 감정보다 화자를 먼저 가르는지 보려는 것
top_spk = daily.speaker_id.value_counts().index[:8]
sub = daily[daily.speaker_id.isin(top_spk)].sample(1200, random_state=SEED)
Z = PCA(n_components=2, random_state=SEED).fit(StandardScaler().fit_transform(daily[AUDIO_COLS]))
ev = Z.explained_variance_ratio_
P = Z.transform(StandardScaler().fit(daily[AUDIO_COLS]).transform(sub[AUDIO_COLS]))
pd.DataFrame(dict(component=["PC1", "PC2"], explained_variance_ratio=ev.round(4))).to_csv(
    RESULTS / "eda_audio_pca.csv", index=False
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
spk_colors = dict(zip(top_spk, plt.cm.tab10.colors[:8]))
axes[0].scatter(P[:, 0], P[:, 1], c=[spk_colors[s] for s in sub.speaker_id], s=10, alpha=0.7)
axes[0].set_title("음성 피처 PCA — 화자별 색 (상위 8명)")
axes[1].scatter(P[:, 0], P[:, 1], c=[EMO_COLORS[e] for e in sub.emotion_label], s=10, alpha=0.7)
axes[1].set_title("같은 점 — 감정별 색")
for ax in axes:
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(f"PC2 ({ev[1]:.0%})")
axes[1].legend(
    handles=[Line2D([], [], marker="o", ls="", color=c, label=e) for e, c in EMO_COLORS.items()],
    fontsize=7,
    loc="lower right",
    ncol=2,
)
plt.tight_layout()
plt.savefig(FIGURES / "fig7_audio_pca_speaker_vs_emotion.png")
plt.close()
print(style)
print(spk_summary)
print(cells)
