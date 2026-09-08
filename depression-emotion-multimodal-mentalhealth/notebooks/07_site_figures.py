"""07. 그림 1~8을 사이트 톤으로 다시 그린다 (Pretendard, 사이트 팔레트, webp).

형식과 축·데이터·숫자는 02/04/05/06이 그린 그림과 같다. 값은 outputs/results/*.csv에서 읽고
다시 계산하지 않는다. 예외는 fig7의 PCA 좌표 하나로, 02와 같은 코드·시드로 다시 구한 뒤
설명 분산이 eda_audio_pca.csv와 같은지 assert 한다. 출력: outputs/figures/site/figN_*.webp (+ .png)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, DATA, RESULTS, EMOTIONS, AUDIO_COLS
from sitestyle import (setup, save, BLUE, ORANGE, GRAY, LIGHT, DARK, LIGHT_BLUE, LIGHT_ORANGE, MUTED, RED,
                       CAT8, DIVERGING)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

setup()
EMO_KO = {"happiness": "happiness", "angry": "angry", "neutral": "neutral", "sadness": "sadness",
          "disgust": "disgust", "surprise": "surprise", "fear": "fear"}

# ---------------------------------------------------------------- fig1 라벨×출처 모자이크 + 감정 비중 (02)
cells = pd.read_csv(RESULTS / "eda_label_source_cells.csv")
emo = pd.read_csv(RESULTS / "eda_emotion_share.csv").set_index("emotion_label")["share"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
W = cells.groupby("source").n.sum(); total = W.sum(); x = 0
for src in ["counsel", "daily"]:
    w = W[src] / total
    sub = cells[cells.source == src]; y = 0
    for _, r in sub.iterrows():
        h = r.n / W[src]
        if r.n == 0:
            continue
        col = BLUE if r.used_in_training else ORANGE
        ax.add_patch(Rectangle((x, y), w - 0.02, h - 0.02, color=col))
        ax.text(x + w / 2 - 0.01, y + h / 2 - 0.01,
                f"{r.label}\n{r.n:,}건" + ("\n학습에서 제외" if not r.used_in_training else ""),
                ha="center", va="center", color="white", fontsize=9.5, linespacing=1.5)
        y += h
    if src == "daily":
        ax.text(x + w / 2 - 0.01, 1.03, "라벨 1 · 0건\n(이 칸이 비어 있다)", ha="center", va="bottom", color=DARK, fontsize=9.5)
    ax.text(x + w / 2 - 0.01, -0.06, "상담 스크립트" if src == "counsel" else "일상 대화", ha="center", va="top", fontsize=10.5, color=DARK)
    x += w
ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1.16); ax.axis("off")
ax.set_title("원 프로젝트의 학습 데이터: 라벨 × 출처")
ax.legend(handles=[Patch(color=BLUE, label="학습에 사용"), Patch(color=ORANGE, label="학습에서 제외")],
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2)
ax = axes[1]
order = emo.sort_values(ascending=True)
ax.hlines(order.index, 0, order.values, color=LIGHT, lw=2.5)
ax.scatter(order.values, order.index, color=BLUE, s=90, zorder=3)
for e, v in order.items():
    ax.text(v + 0.008, e, f"{v:.1%}", va="center", fontsize=9, color=DARK)
ax.set_xlim(0, 0.32); ax.set_xlabel("share"); ax.set_title("일상 대화 감정 라벨 비중 (가상데이터)")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="x")
fig.tight_layout()
save(fig, "fig1_data_structure")

# ---------------------------------------------------------------- fig2 와플 + 덤벨 (04)
grp = pd.read_csv(RESULTS / "depression_v1_by_group.csv")
v1m = pd.read_csv(RESULTS / "depression_v1_metrics.csv")
cmp_ = pd.read_csv(RESULTS / "depression_v1_v2_compare.csv")
def get(v, sub, col="accuracy"):
    return cmp_[(cmp_.version == v) & (cmp_.subset == sub)][col].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
rate = grp.iloc[0].pred_depressed_rate; k = int(round(rate * 100))
for i in range(100):
    r_, c_ = divmod(i, 10)
    ax.add_patch(Rectangle((c_, 9 - r_), 0.9, 0.9, color=ORANGE if i < k else LIGHT))
ax.set_xlim(-0.2, 10.2); ax.set_ylim(-0.2, 10.2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("학습에서 뺀 상담 일상 발화 4,000건을 v1에 넣으면")
ax.text(5, -0.9, f"{rate:.1%}를 우울로 예측 (칸 하나 = 1%)", ha="center", fontsize=9.5, color=DARK)
ax = axes[1]
rows_d = [("전체 테스트 (11,813)", get("v1_exclude_normal", "test_all"), get("v2_include_normal", "test_all")),
          ("상담 발화만 (6,000)", get("v1_exclude_normal", "test_counsel_only"), get("v2_include_normal", "test_counsel_only")),
          ("일상 대화만 (5,813)", get("v1_exclude_normal", "test_daily_only"), get("v2_include_normal", "test_daily_only"))]
for i, (lab, a, b) in enumerate(rows_d[::-1]):
    ax.plot([a, b], [i, i], color=LIGHT, lw=3.5, zorder=1, solid_capstyle="round")
    ax.scatter([a], [i], color=GRAY, s=90, zorder=3, label="v1 (일상 의도 제외)" if i == 0 else None)
    ax.scatter([b], [i], color=BLUE, s=90, zorder=3, label="v2 (일상 의도 = 0)" if i == 0 else None)
    ax.text(a - 0.006, i + 0.22, f"{a:.3f}", ha="right", fontsize=9, color=MUTED)
    ax.text(b + 0.006, i + 0.22, f"{b:.3f}", ha="left", fontsize=9, color=BLUE)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows_d[::-1]])
orig = v1m[v1m.scoring == "train_label"].accuracy.iloc[0]
ax.axvline(orig, color=MUTED, ls="--", lw=1.2)
ax.text(orig, 2.55, f"v1 · 원 범위\n학습 라벨 기준 {orig:.3f}", ha="center", fontsize=8.5, color=DARK, linespacing=1.4)
ax.set_xlim(0.78, 1.02); ax.set_ylim(-0.5, 3.0); ax.set_xlabel("accuracy (실제 우울 기준)")
ax.legend(loc="lower left"); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="x")
ax.set_title("같은 테스트셋, 학습에 4,000건을 넣었는지만 다르다")
fig.tight_layout()
save(fig, "fig2_depression_scoring")

# ---------------------------------------------------------------- fig3 토큰 종류별 발산 막대 (04)
kinds = pd.read_csv(RESULTS / "depression_feature_kinds.csv")
KC = {"어미": GRAY, "관용어": ORANGE, "내용": BLUE}
v1f = kinds[kinds.version == "v1"].reset_index(drop=True)
v2f = kinds[kinds.version == "v2"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, r in v1f.iterrows():
    ax.barh(11 - i, -r.coef, color=KC[r.kind], height=0.72)
    ax.text(-r.coef - 0.15, 11 - i, f"'{r.feature.strip()}'", ha="right", va="center", fontsize=9, color=DARK)
for i, r in v2f.iterrows():
    ax.barh(11 - i, r.coef, color=KC[r.kind], height=0.72)
    ax.text(r.coef + 0.15, 11 - i, f"'{r.feature.strip()}'", ha="left", va="center", fontsize=9, color=DARK)
ax.axvline(0, color=DARK, lw=1); ax.set_yticks([]); ax.set_xlim(-12, 12)
ax.set_xticks([-8, -4, 0, 4, 8]); ax.set_xticklabels(["8", "4", "0", "4", "8"]); ax.set_xlabel("|LR coef| (char n-gram, 우울 방향)")
ax.text(-6, 12.3, "v1 · 상담 일상 의도 제외", ha="center", fontsize=10.5, color=DARK)
ax.text(6, 12.3, "v2 · 상담 일상 의도 = 0", ha="center", fontsize=10.5, color=DARK)
ax.legend(handles=[Patch(color=c, label=k) for k, c in KC.items()], loc="lower right", title="토큰 종류")
ax.spines["left"].set_visible(False); ax.set_ylim(-0.8, 13); ax.grid(axis="x")
save(fig, "fig3_depression_features")

# ---------------------------------------------------------------- fig4 프로토콜별 화살표 (05)
res = pd.read_csv(RESULTS / "multimodal_protocols.csv")
def g(protocol, data, model, col="accuracy"):
    return res[(res.protocol == protocol) & (res.data == data) & (res.model == model)][col].iloc[0]
arrows = [
    ("(a) 원 방식\n텍스트 전체 → 멀티모달 2,000\n(테스트셋이 다름)", g("a_original", "all_19374", "text"), g("a_original", "sub_2000", "multimodal"),
     f"n={g('a_original','all_19374','text','n_test'):,} vs {g('a_original','sub_2000','multimodal','n_test'):,}", GRAY),
    ("(b) 같은 2,000건 · 랜덤", g("b_same_subset", "sub_2000", "text"), g("b_same_subset", "sub_2000", "multimodal"),
     f"n={g('b_same_subset','sub_2000','text','n_test'):,}", ORANGE),
    ("(c) 같은 2,000건 · 화자 분리", g("c_speaker_split", "sub_2000", "text"), g("c_speaker_split", "sub_2000", "multimodal"),
     f"n={g('c_speaker_split','sub_2000','text','n_test'):,}", BLUE),
    ("전체 · 랜덤", g("a_original", "all_19374", "text"), g("c_random_all", "all_19374", "multimodal"),
     f"n={g('c_random_all','all_19374','multimodal','n_test'):,}", ORANGE),
    ("(c) 전체 · 화자 분리", g("c_speaker_split", "all_19374", "text"), g("c_speaker_split", "all_19374", "multimodal"),
     f"n={g('c_speaker_split','all_19374','text','n_test'):,}", BLUE),
]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (lab, a, b, n, col) in enumerate(arrows[::-1]):
    ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=16))
    ax.scatter([a], [i], s=70, facecolor="white", edgecolor=DARK, linewidth=1.2, zorder=3)
    ax.text(a - 0.004, i + 0.28, f"{a:.3f}", ha="right", fontsize=8.5, color=DARK)
    ax.text(b + 0.004, i + 0.28, f"{b:.3f}  (Δ{b - a:+.3f})", ha="left", fontsize=8.5, color=col, fontweight="semibold")
    ax.text(0.855, i - 0.02, n, va="center", fontsize=8, color=MUTED)
ax.set_yticks(range(len(arrows))); ax.set_yticklabels([a[0] for a in arrows[::-1]], fontsize=9)
ax.set_xlim(0.74, 0.87); ax.set_ylim(-0.6, len(arrows) - 0.2); ax.set_xlabel("accuracy   (● 텍스트 → ▶ 멀티모달)")
ax.set_title("멀티모달이 텍스트에 더한 것: 비교 조건에 따라 달라진다 (가상데이터)")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); ax.grid(axis="x")
ax.legend(handles=[Line2D([], [], color=GRAY, lw=2.5, label="테스트셋이 다른 비교"),
                   Line2D([], [], color=ORANGE, lw=2.5, label="랜덤 분할"),
                   Line2D([], [], color=BLUE, lw=2.5, label="화자 분리")],
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
save(fig, "fig4_protocols")

# ---------------------------------------------------------------- fig5 N 곡선 밴드 + 이득 패널 (05)
agg = pd.read_csv(RESULTS / "multimodal_n_curve.csv")
gap = pd.read_csv(RESULTS / "multimodal_n_curve_gain.csv")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
style = {("random", "text"): (GRAY, "--", "텍스트 · 랜덤"), ("random", "multimodal"): (BLUE, "--", "멀티모달 · 랜덤"),
         ("speaker", "text"): (GRAY, "-", "텍스트 · 화자 분리"), ("speaker", "multimodal"): (BLUE, "-", "멀티모달 · 화자 분리")}
for (how, model), (c, ls, lab) in style.items():
    a = agg[(agg.split == how) & (agg.model == model)]
    ax1.plot(a.n, a["mean"], color=c, ls=ls, marker="o", ms=4, lw=1.7, label=lab)
    ax1.fill_between(a.n, a["mean"] - a["std"], a["mean"] + a["std"], color=c, alpha=0.12, lw=0)
ax1.set_xscale("log"); ax1.set_ylabel("accuracy (시드 3개, 밴드 = ±1 std)"); ax1.legend(ncol=2, loc="lower right")
ax1.axvline(2000, color=MUTED, ls=":", lw=1)
ax1.text(2000, ax1.get_ylim()[1] - 0.004, " 원 프로젝트의 멀티모달 N", fontsize=8.5, color=MUTED, va="top")
ax1.grid(axis="y")
for how, c, ls, lab in [("random", ORANGE, "--", "랜덤 분할"), ("speaker", BLUE, "-", "화자 분리")]:
    gg = gap[gap.split == how]
    ax2.plot(gg.n, gg.gain, color=c, marker="o", ms=4, ls=ls, lw=1.7, label=lab)
ax2.axhline(0, color=DARK, lw=0.8); ax2.set_ylabel("멀티모달 − 텍스트"); ax2.set_xlabel("학습에 쓴 발화 수 N (log)")
ax2.legend(); ax2.grid(axis="y")
ax1.set_title("N과 분할 방식에 따른 accuracy, 그리고 음성이 더한 것")
fig.tight_layout()
save(fig, "fig5_n_curve")

# ---------------------------------------------------------------- fig6 레이더 + 혼동행렬 차이 히트맵 (06)
per = pd.read_csv(RESULTS / "multimodal_per_class_gain_speaker_split.csv").set_index("emotion").loc[EMOTIONS]
diff = pd.read_csv(RESULTS / "confusion_diff_multimodal_minus_text.csv", index_col=0).loc[EMOTIONS, EMOTIONS]
ang = np.linspace(0, 2 * np.pi, len(EMOTIONS), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(11.5, 4.8))
ax = fig.add_subplot(1, 2, 1, polar=True)
for col, c, lab in [("text_recall", GRAY, "텍스트"), ("multimodal_recall", BLUE, "멀티모달")]:
    v = per[col].tolist(); v += v[:1]
    ax.plot(ang, v, color=c, lw=2, label=lab); ax.fill(ang, v, color=c, alpha=0.12)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(EMOTIONS, fontsize=9, color=DARK)
ax.set_ylim(0.6, 0.95); ax.set_yticks([0.7, 0.8, 0.9]); ax.set_yticklabels(["0.7", "0.8", "0.9"], fontsize=7, color=MUTED)
ax.spines["polar"].set_color(LIGHT); ax.grid(color=LIGHT)
ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1)); ax.set_title("감정별 recall (화자 분리, 전체)", pad=14)
ax2 = fig.add_subplot(1, 2, 2)
lim = np.abs(diff.values).max()
im = ax2.imshow(diff.values, cmap=DIVERGING, vmin=-lim, vmax=lim)
ax2.set_xticks(range(7)); ax2.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=8)
ax2.set_yticks(range(7)); ax2.set_yticklabels(EMOTIONS, fontsize=8)
for i in range(7):
    for j in range(7):
        v = diff.values[i, j]
        if v != 0:
            ax2.text(j, i, f"{v:+d}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > lim * 0.55 else DARK)
ax2.set_xlabel("예측"); ax2.set_ylabel("정답"); ax2.set_title("혼동행렬 차이: 멀티모달 − 텍스트 (건수)")
for s in ("top", "right", "left", "bottom"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(length=0)
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
fig.tight_layout()
save(fig, "fig6_audio_gain_by_class")

# ---------------------------------------------------------------- fig7 음성 피처 PCA: 화자 색 vs 감정 색 (02와 같은 코드·시드)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(DATA / "utterances.csv", low_memory=False)
daily = df[df.source == "daily"]
top_spk = daily.speaker_id.value_counts().index[:8]
sub = daily[daily.speaker_id.isin(top_spk)].sample(1200, random_state=SEED)
Z = PCA(n_components=2, random_state=SEED).fit(StandardScaler().fit_transform(daily[AUDIO_COLS]))
ev = Z.explained_variance_ratio_
ev_saved = pd.read_csv(RESULTS / "eda_audio_pca.csv")["explained_variance_ratio"].values
assert np.allclose(ev.round(4), ev_saved), (ev, ev_saved)
P = Z.transform(StandardScaler().fit(daily[AUDIO_COLS]).transform(sub[AUDIO_COLS]))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
spk_colors = dict(zip(top_spk, CAT8))
emo_colors = dict(zip(EMOTIONS, CAT8))
axes[0].scatter(P[:, 0], P[:, 1], c=[spk_colors[s] for s in sub.speaker_id], s=10, alpha=0.75, lw=0)
axes[0].set_title("음성 피처 PCA: 화자별 색 (상위 8명)")
axes[1].scatter(P[:, 0], P[:, 1], c=[emo_colors[e] for e in sub.emotion_label], s=10, alpha=0.75, lw=0)
axes[1].set_title("같은 점: 감정별 색")
for ax in axes:
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.grid()
axes[0].set_ylabel(f"PC2 ({ev[1]:.0%})")
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in spk_colors.items()], fontsize=7, loc="lower right", ncol=2)
axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=e) for e, c in emo_colors.items()], fontsize=7, loc="lower right", ncol=2)
fig.tight_layout()
save(fig, "fig7_audio_pca_speaker_vs_emotion")

# ---------------------------------------------------------------- fig8 로컬 토큰 칩 + 전역 상위 피처 (06)
local = pd.read_csv(RESULTS / "shap_like_local_example.csv")
tok_contrib = list(zip(local.token, local.sadness_contribution))
true_emo, true_dep = local.true_emotion.iloc[0], int(local.truth_depressed.iloc[0])
glob_sad = pd.read_csv(RESULTS / "emotion_top_features_by_class.csv")
glob_sad = glob_sad[glob_sad.emotion == "sadness"].copy()
glob_sad["stem"] = glob_sad.feature.str.strip()
glob_sad = glob_sad.sort_values("coef", ascending=False).drop_duplicates("stem").head(8)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.4), gridspec_kw={"height_ratios": [1, 1.6]})
lim = max(abs(c) for _, c in tok_contrib) or 1
x = 0.02
for t, c in tok_contrib:
    col = DIVERGING(0.5 + 0.5 * c / lim)
    w = 0.012 * max(len(t), 2) + 0.03
    ax1.add_patch(Rectangle((x, 0.35), w, 0.4, facecolor=col, edgecolor=LIGHT, lw=0.8))
    ax1.text(x + w / 2, 0.55, t, ha="center", va="center", fontsize=10, color="white" if abs(c) / lim > 0.45 else DARK)
    ax1.text(x + w / 2, 0.2, f"{c:+.2f}", ha="center", va="center", fontsize=8.5, color=DARK)
    x += w + 0.012
ax1.set_xlim(0, max(1, x)); ax1.set_ylim(0, 1); ax1.axis("off")
ax1.set_title(f"로컬: 예문 한 건의 토큰별 sadness 기여 (정답 감정 = {true_emo}, truth_depressed = {true_dep})", fontsize=10.5)
ax2.barh([f"'{f.strip()}'" for f in glob_sad.feature[::-1]], glob_sad.coef[::-1], color=BLUE, height=0.68)
ax2.set_xlabel("LR coef (sadness, char n-gram)"); ax2.set_title("전역: sadness 상위 피처 (8위 안, 같은 계수는 하나로). '외롭', '무의미'는 없다", fontsize=10.5)
ax2.tick_params(axis="y", length=0); ax2.spines["left"].set_visible(False); ax2.grid(axis="x")
fig.tight_layout()
save(fig, "fig8_local_vs_global")
print("font:", plt.rcParams["font.family"])
