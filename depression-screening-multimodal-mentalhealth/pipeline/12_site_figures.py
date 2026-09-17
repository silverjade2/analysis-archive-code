"""사이트용 그림 8장. 값은 11과 같은 CSV. 예외는 fig5 PCA 좌표 (같은 피처, seed로 재계산)"""

import _path  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DATA, EMOTIONS, PHQ_CUTOFF, RESULTS, SEED, TASK_KO, TASKS
from evalutil import load_recordings
from matplotlib.patches import Patch
from sitestyle import BLUE, DARK, GRAY, LIGHT, LIGHT_BLUE, MUTED, RED, SEQUENTIAL, save, setup
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

setup()
R = RESULTS

coh = pd.read_csv(DATA / "participants.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
rng = np.random.default_rng(0)
for gi, grp in enumerate(["control", "patient"]):
    sub = coh[coh.group == grp]
    x = gi + rng.uniform(-0.28, 0.28, len(sub))
    ax1.scatter(x, sub.phq9, c=np.where(sub.phq9 >= PHQ_CUTOFF, RED, GRAY), s=8, alpha=0.5, lw=0)
ax1.axhline(PHQ_CUTOFF, color=DARK, ls="--", lw=1)
ax1.text(1.62, PHQ_CUTOFF, f"PHQ-9 = {PHQ_CUTOFF}\n라벨 cutoff", ha="left", va="center", fontsize=8.5, color=DARK)
ax1.set_xlim(-0.55, 2.15)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["대조군", "환자군"])
ax1.set_ylabel("PHQ-9")
ax1.set_title("그룹별 PHQ-9 분포")
ct = pd.read_csv(R / "eda_group_by_label.csv").set_index("group")
cats = ["대조군\n(우울 아님)", "대조군\n(우울)", "환자군\n(우울 아님)", "환자군\n(우울)"]
vals = [
    ct.loc["control", "not_depressed"],
    ct.loc["control", "depressed"],
    ct.loc["patient", "not_depressed"],
    ct.loc["patient", "depressed"],
]
bars = ax2.bar(cats, vals, color=[GRAY, RED, GRAY, RED])
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 8, str(int(v)), ha="center", fontsize=9, color=DARK)
ax2.set_title("그룹 × 라벨 교차표")
ax2.set_ylabel("참가자 수")
ax2.tick_params(axis="x", labelsize=8)
fig.tight_layout()
save(fig, "fig1_cohort")

sc = pd.read_csv(R / "split_comparison.csv")


def gv(split, model, col="par_auc_mean"):
    return sc[(sc.split == split) & (sc.model == model)][col].iloc[0]


models = [("text", "텍스트"), ("audio_gbm", "음성 GBM"), ("audio_knn", "음성 k-NN"), ("fusion", "융합")]
fig, ax = plt.subplots(figsize=(9, 4.3))
for i, (m, name) in enumerate(models[::-1]):
    a, b = gv("recording_random", m), gv("participant", m)
    ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5, mutation_scale=15))
    ax.scatter([a], [i], color=BLUE, s=55, zorder=3, edgecolor="white", lw=1)
    if a - b < 0.01:
        ax.text(
            a + 0.004,
            i,
            f"recording random {a:.3f} -> speaker-independent {b:.3f} (Δ{b - a:+.3f})",
            ha="left",
            va="center",
            fontsize=8,
            color=DARK,
        )
    else:
        ax.text(a + 0.002, i + 0.26, f"recording random {a:.3f}", ha="left", fontsize=8, color=MUTED)
        ax.text(
            b - 0.002, i + 0.26, f"speaker-independent {b:.3f}  (Δ{b - a:+.3f})", ha="right", fontsize=8, color=DARK
        )
    ax.text(0.813, i, name, va="center", ha="left", fontsize=10, color=DARK)
sp = gv("recording_random", "speaker_prior")
ax.axvline(sp, color=MUTED, ls=":", lw=1)
ax.text(sp, 3.6, f"speaker-lookup baseline {sp:.2f}", ha="right", fontsize=8, color=MUTED)
ax.set_ylim(-0.5, 3.9)
ax.set_xlim(0.81, 1.02)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("참가자 단위 AUC")
ax.set_title("분할 방식에 따른 참가자 단위 AUC (시드 3개 평균)")
fig.tight_layout()
save(fig, "fig2_split_leakage")

ta = pd.read_csv(R / "task_ablation.csv")
piv = ta.pivot_table(index="task_ko", columns="model", values="auc").reindex([TASK_KO[t] for t in TASKS])[
    ["text", "audio", "fusion"]
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
im = ax1.imshow(piv.values, cmap=SEQUENTIAL, vmin=0.5, vmax=0.9, aspect="auto")
ax1.set_xticks(range(3))
ax1.set_xticklabels(["텍스트", "음성", "융합"])
ax1.set_yticks(range(len(piv)))
ax1.set_yticklabels(piv.index, fontsize=9)
for i in range(len(piv)):
    for j in range(3):
        v = piv.values[i, j]
        ax1.text(
            j,
            i,
            "-" if (j == 0 and v == 0.5) else f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if v > 0.72 else DARK,
        )
ax1.set_title("과제 × modality AUC (speaker-independent, 단일 과제)")
ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
agg = pd.read_csv(R / "task_aggregation.csv")
names = {
    "all_8_tasks": "8과제 전체",
    "narrative_5": "free-speech 5과제",
    "memory_2": "기억 2과제",
    "fixed_text_3": "read-speech 3과제",
}
agg = agg.set_index("aggregation").reindex(["all_8_tasks", "narrative_5", "memory_2", "fixed_text_3"])
ax2.barh([names[i] for i in agg.index][::-1], agg.par_auc[::-1], color=[BLUE, BLUE, BLUE, GRAY][::-1])
for i, v in enumerate(agg.par_auc[::-1]):
    ax2.text(v - 0.01, i, f"{v:.3f}", ha="right", va="center", color="white", fontsize=9)
ax2.set_xlim(0.5, 0.97)
ax2.set_xlabel("참가자 단위 AUC")
ax2.set_title("집계 범위별 AUC (융합, speaker-independent)")
fig.tight_layout()
save(fig, "fig3_task_modality")

ld = pd.read_csv(R / "label_definition_comparison.csv")
q = pd.read_csv(R / "label_definition_prob_quantiles.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
groups = [
    ("control", 0, "대조군, 우울 아님"),
    ("patient", 0, "환자군, 우울 아님\n(remission)"),
    ("patient", 1, "환자군, 우울"),
]
for ax, trained, title in [(axes[0], "group_label", "환자군 여부로 학습"), (axes[1], "phq_label", "PHQ-9 라벨로 학습")]:
    for i, (g, lab, name) in enumerate(groups):
        sub = q[(q.trained_on == trained) & (q.group == g) & (q.label_depressed == lab)].sort_values("quantile")
        col = RED if lab == 1 else (GRAY if g == "control" else BLUE)
        ax.plot([i, i], [sub.prob.iloc[0], sub.prob.iloc[-1]], color=col, lw=1.2, zorder=1)
        ax.scatter([i] * len(sub), sub.prob, color=col, s=[20, 35, 60, 35, 20], zorder=2)
        med = sub[sub["quantile"] == 0.5].prob.iloc[0]
        ax.scatter([i], [med], color=col, s=90, edgecolor="white", lw=1.5, zorder=3)
        ax.text(i, -0.06, name, ha="center", fontsize=8.5, color=DARK)
    ax.axhline(0.5, color=DARK, ls="--", lw=0.8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.12, 1.0)
    ax.set_xticks([])
    ax.set_title(title)
    ax.spines["bottom"].set_visible(False)
axes[0].set_ylabel("우울 예측 확률 (분위 10, 25, 50, 75, 90)")
pnd_g = ld[(ld.trained_on == "group_label") & (ld.tokens == "full")].mean_prob_patient_not_depressed.iloc[0]
pnd_p = ld[(ld.trained_on == "phq_label") & (ld.tokens == "full")].mean_prob_patient_not_depressed.iloc[0]
axes[0].text(1, 0.92, f"remission 환자 평균 {pnd_g:.2f}", ha="center", fontsize=8, color=BLUE)
axes[1].text(1, 0.92, f"remission 환자 평균 {pnd_p:.2f}", ha="center", fontsize=8, color=BLUE)
fig.tight_layout()
save(fig, "fig4_label_definition")

df = load_recordings()
top_spk = df.participant_id.unique()[:25]
sub = df[df.participant_id.isin(top_spk)]
TIMBRE = ["f0_mean"] + [f"mfcc{i}" for i in range(1, 7)]
sc0 = StandardScaler().fit(df[TIMBRE])
Zt = PCA(2, random_state=SEED).fit(sc0.transform(df[TIMBRE]))
P = Zt.transform(sc0.transform(sub[TIMBRE]))
ev = Zt.explained_variance_ratio_
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
palette = dict(zip(top_spk, list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)))
axes[0].scatter(P[:, 0], P[:, 1], c=[palette[s] for s in sub.participant_id], s=12, alpha=0.7, lw=0)
axes[0].set_title("음색 계열 feature PCA, 참가자별 색 (25명 × 8건)")
axes[1].scatter(P[:, 0], P[:, 1], c=[RED if v else GRAY for v in sub.label_depressed], s=12, alpha=0.6, lw=0)
axes[1].set_title("같은 점, 우울 라벨별 색")
va = pd.read_csv(R / "eda_audio_variance_decomposition.csv")
axes[1].text(
    0.03,
    0.03,
    f"화자 η² 최대 {va.eta2_speaker.max():.2f}\n라벨 η² 최대 {va.eta2_label.max():.2f}",
    transform=axes[1].transAxes,
    fontsize=8,
    va="bottom",
    color=DARK,
    bbox=dict(boxstyle="round", fc="white", ec=LIGHT),
)
for ax in axes:
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
axes[0].set_ylabel(f"PC2 ({ev[1]:.0%})")
fig.tight_layout()
save(fig, "fig5_audio_pca")

cm = pd.read_csv(R / "emotion_confusion.csv", index_col=0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
cmn = cm.div(cm.sum(1), axis=0)
ax1.imshow(cmn.values, cmap=SEQUENTIAL, vmin=0, vmax=1)
ax1.set_xticks(range(len(EMOTIONS)))
ax1.set_xticklabels(EMOTIONS, fontsize=9)
ax1.set_yticks(range(len(EMOTIONS)))
ax1.set_yticklabels(EMOTIONS, fontsize=9)
for i in range(len(EMOTIONS)):
    for j in range(len(EMOTIONS)):
        v = cmn.values[i, j]
        if v > 0.02:
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.45 else DARK)
ax1.set_xlabel("예측")
ax1.set_ylabel("실제")
ax1.set_title("감정 분류 confusion matrix (행 정규화)")
ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
ps = pd.read_csv(R / "emotion_transfer_participant_scores.csv")
ax2.scatter(ps.phq, ps.neg, c=[RED if v else GRAY for v in ps.y], s=10, alpha=0.5, lw=0)
z = np.polyfit(ps.phq, ps.neg, 1)
xs = np.array([ps.phq.min(), ps.phq.max()])
ax2.plot(xs, z[0] * xs + z[1], color=DARK, lw=1.5)
tr = pd.read_csv(R / "emotion_transfer.csv")
tr_p = tr[(tr.signal.str.contains("transfer")) & (tr.level == "participant")]
ax2.set_xlabel("PHQ-9")
ax2.set_ylabel("감정 모델의 부정 정서 점수\nP(슬픔)+P(불안)+P(상처)")
ax2.set_title(f"transfer 점수와 PHQ-9 (Spearman {tr_p.spearman_phq.iloc[0]:.2f}, 참가자 AUC {tr_p.auc.iloc[0]:.2f})")
fig.tight_layout()
save(fig, "fig6_emotion")

tf = pd.read_csv(R / "top_features_phq_label_kinds.csv")
tf = tf[tf.coef > 0].copy()


def stem(f):
    return " ".join(t.split("/")[0] for t in f.split())


tf["label"] = tf.feature.map(stem)
tf = tf.drop_duplicates("label").head(14)
KC = {"머뭇거림": GRAY, "치료맥락": RED, "증상": BLUE, "무쾌감/무망": DARK, "긍정회상": LIGHT_BLUE, "기타": MUTED}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax1.barh(range(len(tf))[::-1], tf.coef, color=[KC.get(k, MUTED) for k in tf.kind])
ax1.set_yticks(range(len(tf))[::-1])
ax1.set_yticklabels(tf.label, fontsize=9)
ax1.set_xlabel("logistic regression 계수 (우울 방향)")
ax1.set_title("PHQ-9 라벨 텍스트 모델의 상위 feature")
seen = list(dict.fromkeys(tf.kind))
ax1.legend(handles=[Patch(color=KC.get(k, MUTED), label=k) for k in seen], fontsize=8, loc="lower right")
pairs = [
    (
        "환자군 라벨",
        ld[(ld.trained_on == "group_label") & (ld.tokens == "full")].par_auc_vs_phq.iloc[0],
        ld[(ld.trained_on == "group_label") & (ld.tokens == "treatment_masked")].par_auc_vs_phq.iloc[0],
    ),
    (
        "PHQ-9 라벨",
        ld[(ld.trained_on == "phq_label") & (ld.tokens == "full")].par_auc_vs_phq.iloc[0],
        ld[(ld.trained_on == "phq_label") & (ld.tokens == "treatment_masked")].par_auc_vs_phq.iloc[0],
    ),
]
for i, (name, full, masked) in enumerate(pairs[::-1]):
    ax2.plot([full, masked], [i, i], color=GRAY, lw=2, zorder=1)
    ax2.scatter([full], [i], color=RED, s=90, zorder=2, label="원본" if i == 0 else None)
    ax2.scatter([masked], [i], color=BLUE, s=90, zorder=2, label="치료어 마스킹" if i == 0 else None)
    ax2.text(full - 0.002, i, f"{full:.3f}", ha="right", va="center", fontsize=8, color=RED)
    ax2.text(masked + 0.002, i, f"{masked:.3f}", ha="left", va="center", fontsize=8, color=BLUE)
ax2.set_yticks(range(2))
ax2.set_yticklabels([p[0] for p in pairs[::-1]])
ax2.set_xlabel("PHQ-9 기준 참가자 AUC")
ax2.set_title("치료 어휘 마스킹 전후")
ax2.legend(fontsize=8, loc="lower center")
ax2.set_xlim(0.865, 0.955)
ax2.set_ylim(-0.6, 1.6)
fig.tight_layout()
save(fig, "fig7_text_features")

sg = pd.read_csv(R / "subgroup_auc.csv")
sg = sg[sg.model == "fusion"]
order = [
    ("all", "all", "전체"),
    ("subgroup_employed", "직장인", "직장인"),
    ("subgroup_employed", "비직장인", "비직장인"),
    ("sex", "F", "여성"),
    ("sex", "M", "남성"),
    ("group", "patient", "환자군 내"),
    ("group", "control", "대조군 내"),
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
rows_ = []
for dim, s, name in order:
    r = sg[(sg.dimension == dim) & (sg.subgroup == s)]
    if len(r):
        rows_.append((name, r.auc.iloc[0], r.ci_low.iloc[0], r.ci_high.iloc[0], int(r.n.iloc[0])))
for i, (name, a, lo, hi, n) in enumerate(rows_[::-1]):
    ax1.plot([lo, hi], [i, i], color=DARK, lw=1.5)
    ax1.scatter([a], [i], color=BLUE, s=70, zorder=3)
    ax1.text(hi + 0.004, i, f"{a:.2f}", va="center", fontsize=8, color=DARK)
    ax1.text(0.79, i, f"{name} (n={n:,})", va="center", ha="left", fontsize=9, color=DARK)
ax1.axvline(sg[sg.subgroup == "all"].auc.iloc[0], color=GRAY, ls="--", lw=1)
ax1.set_xlim(0.79, 1.0)
ax1.set_ylim(-0.5, len(rows_) - 0.5)
ax1.set_yticks([])
ax1.spines["left"].set_visible(False)
ax1.set_xlabel("참가자 단위 AUC (95% CI)")
ax1.set_title("융합 모델 하위군 AUC")
cal = pd.read_csv(R / "calibration_deciles.csv")
ax2.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
ax2.plot(cal.mean_pred, cal.observed, color=BLUE, marker="o", ms=6)
for _, r in cal.iterrows():
    ax2.annotate(
        f"{r.mean_phq:.0f}",
        (r.mean_pred, r.observed),
        fontsize=7,
        color=DARK,
        xytext=(3, -8),
        textcoords="offset points",
    )
ax2.set_xlabel("예측 확률 (10분위 평균)")
ax2.set_ylabel("실제 우울 비율")
ax2.set_title("calibration (숫자는 분위 평균 PHQ-9)")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
fig.tight_layout()
save(fig, "fig8_subgroup_calibration")
