"""해석. 원 프로젝트는 SHAP 텍스트 플롯 예문 1건으로 "슬픔은 외로운, 무의미에서 온다"를 보였음

- 감정별 global 상위 feature, 예문 토큰('외롭', '무의미')이 sadness 발화 중 몇 %에 있는지
- 융합 분류기의 채널별 가중치 합, 화자 분리 분할 혼동행렬 차이
- SHAP 텍스트 플롯 흉내: char n-gram tfidf x sadness 계수를 어절 단위로 합산
"""

import numpy as np
import pandas as pd
from common import (
    AUDIO_COLS,
    C_AUDIO,
    C_DARK,
    C_TEXT,
    DATA,
    EMO_COLORS,
    EMOTIONS,
    FIGURES,
    RESULTS,
    SEED,
    TEST_SIZE,
    setup_mpl,
)
from models import MultimodalClf, TextClf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

plt = setup_mpl()
df = pd.read_csv(DATA / "utterances.csv", low_memory=False)
daily = df[df.source == "daily"].reset_index(drop=True)
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
i_tr, i_te = next(gss.split(daily, groups=daily.speaker_id))
tr, te = daily.iloc[i_tr], daily.iloc[i_te]

m = TextClf(SEED).fit(tr.text, tr.emotion_label)
feats = m.top_features(30)
rows = []
for e in EMOTIONS:
    seen = set()
    k = 0
    for f, c in feats[e]:
        key = round(c, 3)
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(emotion=e, rank=k + 1, feature=f, coef=c))
        k += 1
        if k == 8:
            break
pd.DataFrame(rows).to_csv(RESULTS / "emotion_top_features_by_class.csv", index=False)

tok = ["외롭", "무의미"]
cov = []
for t in tok:
    has = daily.text.str.contains(t)
    cov.append(
        dict(
            token=t,
            share_of_sadness_utterances_containing=has[daily.emotion_label == "sadness"].mean(),
            share_of_all_utterances_containing=has.mean(),
            sadness_share_given_token=(daily.emotion_label[has] == "sadness").mean(),
            depressed_share_given_token=daily.truth_depressed[has].mean(),
        )
    )
cov = pd.DataFrame(cov).round(4)
cov.to_csv(RESULTS / "shap_example_token_coverage.csv", index=False)

mm = MultimodalClf(SEED).fit(tr.text, tr[AUDIO_COLS].values, tr.emotion_label)
t_norm, a_norm = mm.channel_coef_norm()
pd.DataFrame(
    [
        dict(channel="text", abs_coef_sum=t_norm, share=t_norm / (t_norm + a_norm)),
        dict(channel="audio", abs_coef_sum=a_norm, share=a_norm / (t_norm + a_norm)),
    ]
).round(4).to_csv(RESULTS / "fusion_channel_weight.csv", index=False)

pred_text = TextClf(SEED).fit(tr.text, tr.emotion_label).predict(te.text)
pred_mm = mm.predict(te.text, te[AUDIO_COLS].values)
cm_t = confusion_matrix(te.emotion_label, pred_text, labels=EMOTIONS)
cm_m = confusion_matrix(te.emotion_label, pred_mm, labels=EMOTIONS)
diff = pd.DataFrame(cm_m - cm_t, index=EMOTIONS, columns=EMOTIONS)
diff.to_csv(RESULTS / "confusion_diff_multimodal_minus_text.csv")
pd.DataFrame(cm_t, index=EMOTIONS, columns=EMOTIONS).to_csv(RESULTS / "confusion_text_speaker_split.csv")
pd.DataFrame(
    [
        dict(target=e, diag_change=int(diff.loc[e, e]), pulled_from_other_classes=int(diff[e].drop(e).sum()))
        for e in EMOTIONS
    ]
).to_csv(RESULTS / "confusion_diff_summary.csv", index=False)
pd.DataFrame(cm_m, index=EMOTIONS, columns=EMOTIONS).to_csv(RESULTS / "confusion_multimodal_speaker_split.csv")

per = pd.read_csv(RESULTS / "multimodal_per_class_gain_speaker_split.csv").set_index("emotion").loc[EMOTIONS]
ang = np.linspace(0, 2 * np.pi, len(EMOTIONS), endpoint=False).tolist()
ang += ang[:1]
fig = plt.figure(figsize=(11.5, 4.8))
ax = fig.add_subplot(1, 2, 1, polar=True)
for col, c, lab in [("text_recall", C_TEXT, "텍스트"), ("multimodal_recall", C_AUDIO, "멀티모달")]:
    v = per[col].tolist()
    v += v[:1]
    ax.plot(ang, v, color=c, lw=2, label=lab)
    ax.fill(ang, v, color=c, alpha=0.15)
ax.set_xticks(ang[:-1])
ax.set_xticklabels(EMOTIONS, fontsize=9)
ax.set_ylim(0.6, 0.95)
ax.set_yticks([0.7, 0.8, 0.9])
ax.set_yticklabels(["0.7", "0.8", "0.9"], fontsize=7)
ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=9)
ax.set_title("감정별 recall (화자 분리, 전체)", pad=14)
ax2 = fig.add_subplot(1, 2, 2)
lim = np.abs(diff.values).max()
im = ax2.imshow(diff.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax2.set_xticks(range(7))
ax2.set_xticklabels(EMOTIONS, rotation=40, ha="right", fontsize=8)
ax2.set_yticks(range(7))
ax2.set_yticklabels(EMOTIONS, fontsize=8)
for i in range(7):
    for j in range(7):
        v = diff.values[i, j]
        if v != 0:
            ax2.text(
                j,
                i,
                f"{v:+d}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(v) > lim * 0.55 else C_DARK,
            )
ax2.set_xlabel("예측")
ax2.set_ylabel("정답")
ax2.set_title("혼동행렬 차이: 멀티모달 - 텍스트 (건수)")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
plt.tight_layout()
plt.savefig(FIGURES / "fig6_audio_gain_by_class.png")
plt.close()

ex_pool = daily[daily.text.str.contains("외롭") & daily.text.str.contains("무의미")]
if len(ex_pool) == 0:
    ex_pool = daily[daily.text.str.contains("외롭")]
ex = ex_pool.iloc[0]
vec, clf = m.vec, m.clf
names = vec.get_feature_names_out()
sad_i = list(clf.classes_).index("sadness")
X = vec.transform([ex.text]).tocoo()
contrib = {names[j]: X.data[k] * clf.coef_[sad_i, j] for k, j in enumerate(X.col)}
tokens = ex.text.split(" ")
tok_contrib = []
for t in tokens:
    c = sum(v for f, v in contrib.items() if f.strip() and f.strip() in t)
    tok_contrib.append((t, c))
local = pd.DataFrame(tok_contrib, columns=["token", "sadness_contribution"]).round(4)
local["example_text"] = ex.text
local["true_emotion"] = ex.emotion_label
local["truth_depressed"] = ex.truth_depressed
local.to_csv(RESULTS / "shap_like_local_example.csv", index=False)
glob_sad = pd.read_csv(RESULTS / "emotion_top_features_by_class.csv")
glob_sad = glob_sad[glob_sad.emotion == "sadness"].copy()
glob_sad["stem"] = glob_sad.feature.str.strip()
glob_sad = glob_sad.sort_values("coef", ascending=False).drop_duplicates("stem").head(8)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.4), gridspec_kw={"height_ratios": [1, 1.6]})
lim = max(abs(c) for _, c in tok_contrib) or 1
x = 0.02
for t, c in tok_contrib:
    col = plt.cm.RdBu_r(0.5 + 0.5 * c / lim)
    w = 0.012 * max(len(t), 2) + 0.03
    ax1.add_patch(plt.Rectangle((x, 0.35), w, 0.4, color=col, alpha=0.95))
    ax1.text(
        x + w / 2, 0.55, t, ha="center", va="center", fontsize=10, color="white" if abs(c) / lim > 0.45 else C_DARK
    )
    ax1.text(x + w / 2, 0.2, f"{c:+.2f}", ha="center", va="center", fontsize=8, color=C_DARK)
    x += w + 0.012
ax1.set_xlim(0, max(1, x))
ax1.set_ylim(0, 1)
ax1.axis("off")
ax1.set_title(
    f"로컬: 예문 한 건의 토큰별 sadness 기여 (정답 감정 = {ex.emotion_label}, truth_depressed = {ex.truth_depressed})",
    fontsize=10,
    loc="left",
)
ax2.barh([repr(f.strip()) for f in glob_sad.feature[::-1]], glob_sad.coef[::-1], color=EMO_COLORS["sadness"])
hl = [i for i, f in enumerate(glob_sad.feature[::-1]) if "외롭" in f or "무의미" in f]
ax2.set_xlabel("LR coef (sadness, char n-gram)")
ax2.set_title("전역: sadness 상위 피처 8개", fontsize=10, loc="left")
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "fig8_local_vs_global.png")
plt.close()
print(local.to_string())
print(diff)
