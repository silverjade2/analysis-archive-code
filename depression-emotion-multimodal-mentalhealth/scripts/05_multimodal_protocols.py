"""감정 분류. multimodal vs 텍스트, 프로토콜 3개

(a) 원 방식. 텍스트는 전체 19374건, multimodal은 2000건 랜덤 subset, 각자 70/30 (테스트셋이 다름)
(b) 같은 2000건, 같은 랜덤 분할
(c) 화자 분리 분할(GroupShuffleSplit), 전체와 2000건
+ 화자 사전확률 baseline, oracle 상한, N x 분할 x 모델 곡선 (seed 3개)
"""

import numpy as np
import pandas as pd
from common import (
    AUDIO_COLS,
    C_AUDIO,
    C_DARK,
    C_GRAY,
    C_MM,
    C_TEXT,
    DATA,
    EMOTIONS,
    FIGURES,
    N_MULTIMODAL,
    RESULTS,
    SEED,
    TEST_SIZE,
    setup_mpl,
)
from models import AudioClf, MultimodalClf, TextClf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

plt = setup_mpl()
df = pd.read_csv(DATA / "utterances.csv", low_memory=False)
daily = df[df.source == "daily"].reset_index(drop=True)


def split(d, how, seed):
    if how == "random":
        tr, te = train_test_split(d, test_size=TEST_SIZE, stratify=d.emotion_label, random_state=seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        i_tr, i_te = next(gss.split(d, groups=d.speaker_id))
        tr, te = d.iloc[i_tr], d.iloc[i_te]
    return tr, te


def run(tr, te, model, seed):
    y_tr, y_te = tr.emotion_label, te.emotion_label
    if model == "text":
        p = TextClf(seed).fit(tr.text, y_tr).predict(te.text)
    elif model == "audio":
        p = AudioClf(seed).fit(tr[AUDIO_COLS].values, y_tr).predict(te[AUDIO_COLS].values)
    elif model == "multimodal":
        p = MultimodalClf(seed).fit(tr.text, tr[AUDIO_COLS].values, y_tr).predict(te.text, te[AUDIO_COLS].values)
    elif model == "speaker_prior":
        maj = tr.groupby("speaker_id").emotion_label.agg(lambda s: s.value_counts().index[0])
        glob = tr.emotion_label.value_counts().index[0]
        p = te.speaker_id.map(maj).fillna(glob).values
    return dict(accuracy=accuracy_score(y_te, p), f1_weighted=f1_score(y_te, p, average="weighted"), pred=p)


def oracle_pred(te, rng):
    """텍스트 신호 있으면 정답, 아니면 음성 신호 있으면 정답(neutral 제외), 둘 다 없으면 happiness. 관대한 상한"""
    out = []
    for _, r in te.iterrows():
        if r.truth_text_informative_emo == 1:
            out.append(r.emotion_label)
        elif r.truth_audio_informative == 1 and r.emotion_label != "neutral":
            out.append(r.emotion_label)
        else:
            out.append("happiness")
    return np.array(out)


rows = []
sub2000 = daily.sample(N_MULTIMODAL, random_state=SEED)

tr_all, te_all = split(daily, "random", SEED)
r = run(tr_all, te_all, "text", SEED)
rows.append(
    dict(
        protocol="a_original",
        data="all_19374",
        split="random",
        model="text",
        n_train=len(tr_all),
        n_test=len(te_all),
        accuracy=r["accuracy"],
        f1_weighted=r["f1_weighted"],
    )
)
tr_s, te_s = split(sub2000, "random", SEED)
for model in ["multimodal", "speaker_prior"]:
    r = run(tr_s, te_s, model, SEED)
    rows.append(
        dict(
            protocol="a_original",
            data="sub_2000",
            split="random",
            model=model,
            n_train=len(tr_s),
            n_test=len(te_s),
            accuracy=r["accuracy"],
            f1_weighted=r["f1_weighted"],
        )
    )

for model in ["text", "audio", "multimodal"]:
    r = run(tr_s, te_s, model, SEED)
    rows.append(
        dict(
            protocol="b_same_subset",
            data="sub_2000",
            split="random",
            model=model,
            n_train=len(tr_s),
            n_test=len(te_s),
            accuracy=r["accuracy"],
            f1_weighted=r["f1_weighted"],
        )
    )

for data_name, d in [("all_19374", daily), ("sub_2000", sub2000)]:
    tr, te = split(d, "speaker", SEED)
    for model in ["text", "audio", "multimodal", "speaker_prior"]:
        r = run(tr, te, model, SEED)
        rows.append(
            dict(
                protocol="c_speaker_split",
                data=data_name,
                split="speaker",
                model=model,
                n_train=len(tr),
                n_test=len(te),
                accuracy=r["accuracy"],
                f1_weighted=r["f1_weighted"],
            )
        )
    if data_name == "all_19374":
        te_c_all, pred_c_mm = te, run(tr, te, "multimodal", SEED)["pred"]
        pred_c_text = run(tr, te, "text", SEED)["pred"]
# 전체 랜덤 분할 multimodal. (a) 텍스트와 같은 테스트셋
r = run(tr_all, te_all, "multimodal", SEED)
rows.append(
    dict(
        protocol="c_random_all",
        data="all_19374",
        split="random",
        model="multimodal",
        n_train=len(tr_all),
        n_test=len(te_all),
        accuracy=r["accuracy"],
        f1_weighted=r["f1_weighted"],
    )
)
r = run(tr_all, te_all, "speaker_prior", SEED)
rows.append(
    dict(
        protocol="c_random_all",
        data="all_19374",
        split="random",
        model="speaker_prior",
        n_train=len(tr_all),
        n_test=len(te_all),
        accuracy=r["accuracy"],
        f1_weighted=r["f1_weighted"],
    )
)

rng = np.random.default_rng(SEED)
for name, te in [("all_19374_random", te_all), ("sub_2000_random", te_s), ("all_19374_speaker", te_c_all)]:
    op = oracle_pred(te, rng)
    rows.append(
        dict(
            protocol="oracle",
            data=name,
            split=name.split("_")[-1],
            model="oracle",
            n_train=0,
            n_test=len(te),
            accuracy=accuracy_score(te.emotion_label, op),
            f1_weighted=f1_score(te.emotion_label, op, average="weighted"),
        )
    )

res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "multimodal_protocols.csv", index=False)
print(res.to_string())

per = []
for e in EMOTIONS:
    m = (te_c_all.emotion_label == e).values
    per.append(
        dict(
            emotion=e,
            n_test=int(m.sum()),
            text_recall=(pred_c_text[m] == e).mean(),
            multimodal_recall=(pred_c_mm[m] == e).mean(),
        )
    )
per = pd.DataFrame(per)
per["gain"] = per.multimodal_recall - per.text_recall
per.round(4).to_csv(RESULTS / "multimodal_per_class_gain_speaker_split.csv", index=False)
print(per.round(3).to_string())

curve = []
for n in [500, 1000, 2000, 5000, 10000, len(daily)]:
    for s in range(3):
        seed = SEED + s
        d = daily if n == len(daily) else daily.sample(n, random_state=seed)
        for how in ["random", "speaker"]:
            tr, te = split(d, how, seed)
            for model in ["text", "multimodal"]:
                r = run(tr, te, model, seed)
                curve.append(dict(n=n, seed=seed, split=how, model=model, accuracy=r["accuracy"]))
curve = pd.DataFrame(curve)
agg = curve.groupby(["n", "split", "model"]).accuracy.agg(["mean", "std"]).round(4).reset_index()
agg.to_csv(RESULTS / "multimodal_n_curve.csv", index=False)
print(agg.to_string())


def g(protocol, data, model, col="accuracy"):
    return res[(res.protocol == protocol) & (res.data == data) & (res.model == model)][col].iloc[0]


arrows = [
    (
        "(a) 원 방식\n텍스트 전체 -> 멀티모달 2,000\n(테스트셋이 다름)",
        g("a_original", "all_19374", "text"),
        g("a_original", "sub_2000", "multimodal"),
        f"n={g('a_original', 'all_19374', 'text', 'n_test'):,} "
        f"vs {g('a_original', 'sub_2000', 'multimodal', 'n_test'):,}",
        C_GRAY,
    ),
    (
        "(b) 같은 2,000건, 랜덤",
        g("b_same_subset", "sub_2000", "text"),
        g("b_same_subset", "sub_2000", "multimodal"),
        f"n={g('b_same_subset', 'sub_2000', 'text', 'n_test'):,}",
        C_MM,
    ),
    (
        "(c) 같은 2,000건, 화자 분리",
        g("c_speaker_split", "sub_2000", "text"),
        g("c_speaker_split", "sub_2000", "multimodal"),
        f"n={g('c_speaker_split', 'sub_2000', 'text', 'n_test'):,}",
        C_AUDIO,
    ),
    (
        "전체, 랜덤",
        g("a_original", "all_19374", "text"),
        g("c_random_all", "all_19374", "multimodal"),
        f"n={g('c_random_all', 'all_19374', 'multimodal', 'n_test'):,}",
        C_MM,
    ),
    (
        "(c) 전체, 화자 분리",
        g("c_speaker_split", "all_19374", "text"),
        g("c_speaker_split", "all_19374", "multimodal"),
        f"n={g('c_speaker_split', 'all_19374', 'text', 'n_test'):,}",
        C_AUDIO,
    ),
]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (lab, a, b, n, col) in enumerate(arrows[::-1]):
    ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=16))
    ax.scatter([a], [i], color=C_TEXT, s=70, zorder=3, edgecolor=C_DARK, lw=0.5)
    ax.text(a - 0.004, i + 0.28, f"{a:.3f}", ha="right", fontsize=8.5, color=C_DARK)
    ax.text(b + 0.004, i + 0.28, f"{b:.3f}  (Δ{b - a:+.3f})", ha="left", fontsize=8.5, color=col)
    ax.text(0.855, i - 0.02, n, va="center", fontsize=8, color=C_GRAY)
ax.set_yticks(range(len(arrows)))
ax.set_yticklabels([a[0] for a in arrows[::-1]], fontsize=9)
ax.set_xlim(0.74, 0.87)
ax.set_ylim(-0.6, len(arrows) - 0.2)
ax.set_xlabel("accuracy   (● 텍스트 -> ▶ 멀티모달)")
ax.set_title("프로토콜별 accuracy, 텍스트 vs 멀티모달 (가상데이터)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "fig4_protocols.png")
plt.close()

gap = agg.pivot_table(index=["n", "split"], columns="model", values="mean").reset_index()
gap["gain"] = gap.multimodal - gap.text
gap.round(4).to_csv(RESULTS / "multimodal_n_curve_gain.csv", index=False)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
style = {
    ("random", "text"): (C_TEXT, "--"),
    ("random", "multimodal"): (C_MM, "--"),
    ("speaker", "text"): (C_TEXT, "-"),
    ("speaker", "multimodal"): (C_AUDIO, "-"),
}
for (how, model), (c, ls) in style.items():
    a = agg[(agg.split == how) & (agg.model == model)]
    ax1.plot(
        a.n,
        a["mean"],
        color=c,
        ls=ls,
        marker="o",
        ms=4,
        label=f"{model}, {'랜덤' if how == 'random' else '화자 분리'}",
    )
    ax1.fill_between(a.n, a["mean"] - a["std"], a["mean"] + a["std"], color=c, alpha=0.15)
ax1.set_xscale("log")
ax1.set_ylabel("accuracy (시드 3개, 밴드 = ±1 std)")
ax1.legend(fontsize=8, ncol=2, loc="lower right")
ax1.axvline(2000, color=C_GRAY, ls=":", lw=1)
ax1.text(2000, ax1.get_ylim()[1] - 0.004, " 원 프로젝트의 멀티모달 N", fontsize=8, color=C_GRAY, va="top")
ax1.spines[["top", "right"]].set_visible(False)
for how, c in [("random", C_MM), ("speaker", C_AUDIO)]:
    gg = gap[gap.split == how]
    ax2.plot(
        gg.n,
        gg.gain,
        color=c,
        marker="o",
        ms=4,
        ls="--" if how == "random" else "-",
        label="랜덤 분할" if how == "random" else "화자 분리",
    )
ax2.axhline(0, color=C_DARK, lw=0.8)
ax2.set_ylabel("멀티모달 - 텍스트")
ax2.set_xlabel("학습에 쓴 발화 수 N (log)")
ax2.legend(fontsize=8)
ax2.spines[["top", "right"]].set_visible(False)
ax1.set_title("N과 분할 방식별 accuracy, 멀티모달 - 텍스트")
plt.tight_layout()
plt.savefig(FIGURES / "fig5_n_curve.png")
plt.close()
