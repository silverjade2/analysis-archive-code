"""우울 진단 v2. 버린 4,000건을 되살린다.

상담 스크립트의 일상 의도 발화를 라벨 0으로 학습에 넣는다. 증상 의도는 1, 일상 대화는 0. 일상 대화 안의 실제
우울 발화는 원 데이터에 라벨이 없으므로 0으로 남는다. 라벨 노이즈다. 03과 같은 테스트셋에서 v1과 비교한다.
"""

import numpy as np
import pandas as pd
from common import (
    C_AUDIO,
    C_DARK,
    C_GRAY,
    C_MM,
    C_NEG,
    C_TEXT,
    DATA,
    FIGURES,
    RESULTS,
    SEED,
    depression_split,
    setup_mpl,
)
from matplotlib.patches import Patch
from models import TextClf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

plt = setup_mpl()
df = pd.read_csv(DATA / "utterances.csv", low_memory=False)
df["is_symptom"] = df.intent.fillna("").str.startswith("정신증상").astype(int)
df["label"] = df.is_symptom
tr, te = depression_split(df)
tr_v1 = tr[~((tr.source == "counsel") & (tr.is_symptom == 0))]


def metrics(y, p):
    return dict(
        accuracy=accuracy_score(y, p),
        precision=precision_score(y, p, zero_division=0),
        recall=recall_score(y, p, zero_division=0),
        f1=f1_score(y, p, zero_division=0),
    )


rows = []
for name, train in [("v1_exclude_normal", tr_v1), ("v2_include_normal", tr)]:
    m = TextClf(SEED).fit(train.text, train.label)
    pred = m.predict(te.text)
    proba = m.predict_proba(te.text)[:, 1]
    for subset, mask in [
        ("test_all", np.ones(len(te), bool)),
        ("test_counsel_only", (te.source == "counsel").values),
        ("test_daily_only", (te.source == "daily").values),
    ]:
        rows.append(
            dict(
                version=name,
                scoring="truth_depressed",
                subset=subset,
                n=int(mask.sum()),
                n_train=len(train),
                auc=roc_auc_score(te.truth_depressed[mask], proba[mask]),
                **metrics(te.truth_depressed[mask], pred[mask]),
            )
        )
    if name == "v2_include_normal":
        feats = m.top_features(40)
        pd.DataFrame(
            [dict(direction="depressed(+)", feature=f, coef=c) for f, c in feats["pos"]]
            + [dict(direction="not_depressed(-)", feature=f, coef=c) for f, c in feats["neg"]]
        ).to_csv(RESULTS / "depression_v2_top_features.csv", index=False)
res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "depression_v1_v2_compare.csv", index=False)
print(res.to_string())


def get(v, sub, col="accuracy"):
    return res[(res.version == v) & (res.subset == sub)][col].iloc[0]


v1m = pd.read_csv(RESULTS / "depression_v1_metrics.csv")
grp = pd.read_csv(RESULTS / "depression_v1_by_group.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
rate = grp.iloc[0].pred_depressed_rate
k = int(round(rate * 100))
for i in range(100):
    r_, c_ = divmod(i, 10)
    ax.add_patch(plt.Rectangle((c_, 9 - r_), 0.9, 0.9, color=C_NEG if i < k else C_GRAY))
ax.set_xlim(-0.2, 10.2)
ax.set_ylim(-0.2, 10.2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("학습에서 뺀 상담 일상 발화 4,000건을 v1에 넣으면")
ax.text(5, -0.9, f"{rate:.1%}를 우울로 예측 (칸 하나 = 1%)", ha="center", fontsize=9, color=C_DARK)

ax = axes[1]
rows_d = [
    ("전체 테스트 (11,813)", get("v1_exclude_normal", "test_all"), get("v2_include_normal", "test_all")),
    (
        "상담 발화만 (6,000)",
        get("v1_exclude_normal", "test_counsel_only"),
        get("v2_include_normal", "test_counsel_only"),
    ),
    ("일상 대화만 (5,813)", get("v1_exclude_normal", "test_daily_only"), get("v2_include_normal", "test_daily_only")),
]
for i, (lab, a, b) in enumerate(rows_d[::-1]):
    ax.plot([a, b], [i, i], color=C_GRAY, lw=3, zorder=1)
    ax.scatter([a], [i], color=C_MM, s=110, zorder=3, label="v1 (일상 의도 제외)" if i == 0 else None)
    ax.scatter([b], [i], color=C_AUDIO, s=110, zorder=3, label="v2 (일상 의도 = 0)" if i == 0 else None)
    ax.text(a - 0.006, i + 0.22, f"{a:.3f}", ha="right", fontsize=9, color=C_MM)
    ax.text(b + 0.006, i + 0.22, f"{b:.3f}", ha="left", fontsize=9, color=C_AUDIO)
ax.set_yticks(range(3))
ax.set_yticklabels([r[0] for r in rows_d[::-1]])
orig = v1m[v1m.scoring == "train_label"].accuracy.iloc[0]
ax.axvline(orig, color=C_TEXT, ls="--", lw=1.5)
ax.text(orig, 2.55, f"v1 · 원 범위\n학습 라벨 기준 {orig:.3f}", ha="center", fontsize=8, color=C_DARK)
ax.set_xlim(0.78, 1.02)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel("accuracy (실제 우울 기준)")
ax.legend(loc="lower left", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("같은 테스트셋, 학습에 4,000건을 넣었는지만 다르다")
plt.tight_layout()
plt.savefig(FIGURES / "fig2_depression_scoring.png")
plt.close()

# 상위 feature를 어미, 관용어, 내용으로 나눈다. 어미와 관용어는 출처의 문체이고 내용만 우울 자체다
ENDINGS = [
    "요",
    "어요",
    "습니다",
    "네요",
    "거든요",
    "니다",
    "더라고요",
    "것 같아요",
    "어",
    "네",
    "야",
    "지",
    "다",
    "ㅋㅋ",
    "ㅠㅠ",
    "임",
]
IDIOM = [
    "요즘",
    "자꾸",
    "선생님",
    "계속",
    "사실은",
    "그래서",
    "제가",
    "아무래도",
    "며칠째",
    "진짜",
    "완전",
    "오늘",
    "어제",
    "나",
    "그냥",
    "아",
    "근데",
    "약간",
]


def kind(f):
    t = f.strip()
    if any(t == e or t.endswith(e) and len(t) <= len(e) + 1 for e in ENDINGS):
        return "어미"
    if any(t == w or (t and w.startswith(t) and len(t) >= 2) or t in w for w in IDIOM):
        return "관용어"
    return "내용"


KC = {"어미": C_GRAY, "관용어": C_TEXT, "내용": C_AUDIO}


def dedupe(f):
    f = f.assign(L=f.feature.str.len()).sort_values("L", ascending=False)
    return f.drop_duplicates("coef").sort_values("coef", ascending=False)


v1f = dedupe(pd.read_csv(RESULTS / "depression_v1_top_features.csv"))
v1f = v1f[v1f.direction.str.contains(r"\+")].head(12)
v2f = dedupe(pd.read_csv(RESULTS / "depression_v2_top_features.csv"))
v2f = v2f[v2f.direction.str.contains(r"\+")].head(12)
v1f = v1f.assign(kind=v1f.feature.map(kind))
v2f = v2f.assign(kind=v2f.feature.map(kind))
pd.concat([v1f.assign(version="v1"), v2f.assign(version="v2")])[["version", "feature", "coef", "kind"]].to_csv(
    RESULTS / "depression_feature_kinds.csv", index=False
)
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, (_, r) in enumerate(v1f.iterrows()):
    ax.barh(11 - i, -r.coef, color=KC[r.kind])
    ax.text(-r.coef - 0.15, 11 - i, repr(r.feature.strip()), ha="right", va="center", fontsize=9)
for i, (_, r) in enumerate(v2f.iterrows()):
    ax.barh(11 - i, r.coef, color=KC[r.kind])
    ax.text(r.coef + 0.15, 11 - i, repr(r.feature.strip()), ha="left", va="center", fontsize=9)
ax.axvline(0, color=C_DARK, lw=1)
ax.set_yticks([])
ax.set_xlim(-12, 12)
ax.set_xticks([-8, -4, 0, 4, 8])
ax.set_xticklabels(["8", "4", "0", "4", "8"])
ax.set_xlabel("|LR coef| (char n-gram, 우울 방향)")
ax.text(-6, 12.3, "v1 · 상담 일상 의도 제외", ha="center", fontsize=10)
ax.text(6, 12.3, "v2 · 상담 일상 의도 = 0", ha="center", fontsize=10)
ax.legend(handles=[Patch(color=c, label=k) for k, c in KC.items()], loc="lower right", fontsize=9, title="토큰 종류")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_ylim(-0.8, 13)
plt.tight_layout()
plt.savefig(FIGURES / "fig3_depression_features.png")
plt.close()
