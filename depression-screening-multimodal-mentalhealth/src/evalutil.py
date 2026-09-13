"""평가 유틸. 녹음 단위 / 참가자 단위 교차검증과 지표."""

import numpy as np
import pandas as pd
from audiogen import AUDIO_COLS
from config import DATA, N_FOLDS
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def load_recordings():
    rec = pd.read_csv(DATA / "recordings.csv")
    tok = pd.read_csv(DATA / "recordings_tokens.csv")
    coh = pd.read_csv(DATA / "participants.csv")
    df = rec.merge(tok, on="recording_id").merge(coh, on="participant_id")
    df["tokens"] = df.tokens.fillna("")
    return df


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    return dict(
        auc=roc_auc_score(y, p),
        f1=f1_score(y, p >= thr),
        accuracy=accuracy_score(y, p >= thr),
        balanced_accuracy=balanced_accuracy_score(y, p >= thr),
    )


def participant_level(df, p, label_col="label_depressed"):
    """녹음 확률을 참가자별 평균으로 모아 참가자 단위로 채점."""
    g = (
        pd.DataFrame(dict(pid=df.participant_id.values, y=df[label_col].values, p=p))
        .groupby("pid")
        .agg(y=("y", "first"), p=("p", "mean"))
    )
    return metrics(g.y, g.p)


def cross_val_predict(df, fit_predict, y_col="label_depressed", split="participant", seed=0, n_folds=N_FOLDS):
    """fit_predict(train_df, test_df, seed) -> test 확률. OOF 확률 벡터를 돌려준다."""
    y = df[y_col].values
    oof = np.zeros(len(df))
    if split == "participant":
        folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df, y, df.participant_id)
    else:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df, y)
    for tr, te in folds:
        oof[te] = fit_predict(df.iloc[tr], df.iloc[te], seed)
    return oof


def fp_text(tr, te, seed, y_col="label_depressed", token_col="tokens"):
    from models import TextModel

    return TextModel(seed).fit(list(tr[token_col]), tr[y_col].values).predict_proba(list(te[token_col]))[:, 1]


def fp_audio(tr, te, seed, y_col="label_depressed"):
    from models import AudioModel

    return AudioModel(seed).fit(tr[AUDIO_COLS].values, tr[y_col].values).predict_proba(te[AUDIO_COLS].values)[:, 1]


def fp_fusion(tr, te, seed, y_col="label_depressed", token_col="tokens", grouped=True):
    from models import FusionModel

    m = FusionModel(seed).fit(
        list(tr[token_col]),
        tr[AUDIO_COLS].values,
        tr[y_col].values,
        groups=tr.participant_id.values if grouped else None,
    )
    return m.predict_proba(list(te[token_col]), te[AUDIO_COLS].values)[:, 1]


def fp_speaker_prior(tr, te, seed, y_col="label_depressed"):
    """학습셋에 같은 참가자가 있으면 그 참가자의 라벨을 그대로 답한다. 누수의 상한."""
    prior = tr.groupby("participant_id")[y_col].mean()
    return te.participant_id.map(prior).fillna(tr[y_col].mean()).values


def fp_audio_knn(tr, te, seed, y_col="label_depressed", k=15):
    """음성 k-NN. 이웃 기반이라 같은 화자의 다른 녹음을 그대로 찾아낸다 — 누수에 가장 민감한 모델."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(tr[AUDIO_COLS].values)
    m = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(
        sc.transform(tr[AUDIO_COLS].values), tr[y_col].values
    )
    return m.predict_proba(sc.transform(te[AUDIO_COLS].values))[:, 1]
