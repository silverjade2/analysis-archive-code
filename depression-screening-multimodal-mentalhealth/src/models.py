"""모델. 텍스트: 형태소 TF-IDF(1~2gram) + 로지스틱 회귀. 음성: 표준화 + HistGradientBoosting.
융합: 두 채널의 out-of-fold 확률을 로지스틱 회귀로 결합(late fusion). OOF는 참가자 단위로 접는다."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class TextModel:
    def __init__(self, seed, C=1.0):
        self.vec = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
        )
        self.clf = LogisticRegression(C=C, max_iter=3000, random_state=seed, class_weight="balanced")

    def fit(self, tokens, y):
        self.clf.fit(self.vec.fit_transform(tokens), y)
        return self

    def predict_proba(self, tokens):
        return self.clf.predict_proba(self.vec.transform(tokens))

    def top_features(self, k=20):
        names = np.array(self.vec.get_feature_names_out())
        coef = self.clf.coef_
        if coef.shape[0] == 1:
            o = np.argsort(coef[0])
            return pd.DataFrame(
                dict(feature=np.r_[names[o[::-1][:k]], names[o[:k]]], coef=np.r_[coef[0][o[::-1][:k]], coef[0][o[:k]]])
            )
        rows = []
        for i, c in enumerate(self.clf.classes_):
            o = np.argsort(coef[i])[::-1][:k]
            rows += [dict(cls=c, feature=names[j], coef=coef[i][j]) for j in o]
        return pd.DataFrame(rows)


class AudioModel:
    def __init__(self, seed):
        self.sc = StandardScaler()
        self.clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
        )

    def fit(self, X, y):
        self.clf.fit(self.sc.fit_transform(X), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self.sc.transform(X))


class FusionModel:
    """채널별 OOF 확률 → 상위 로지스틱 회귀. groups를 주면 OOF도 참가자 단위로 접는다."""

    def __init__(self, seed, n_folds=5):
        self.seed, self.n_folds = seed, n_folds
        self.fuser = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)

    def _oof(self, make, X, y, groups):
        y = np.asarray(y)
        oof = np.zeros((len(y), len(np.unique(y))))
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed).split(
                X, y, groups
            )
        else:
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed).split(X, y)
        for tr, va in splitter:
            m = make().fit(_take(X, tr), y[tr])
            oof[va] = m.predict_proba(_take(X, va))
        return oof

    def fit(self, tokens, A, y, groups=None):
        tokens = list(tokens)
        A = np.asarray(A)
        y = np.asarray(y)
        self.text = TextModel(self.seed).fit(tokens, y)
        self.audio = AudioModel(self.seed).fit(A, y)
        Z = np.hstack(
            [
                self._oof(lambda: TextModel(self.seed), tokens, y, groups),
                self._oof(lambda: AudioModel(self.seed), A, y, groups),
            ]
        )
        self.fuser.fit(Z, y)
        return self

    def predict_proba(self, tokens, A):
        return self.fuser.predict_proba(
            np.hstack([self.text.predict_proba(list(tokens)), self.audio.predict_proba(np.asarray(A))])
        )


def _take(X, idx):
    return [X[i] for i in idx] if isinstance(X, list) else X[idx]
