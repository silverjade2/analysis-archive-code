"""텍스트·음성 모델. PLM fine-tuning 대신 char n-gram TF-IDF + LR, 재현 대상은 실험 구조이지 모델 자체가 아니다"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class TextClf:
    def __init__(self, seed):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
        self.clf = LogisticRegression(C=2.0, max_iter=2000, random_state=seed)
    def fit(self, texts, y):
        self.clf.fit(self.vec.fit_transform(texts), y); return self
    def predict(self, texts):
        return self.clf.predict(self.vec.transform(texts))
    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))
    @property
    def classes_(self):
        return self.clf.classes_
    def top_features(self, k=15):
        names = np.array(self.vec.get_feature_names_out())
        coef = self.clf.coef_
        if coef.shape[0] == 1:
            order = np.argsort(coef[0])
            return {"pos": list(zip(names[order[::-1][:k]], coef[0][order[::-1][:k]].round(3))),
                    "neg": list(zip(names[order[:k]], coef[0][order[:k]].round(3)))}
        return {c: list(zip(names[np.argsort(coef[i])[::-1][:k]], np.sort(coef[i])[::-1][:k].round(3)))
                for i, c in enumerate(self.clf.classes_)}


class AudioClf:
    def __init__(self, seed):
        self.sc = StandardScaler()
        self.clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    def fit(self, X, y):
        self.clf.fit(self.sc.fit_transform(X), y); return self
    def predict(self, X):
        return self.clf.predict(self.sc.transform(X))
    def predict_proba(self, X):
        return self.clf.predict_proba(self.sc.transform(X))
    @property
    def classes_(self):
        return self.clf.classes_


class MultimodalClf:
    """late fusion. 채널별 LR의 클래스 확률을 이어 붙여 상위 LR이 결합, 상위 분류기는 out-of-fold 확률로 학습해 채널 과적합이 새지 않게 한다"""
    def __init__(self, seed, n_folds=5):
        from sklearn.model_selection import StratifiedKFold
        self.seed, self.n_folds = seed, n_folds
        self.kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        self.fuser = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    def _oof(self, make, X, y):
        y = np.asarray(y); classes = np.unique(y)
        oof = np.zeros((len(y), len(classes)))
        for tr, va in self.kf.split(X, y):
            m = make().fit(X[tr] if not isinstance(X, pd.Series) else X.iloc[tr], y[tr])
            p = m.predict_proba(X[va] if not isinstance(X, pd.Series) else X.iloc[va])
            oof[va] = p  # StratifiedKFold라 fold마다 모든 클래스가 있고 classes_ 순서가 같다
        return oof
    def fit(self, texts, A, y):
        texts = pd.Series(texts).reset_index(drop=True); A = np.asarray(A); y = np.asarray(y)
        self.text = TextClf(self.seed); self.audio = AudioClf(self.seed)
        oof_t = self._oof(lambda: TextClf(self.seed), texts, y)
        oof_a = self._oof(lambda: AudioClf(self.seed), A, y)
        self.text.fit(texts, y); self.audio.fit(A, y)
        self.fuser.fit(np.hstack([oof_t, oof_a]), y); return self
    def _Z(self, texts, A):
        return np.hstack([self.text.predict_proba(texts), self.audio.predict_proba(np.asarray(A))])
    def predict(self, texts, A):
        return self.fuser.predict(self._Z(texts, A))
    def channel_coef_norm(self):
        k = self.fuser.coef_.shape[1] // 2
        c = np.abs(self.fuser.coef_)
        return float(c[:, :k].sum()), float(c[:, k:].sum())
