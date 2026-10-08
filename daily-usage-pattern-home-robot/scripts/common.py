"""경로, 기간, 기능표, 판단 기준 상수, numpy k-means. 스크립트 위치 기준이라 어디서 실행해도 됨"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "outputs" / "results"

SEED = 42
N_DEVICES = 1500
DAYS = pd.bdate_range("2025-09-01", "2025-09-26")  # 평일 20일

# key, 이름, 종류 (occ = 켜 두는 기능, instant = 순간 기능)
FEATURES = [
    ("fixed", "제자리 모드", "occ"),
    ("patrol", "이동 모드", "occ"),
    ("schedule", "기본 스케줄", "occ"),
    ("autocall", "센서 호출 모드", "occ"),
    ("camera", "원격 카메라", "occ"),
    ("follow", "따라오기", "occ"),
    ("voice", "음성 대화", "instant"),
    ("vital", "생체 측정", "instant"),
    ("spacescan", "공간 스캔", "instant"),
]
FEAT = [f[0] for f in FEATURES]
NAME = {f[0]: f[1] for f in FEATURES}
OCC = [f[0] for f in FEATURES if f[2] == "occ"]

SCHEDULE_HOURS = [9, 12, 15, 18]  # 최초 세팅 기본 스케줄 시각
SCHEDULE_MAX_MIN = 10  # 이 시각에 시작한 이 길이 미만의 이동 모드를 기본 스케줄로 봄 (휴리스틱)
MIN_EVENTS = 2  # 군집 대상 기기-일의 하루 시작 건수 하한
BLOCK_HOURS = 2  # 군집 행렬의 시간 구간 길이
CONT_GAP_SEC = 60  # 같은 기능이 끝나고 이 안에 다시 시작되면 계속 켜져 있던 것으로 봄
SHORT_FLOOR = 0.1  # 켜 두는 기능의 구간 점유 하한 (짧게 쓰고 끈 경우)
K_RANGE = list(range(3, 9))
SIL_TOL = 0.03  # silhouette 최대값에서 이 안의 k 중 가장 작은 k
SIL_SAMPLE = 3000


def relabel_schedule(e: pd.DataFrame) -> pd.DataFrame:
    """이동 모드 중 기본 스케줄 시각에 시작한 짧은 것을 기본 스케줄로 (로그에 따로 안 찍혀서)"""
    h = e["start_ts"].dt.hour
    sched = (e["feature"] == "patrol") & h.isin(SCHEDULE_HOURS) & (e["dur_sec"] < SCHEDULE_MAX_MIN * 60)
    e = e.copy()
    e.loc[sched, "feature"] = "schedule"
    return e


def kmeans(x: np.ndarray, k: int, seed: int = SEED, n_init: int = 5, iters: int = 100):
    """k-means++ 초기화 + Lloyd. 가장 낮은 inertia 판을 돌려줌 (분석 환경에 sklearn 이 없어 numpy 로)"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        c = [x[rng.integers(len(x))]]
        d2 = ((x - c[0]) ** 2).sum(1)
        for _ in range(1, k):
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
            d2 = np.minimum(d2, ((x - c[-1]) ** 2).sum(1))
        c = np.array(c)
        for _ in range(iters):
            dist = (x**2).sum(1)[:, None] - 2 * x @ c.T + (c**2).sum(1)[None, :]
            lab = dist.argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        dist = (x**2).sum(1)[:, None] - 2 * x @ c.T + (c**2).sum(1)[None, :]
        lab = dist.argmin(1)
        inertia = float(np.maximum(dist[np.arange(len(x)), lab], 0).sum())
        if best is None or inertia < best[2]:
            best = (lab, c, inertia)
    return best


def silhouette(x: np.ndarray, lab: np.ndarray, sample: int = SIL_SAMPLE, seed: int = SEED) -> float:
    """표본 silhouette (전체 쌍 거리는 커서 표본으로)"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    xs, ls = x[idx], lab[idx]
    sq = (xs**2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] - 2 * xs @ xs.T + sq[None, :], 0))
    s = np.zeros(len(xs))
    for i in range(len(xs)):
        same = ls == ls[i]
        same[i] = False
        if not same.any():
            continue
        a = d[i, same].mean()
        b = min(d[i, ls == j].mean() for j in np.unique(ls) if j != ls[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def k_sweep(x: np.ndarray, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        lab, _, inertia = kmeans(x, k, seed)
        rows.append(
            {
                "k": k,
                "inertia": round(inertia, 1),
                "silhouette": round(silhouette(x, lab), 3),
                "smallest_share": round(np.bincount(lab).min() / len(lab), 3),
            }
        )
    return pd.DataFrame(rows)


def pick_k(sweep: pd.DataFrame) -> int:
    """silhouette 최대값과 SIL_TOL 안인 k 중 가장 작은 k"""
    top = sweep["silhouette"].max()
    return int(sweep.loc[sweep["silhouette"] >= top - SIL_TOL, "k"].min())


def relabel_by_size(lab: np.ndarray) -> np.ndarray:
    """크기 순으로 1, 2, ... 번호"""
    order = np.argsort(-np.bincount(lab))
    m = {old: new + 1 for new, old in enumerate(order)}
    return np.array([m[v] for v in lab])


def ari(a, b) -> float:
    """adjusted Rand index"""
    t = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy()
    comb = lambda n: n * (n - 1) / 2  # noqa: E731
    s_ij = comb(t).sum()
    s_a, s_b = comb(t.sum(1)).sum(), comb(t.sum(0)).sum()
    exp = s_a * s_b / comb(t.sum())
    return float((s_ij - exp) / ((s_a + s_b) / 2 - exp))


def day_matrix(cells: pd.DataFrame, keys: pd.DataFrame, features: list, normalize: bool = True) -> np.ndarray:
    """기기-일 x (기능 x 시간 구간) 행렬. 켜 두는 기능은 구간 점유 비율 (하한 SHORT_FLOOR), 순간 기능은 사용 여부.
    normalize 면 기기-일마다 기능별 구간 벡터를 길이 1 로 맞춤"""
    nb = 24 // BLOCK_HOURS
    c = cells[cells["feature"].isin(features)].copy()
    c["block"] = c["hr"] // BLOCK_HOURS
    g = (
        c.groupby(["serial", "dt", "feature", "block"])
        .agg(occ=("occ_sec", "sum"), n=("start_cnt", "sum"))
        .reset_index()
    )
    is_occ = g["feature"].isin(OCC)
    val = np.where(is_occ, np.clip(g["occ"] / (BLOCK_HOURS * 3600), 0, 1), (g["n"] > 0).astype(float))
    val = np.where(is_occ & (g["n"] > 0), np.maximum(val, SHORT_FLOOR), val)
    g["val"] = val
    row = keys.reset_index(drop=True).reset_index().set_index(["serial", "dt"])["index"]
    g = g[g.set_index(["serial", "dt"]).index.isin(row.index)]
    ri = row.reindex(pd.MultiIndex.from_frame(g[["serial", "dt"]])).to_numpy()
    fi = g["feature"].map({f: i for i, f in enumerate(features)}).to_numpy()
    x = np.zeros((len(keys), len(features), nb))
    x[ri, fi, g["block"].to_numpy()] = g["val"].to_numpy()
    if normalize:
        n = np.linalg.norm(x, axis=2, keepdims=True)
        x = np.divide(x, n, out=np.zeros_like(x), where=n > 0)
    return x.reshape(len(keys), -1)
