# 4호 글 — 미회수 28대의 해부: 왜 어떤 방법으로도 안 잡히는가
#
# 01의 결과(알고리즘 교체 0~11%, 피처 재설계 0% 회수)에 대한 원인 규명.
#   1. 생성 과정 재현(동일 시드)으로 blur 플래그 복원 → 28대 중 혼합 기기는 몇 대인가
#   2. 28대의 피크 시각 분포 → 야간 창에 몰려 있는가
#   3. "매끈한 범프 vs 뾰족한 스파이크" 지표(roughness)와 요일 간 버스트 변동(dow_burst_cv)
#      에서 28대는 night와 intermittent 사이 어디에 있는가
#   4. GMM 사후확률 — 소프트 배정은 이들의 애매함을 알고 있는가
#   5. 강화 피처(roughness, dow_burst_cv 추가)로 재클러스터링 — 그래도 안 되는가
#
# 주의: 1번은 08 모듈을 임포트해 실행한다 (08은 임포트 시 전체가 실행되지만
#   시드 고정이라 동일 데이터를 다시 쓸 뿐이다).
#
# 실행: .venv/bin/python notebooks/02_diagnose_missed.py
# 출력: outputs/figures/fig3_missed_anatomy.png

import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

SEED = 42
base = Path(__file__).resolve().parents[1]
fig_dir = base / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ── 0. 기준선과 미회수 28대 (01과 동일) ──────────────────────────
df = pd.read_csv(base / "data" / "usage_profiles.csv")
hour_cols = [c for c in df.columns if c.startswith("u_")]
true = df["true_cluster"]
cube = df[hour_cols].to_numpy().reshape(len(df), 7, 24)
hm = cube.mean(axis=1)

X_log = StandardScaler().fit_transform(np.log1p(df[hour_cols + ["total_usage"]].to_numpy()))
km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit_predict(X_log)
maj = {c: true[km == c].value_counts().index[0] for c in range(5)}
missed = np.flatnonzero((true == "intermittent") & (pd.Series(km).map(maj) != "intermittent"))
print(f"미회수 intermittent: {len(missed)}대")

# ── 1. blur 플래그 복원 — 동일 시드로 생성 과정의 난수 시퀀스를 재밟기 ──
spec = importlib.util.spec_from_file_location("gen08", Path(__file__).with_name("00_generate_usage_profiles.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

rng = np.random.default_rng(gen.SEED)
labels = np.repeat(list(gen.CLUSTER_SIZES), list(gen.CLUSTER_SIZES.values()))
labels = labels[rng.permutation(gen.N_DEVICES)]
blurred = []
for label in labels:
    gen.PROFILE_FN[label](rng)  # 난수 소비 순서를 00과 동일하게 유지
    is_blur = False
    if label in ("night", "intermittent") and rng.random() < gen.BLUR_FRAC:
        other = "intermittent" if label == "night" else "night"
        rng.uniform(*gen.BLUR_ALPHA)
        gen.PROFILE_FN[other](rng)
        is_blur = True
    rng.lognormal(0, gen.AMP_SIGMA)
    rng.normal(0, gen.CELL_NOISE_SD, size=(7, 24))
    blurred.append(is_blur)
blur = np.array(blurred)
assert (true.to_numpy() == labels).all(), "레이블 시퀀스 불일치 — 시드 재현 실패"

n_blur_missed = blur[missed].sum()
int_mask = (true == "intermittent").to_numpy()
print(f"혼합(blur) 기기: 전체 {blur.sum()}대 (intermittent {blur[int_mask].sum()}대)")
print(f"미회수 28대 중 혼합: {n_blur_missed}대 / 순수인데 미회수: {len(missed) - n_blur_missed}대")
print(f"혼합인데 회수됨: {blur[int_mask].sum() - n_blur_missed}대 → 혼합 여부가 회수를 가르지 않는다")

# ── 2. 미회수 28대의 피크 시각 ───────────────────────────────────
peak_missed = hm[missed].argmax(axis=1)
night_window = {21, 22, 23, 0, 1, 2}
in_night = sum(1 for h in peak_missed if h in night_window)
print(f"\n미회수 28대의 피크 시각: {np.sort(peak_missed).tolist()}")
print(f"야간 창(21~02시)에 피크: {in_night}/{len(missed)}대")

# ── 3. 모양 지표에서의 중간성 ────────────────────────────────────
def roughness(h: np.ndarray) -> np.ndarray:
    """인접 시각 차이 합 / 총량 — 매끈한 범프(night)면 작고 스파이크(간헐)면 크다."""
    return np.abs(np.diff(h, axis=1)).sum(axis=1) / np.maximum(h.sum(axis=1), 1e-9)

def dow_burst_cv(c: np.ndarray) -> np.ndarray:
    """피크 시각 강도의 요일 간 변동계수 — 간헐은 요일별 가동률이 들쭉날쭉."""
    top_hour = c.mean(axis=1).argmax(axis=1)
    vals = np.take_along_axis(c, top_hour[:, None, None], axis=2).squeeze(-1)
    return vals.std(axis=1) / np.maximum(vals.mean(axis=1), 1e-9)

r = roughness(hm)
dcv = dow_burst_cv(cube)
night_mask = (true == "night").to_numpy()
print("\n모양 지표 (평균): night / 미회수 28대 / intermittent 전체")
print(f"  roughness:    {r[night_mask].mean():.3f} / {r[missed].mean():.3f} / {r[int_mask].mean():.3f}")
print(f"  dow_burst_cv: {dcv[night_mask].mean():.3f} / {dcv[missed].mean():.3f} / {dcv[int_mask].mean():.3f}")
print("→ 28대는 두 지표 모두에서 night 쪽에 치우친 중간 — 모양 정보로도 못 가른다")

# ── 4. GMM 사후확률 — 소프트 배정은 애매함을 아는가 ──────────────
gmm = GaussianMixture(n_components=5, covariance_type="diag", random_state=SEED, n_init=3).fit(X_log)
post = gmm.predict_proba(X_log)
gmm_maj = {c: true[post.argmax(axis=1) == c].value_counts().index[0] for c in range(5)}
int_comp = next(c for c, m in gmm_maj.items() if m == "intermittent")
print(f"\nGMM — 미회수 28대의 간헐 성분 사후확률: 평균 {post[missed, int_comp].mean():.2f}, "
      f"중앙값 {np.median(post[missed, int_comp]):.2f}")
print(f"      최대 성분 사후확률 평균: {post[missed].max(axis=1).mean():.2f} "
      f"→ 소프트 배정조차 이들을 '확신을 갖고' night로 분류한다")

# ── 5. 강화 피처 재시도 — roughness·dow_burst_cv를 넣어도 안 되는가 ──
daily_sum = cube.sum(axis=2)
burst_plus = np.column_stack([
    (cube < 0.5).mean(axis=(1, 2)),
    np.sort(cube.reshape(len(df), -1), axis=1)[:, -8:].mean(axis=1),
    np.sort(hm, axis=1)[:, -3:].sum(axis=1) / np.maximum(hm.sum(axis=1), 1e-9),
    daily_sum.std(axis=1) / np.maximum(daily_sum.mean(axis=1), 1e-9),
    hm[:, [21, 22, 23, 0, 1, 2]].sum(axis=1) / np.maximum(hm.sum(axis=1), 1e-9),
    hm[:, 6:11].sum(axis=1) / np.maximum(hm.sum(axis=1), 1e-9),
    np.log1p(df["total_usage"]),
    r,
    dcv,
])
bp = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit_predict(
    StandardScaler().fit_transform(burst_plus)
)
bmaj = {c: true[bp == c].value_counts().index[0] for c in set(bp)}
rec = sum(1 for i in missed if bmaj[bp[i]] == "intermittent")
print(f"\n강화 피처(9개) 재클러스터링: ARI {adjusted_rand_score(true, bp):.3f}, "
      f"미회수 28대 회수 {rec}/28 — 모양 지표를 넣어도 못 가른다")

# ── 플롯: 해부 3면 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) 피크 시각 분포
ax = axes[0]
bins = np.arange(25)
ax.hist(hm[night_mask].argmax(axis=1), bins=bins, alpha=0.5, color="tab:purple",
        label="night", density=True)
ax.hist(peak_missed, bins=bins, alpha=0.6, color="black", label="미회수 28대",
        density=True, histtype="step", lw=2)
ax.set_xlabel("피크 시각")
ax.set_ylabel("밀도")
ax.set_title(f"(a) 피크 시각 — {in_night}/28대가 야간 창")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) 모양 지표 산점도
ax = axes[1]
for label, color in [("night", "tab:purple"), ("intermittent", "tab:orange")]:
    m = (true == label).to_numpy()
    ax.scatter(r[m], dcv[m], s=10, alpha=0.4, color=color, label=label)
ax.scatter(r[missed], dcv[missed], s=42, facecolors="none", edgecolors="black",
           lw=1.2, label="미회수 28대")
ax.set_xlabel("roughness (프로파일 요철)")
ax.set_ylabel("dow_burst_cv (요일 간 변동)")
ax.set_title("(b) 모양 지표 — 28대는 night 쪽 중간")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (c) GMM 간헐 성분 사후확률 — 확률이 0/1에 몰려 있어 밀도 대신 비율 막대로
ax = axes[2]
pure_int_idx = np.flatnonzero(int_mask & ~np.isin(np.arange(len(df)), missed))
bins_p = np.linspace(0, 1, 11)
for idx, color, label, shift in [
    (pure_int_idx, "tab:orange", "회수된 intermittent", -0.012),
    (missed, "black", "미회수 28대", 0.012),
]:
    counts, _ = np.histogram(post[idx, int_comp], bins=bins_p)
    frac = counts / counts.sum()
    ax.bar(bins_p[:-1] + 0.05 + shift, frac, width=0.024, color=color, alpha=0.75, label=label)
ax.set_xlabel("GMM 간헐 성분 사후확률")
ax.set_ylabel("기기 비율")
ax.set_title("(c) 소프트 배정도 확신에 차 있다 (0 또는 1)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("미회수 28대의 해부 — 신호 자체가 night와 겹친다")
fig.tight_layout()
fig.savefig(fig_dir / "fig3_missed_anatomy.png", dpi=150)
print(f"\n플롯 저장 — {fig_dir / 'fig3_missed_anatomy.png'}")
