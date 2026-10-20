"""05의 불균형 처리 비교를 seed x 분할 시작일로 반복. 한 seed, 한 분할의 우연인지

- test 양성 42행이 기기 7대에서 나와서 PR-AUC 차 0.016은 기기 한두 대 순위로 뒤집힐 수 있음
- 01과 03을 SEED만 바꿔 임시 폴더에서 다시 돌려 데이터 10벌, 분할 시작일 4개, 05와 같은 LightGBM 3종
- seed 42 x day 150 test는 기기 단위 bootstrap으로 PR-AUC 95% CI. 05 결과 파일은 안 건드림
"""

import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = [int(a) for a in sys.argv[1:]] or list(range(42, 52))
TEST_STARTS = (120, 130, 140, 150)  # 150이 05의 분할
PURGE_GAP = 7
ORIGINAL = (42, 150)
N_BOOT = 1000
base = Path(__file__).resolve().parents[1]
scripts = base / "scripts"

lgbm_params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30, verbosity=-1, n_jobs=-1)


def build_features(seed: int, workdir: Path) -> pd.DataFrame:
    """01과 03의 SEED 상수만 바꾼 사본을 workdir에서 실행. 03은 SEED가 없어 그대로"""
    (workdir / "scripts").mkdir(parents=True, exist_ok=True)
    for name in ("01_generate_device_events.py", "03_build_features.py"):
        src = (scripts / name).read_text().replace("SEED = 42", f"SEED = {seed}", 1)
        copy = workdir / "scripts" / name
        copy.write_text(src)
        subprocess.run([sys.executable, str(copy)], check=True, capture_output=True)
    return pd.read_csv(workdir / "data" / "features.csv")


def fit_variants(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    feature_cols = [c for c in train.columns if c not in ("device_id", "event_date", "day", "target")]
    X_tr, y_tr, X_te = train[feature_cols], train["target"], test[feature_cols]
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    variants = {
        "no_handling": LGBMClassifier(**lgbm_params, random_state=seed),
        "class_weight": LGBMClassifier(**lgbm_params, random_state=seed, class_weight="balanced"),
        "scale_pos_weight": LGBMClassifier(**lgbm_params, random_state=seed, scale_pos_weight=n_neg / n_pos),
    }
    return {name: model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1] for name, model in variants.items()}


rows = []
original_scores = None
with tempfile.TemporaryDirectory() as tmp:
    for seed in SEEDS:
        workdir = Path(tmp) / f"seed{seed}"
        table = build_features(seed, workdir)
        if seed == 42:
            same = filecmp.cmp(workdir / "data" / "features.csv", base / "data" / "features.csv", shallow=False)
            print(f"seed 42 features.csv가 data/와 같음: {same}")
            assert same, "01과 03 사본이 저장소 데이터를 재현하지 못함"
        for start in TEST_STARTS:
            train = table[table["day"] <= start - 1 - PURGE_GAP]
            test = table[table["day"] >= start]
            scores = fit_variants(train, test, seed)
            if (seed, start) == ORIGINAL:
                original_scores = (test, scores)
            for name, score in scores.items():
                rows.append(
                    {
                        "seed": seed,
                        "test_start_day": start,
                        "variant": name,
                        "pr_auc": average_precision_score(test["target"], score),
                        "roc_auc": roc_auc_score(test["target"], score),
                        "train_positive": int(train["target"].sum()),
                        "test_positive": int(test["target"].sum()),
                        "test_positive_devices": test.loc[test["target"] == 1, "device_id"].nunique(),
                    }
                )
        done = [r for r in rows if r["seed"] == seed and r["test_start_day"] == 150]
        print(f"seed {seed:2d} day150: " + ", ".join(f"{r['variant']} {r['pr_auc']:.3f}" for r in done))

res = pd.DataFrame(rows)
res_dir = base / "outputs" / "results"
res.round(4).to_csv(res_dir / "model_compare_repeat.csv", index=False)

wide = res.pivot(index=["seed", "test_start_day"], columns="variant", values="pr_auc")
summary = []
for name in ("no_handling", "class_weight", "scale_pos_weight"):
    summary.append(
        {"stat": f"pr_auc_{name}", "mean": wide[name].mean(), "min": wide[name].min(), "max": wide[name].max()}
    )
for other in ("class_weight", "scale_pos_weight"):
    diff = wide["no_handling"] - wide[other]
    summary.append(
        {
            "stat": f"diff_no_handling_minus_{other}",
            "mean": diff.mean(),
            "min": diff.min(),
            "max": diff.max(),
            "n_positive": int((diff > 0).sum()),
            "n_total": len(diff),
        }
    )
summary = pd.DataFrame(summary)
summary.round(4).to_csv(res_dir / "model_compare_repeat_summary.csv", index=False)
print(f"\nseed {len(SEEDS)}개 x 분할 {len(TEST_STARTS)}개")
print(summary.round(4).to_string(index=False))

if original_scores is not None:
    test, scores = original_scores
    devices = test["device_id"].to_numpy()
    uniq = np.unique(devices)
    y = test["target"].to_numpy()
    rng = np.random.default_rng(42)
    boot = {name: [] for name in scores}
    boot["diff_no_handling_minus_class_weight"] = []
    for _ in range(N_BOOT):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(devices == d) for d in picked])
        if y[idx].sum() == 0:
            continue
        ap = {name: average_precision_score(y[idx], s[idx]) for name, s in scores.items()}
        for name, v in ap.items():
            boot[name].append(v)
        boot["diff_no_handling_minus_class_weight"].append(ap["no_handling"] - ap["class_weight"])
    point = {name: average_precision_score(y, s) for name, s in scores.items()}
    point["diff_no_handling_minus_class_weight"] = point["no_handling"] - point["class_weight"]
    ci = pd.DataFrame(
        {
            "stat": list(boot),
            "point": [point[k] for k in boot],
            "ci_lo": [np.percentile(v, 2.5) for v in boot.values()],
            "ci_hi": [np.percentile(v, 97.5) for v in boot.values()],
            "n_boot": [len(v) for v in boot.values()],
            "test_devices": len(uniq),
            "test_positive_devices": test.loc[test["target"] == 1, "device_id"].nunique(),
        }
    )
    ci.round(4).to_csv(res_dir / "device_bootstrap.csv", index=False)
    print(f"\nseed 42 x day 150, 기기 {len(uniq)}대 bootstrap {N_BOOT}회")
    print(ci.round(4).to_string(index=False))
