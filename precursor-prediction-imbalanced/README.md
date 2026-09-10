# precursor-prediction-imbalanced

글: [이상 이벤트 예측: 불균형 데이터에서 전조 신호 모델링](https://analysis-archive.vercel.app/analyses/precursor-prediction-imbalanced)

기기 로그로 심각 이벤트를 7일 전에 예측하는 이진 분류를 가상데이터로 재현한다. 기기 500대 × 180일 로그에 전조 신호 2종과 교란 신호 1종을 심고, device-day 단위 feature 테이블(양성 0.36%)을 만든 뒤 시간 기준 분할, leakage 검증, 불균형 처리 3종 비교, SHAP 해석과 ablation까지 간다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

- Apple Silicon 기준 전체 약 35초 (05 학습 12초, 06 SHAP 11초).
- 환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, shap 0.52, matplotlib 3.11. macOS에서는 `brew install libomp`.
- seed 고정(`SEED=42`). `data/`를 지우고 다시 돌려도 CSV가 byte 단위로 같다.
- 제대로 돌았는지 확인할 숫자: 03의 `rows: 77,446`, `positive: 279` (0.3603%, 1:277). 04의 재계산 대조 불일치 0건, 음성 대조 34건. 05의 no_handling PR-AUC 0.6089. 06의 ablation 차이 −0.0068.

## Pipeline

| 스크립트 | 역할 | 출력 |
| --- | --- | --- |
| `01_generate_device_events.py` | 기기 일별 로그 생성, 신호 3종 주입 | `data/device_events.csv`, `data/ground_truth.csv` |
| `02_validate_signals.py` | 심은 신호가 실제로 들어갔는지 이벤트 시점 정렬로 검증 | `figures/fig1~3`, `results/signal_validation.csv` |
| `03_build_features.py` | device-day feature 33개 + 타깃(t+1\~t+7 심각 이벤트) | `data/features.csv`, `results/feature_table_stats.csv` |
| `04_validate_features.py` | leakage 검증 3종: 재계산 대조, 타깃 방향, 음성 대조 | `results/leakage_checks.csv` |
| `05_train_models.py` | 시간 분할(마지막 30일 테스트, purge gap 7일) + LightGBM 불균형 처리 3종 | `results/model_compare.csv`, `split_stats.csv`, `figures/fig4`, `data/model_*.joblib`(미커밋) |
| `06_shap_analysis.py` | SHAP 순위, W7 기여 방향, W7 제거 ablation | `results/shap_ranking.csv`, `ablation_w7.csv`, `figures/fig5~6` |
| `07_threshold_sweep.py` | 테스트 구간 threshold sweep. 글의 인터랙티브 위젯 데이터 | `results/threshold_sweep.json` (사이트 `src/data/threshold-sweep.json`으로 복사) |

## 데이터 스키마

`data/device_events.csv` (251,914행). long 포맷, count > 0인 날만 기록.

| 열 | 설명 |
| --- | --- |
| `device_id` | D0000\~D0499 |
| `event_date` | 2025-01-01부터 180일 |
| `event_type` | `usage` / `warning` / `severe` |
| `event_code` | U1(사용량), W1\~W9(경고), S1(심각 이벤트) |
| `count` | 그날 건수. U1은 사용량 |

`data/ground_truth.csv` (500행). 채점·검증 전용. `group`(normal / severe / confounder), `severe_event_date`(severe 기기 40대만, 나머지 빈값).

`data/features.csv` (77,446행 × 37열). `device_id`, `event_date`, `day`, feature 33개, `target`.

| feature | 정의 |
| --- | --- |
| `u1_mean_{7d,14d}`, `u1_std_{7d,14d}`, `u1_slope_{7d,14d}` | 사용량의 trailing window 평균·표준편차·최소제곱 기울기 |
| `w{1..9}_cnt_{7d,14d}` | 경고 코드별 trailing window 건수 |
| `w{1..9}_slope_7d` | 경고 코드별 7일 기울기 |
| `target` | t+1\~t+7에 심각 이벤트가 있으면 1 |

window는 당일 포함 과거만(t−13\~t). 14일 window가 안 차는 초기 13일, 타깃 window가 잘리는 마지막 7일, 이벤트 발생일 이후 행은 없다.

## 심어둔 구조

| 구조 | 생성기 | 확인 |
| --- | --- | --- |
| 신호 1. 심각 이벤트 7일 전부터 W3 발생률 선형 증가 (+0.5 → +4.0건/일) | `W3_RAMP_*` | fig1, `signal_validation.csv`의 `w3_daily_mean_*` (0.29 → 4.25) |
| 신호 2. 14일 전부터 사용량의 분산만 증가, 평균 유지 (lognormal sigma 0.08 → 0.60) | `USAGE_SIGMA_*` | fig2, `usage_norm_std_*` (0.21 → 0.42) |
| 교란. 기기 12%는 W7이 평시의 10배로 상시 높지만 심각 이벤트와 무관 | `CONFOUNDER_*` | fig3, `w7_daily_mean_confounder` 3.28 vs normal 0.31, `severe_rate_confounder` 0 |

교란군은 심각 기기와 겹치지 않는다. 모델은 "W7 많음 = 안전"을 배우고, 06이 그것을 SHAP 방향과 ablation으로 잡는다.

## 결과 파일

| 파일 | 내용 | 글에서 |
| --- | --- | --- |
| `signal_validation.csv` | 신호 3종 검증 수치 | fig1\~3 캡션 |
| `feature_table_stats.csv` | 행 수, 양성 수·비율, 양성 기기 수 | "최종 테이블은 77,446행, 양성 279행" |
| `leakage_checks.csv` | 재계산 대조 불일치, 이벤트일 이후 행, 타깃 정의 불일치, 음성 대조 검출 | leakage 검증 절 |
| `model_compare.csv` | 불균형 처리 3종의 PR-AUC·ROC-AUC | 비교 표 |
| `split_stats.csv` | 학습·테스트 행 수와 양성 수, scale_pos_weight | 분할 절 |
| `shap_ranking.csv` | mean\|SHAP\| 순위 33개 | SHAP 절의 순위 |
| `ablation_w7.csv` | W7 제거 전후 PR-AUC | "차이 −0.007" |
| `threshold_sweep.json` | threshold별 precision·recall·일평균 경보 | 위젯 ThresholdExplorer |

그림은 `outputs/figures/fig1~6.png`. 사이트에는 같은 그림을 webp로 변환해 올렸다. fig5(SHAP summary)는 shap 라이브러리의 작도라 실행 환경에 따라 픽셀이 조금 다를 수 있다. 값은 `shap_ranking.csv`가 기준이다.
