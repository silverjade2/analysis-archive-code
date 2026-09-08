# precursor-prediction-imbalanced

가상데이터로 재현한 "이상 이벤트 예측: 불균형 데이터에서 전조 신호 모델링" 프로젝트. 글은 [analysis-archive](https://analysis-archive.vercel.app/analyses/precursor-prediction-imbalanced)에 있다.

기기 로그에서 심각 이벤트를 미리 예측하는 이진 분류. 양성 0.36%의 극단적 불균형에서 시간 기준 분할과 누수 검증, 불균형 처리 3종 비교, SHAP으로 확인한 교란 신호까지를 재현한다. 생성기는 전조 신호 3종을 심어 두고, 그중 하나(W7)는 정상 기기의 사용량 수준 차이와 섞이는 교란 신호다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in notebooks/0*.py; do .venv/bin/python "$s"; done
```

Apple Silicon 기준 전체 약 35초 (05 학습 12초, 06 SHAP 11초). 시드 고정(`SEED=42`) — `data/`를 지우고 다시 돌려도 CSV가 바이트 단위로 같다.

환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, shap 0.52, matplotlib 3.11. macOS에서는 LightGBM용 `libomp`가 필요하다(`brew install libomp`).

## 파이프라인

| 스크립트 | 역할 |
| --- | --- |
| `01_generate_device_events.py` | 기기 일별 로그 생성, 전조 신호 3종 주입 (`data/device_events.csv`, `data/ground_truth.csv`) |
| `02_validate_signals.py` | 신호 검증 플롯 3장 (이벤트 시점 정렬) |
| `03_build_features.py` | device-day 피처 테이블 (33개 피처 + 타깃, `data/features.csv`) |
| `04_validate_features.py` | 누수 검증 3종 (재계산 대조·타깃 방향·음성 대조) |
| `05_train_models.py` | 시간 분할 + LightGBM 불균형 처리 3종 비교, PR 커브 (`data/model_*.joblib`) |
| `06_shap_analysis.py` | SHAP 해석 + W7 ablation |
| `07_threshold_sweep.py` | 테스트 구간 임계값 스윕 — 글의 인터랙티브 위젯 데이터 (`outputs/results/threshold_sweep.json`) |

그림은 `outputs/figures/fig1~6.png`. 07의 JSON은 사이트 저장소의 `src/data/threshold-sweep.json`으로 복사해 쓴다(정적 임베드).

`data/`의 CSV는 커밋한다. 코드를 돌리지 않아도 생성 데이터를 열어볼 수 있게 하기 위해서다. 모델 파일(`*.joblib`)은 커밋하지 않는다. 생성기(`01_`)를 고치면 파이프라인 전체를 다시 돌려 데이터·결과·글의 숫자를 같은 커밋에 넣는다.
