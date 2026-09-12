# precursor-prediction-imbalanced

[이상 이벤트 예측: 불균형 데이터에서 전조 신호 모델링](https://analysis-archive.vercel.app/analyses/precursor-prediction-imbalanced) 재현 코드.

원본은 실제 기기 로그로 했던 심각 이벤트 전조 예측. 여기서는 전조 신호를 심은 가상 로그를 만들어 같은 절차를 돌리고, 모델이 심은 신호를 찾는지와 함정을 어떻게 쓰는지 본다 (실제 데이터 없음).

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

전체 35초 정도 (05 학습 12초, 06 SHAP 11초). 경로는 파일 위치 기준이라 어디서 실행해도 된다.

재현에 쓴 버전: Python 3.14, numpy 2.5, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, shap 0.52, matplotlib 3.11. 이 조합에서는 `data/`를 지우고 다시 돌려도 데이터와 결과 CSV가 동일. fig5만 shap 라이브러리가 그리는 그림이라 픽셀이 조금 다를 수 있음. macOS는 `brew install libomp` 먼저.

## 스크립트

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 기기 500대 x 180일 일별 로그. 신호 3종 주입 | `data/device_events.csv`, `ground_truth.csv` |
| 02 | 심은 신호가 들어갔는지 이벤트일 기준 정렬로 확인 | `fig1~3`, `signal_validation.csv` |
| 03 | device-day feature 33개 + target (t+1~t+7 심각 이벤트) | `data/features.csv`, `feature_table_stats.csv` |
| 04 | 누수 검증 3종. 재계산 대조, target 정의, 음성 대조 | `leakage_checks.csv` |
| 05 | 시간 분할 (day 150부터 test, purge 7일) + LightGBM 불균형 처리 3종 | `model_compare.csv`, `split_stats.csv`, `fig4`, `data/model_*.joblib` (미커밋) |
| 06 | SHAP 순위, W7 기여 방향, W7 제거 ablation | `shap_ranking.csv`, `ablation_w7.csv`, `fig5~6` |
| 07 | test 구간 threshold sweep. 위젯 ThresholdExplorer 데이터 | `threshold_sweep.json` (사이트 `src/data/threshold-sweep.json`) |

## 확인할 숫자

- 03: 77,446행, 양성 279 (0.36%, 1:277). 이 불균형이 글의 전제
- 04: 재계산 대조 불일치 0건. 음성 대조 34건은 일부러 넣은 누수를 검증이 잡는다는 뜻, 여기가 0이면 검증 쪽이 망가진 것
- 05: no_handling PR-AUC 0.609. 무작위 기준선 0.004의 150배쯤이고 불균형 처리 2종보다 높음
- 06: W7 제거 ablation 차이 -0.007. 함정은 빼도 성능 그대로

## 데이터

`device_events.csv` 251,914행. long 포맷으로 count > 0인 날만 기록. 열은 `device_id`(D0000~D0499), `event_date`(2025-01-01부터 180일), `event_type`(usage / warning / severe), `event_code`(U1, W1~W9, S1), `count`. `ground_truth.csv` 500행은 `group`(normal / severe / confounder)과 `severe_event_date`(severe 40대만). 채점과 검증에만 쓴다.

`features.csv` 77,446행 x 37열. 사용량의 7d/14d 평균, 표준편차, 기울기 6개 + 경고 코드별 7d/14d 건수와 7d 기울기 27개. window는 당일 포함 과거만 (t-13~t). 14일 window가 안 차는 초기 13일, target window가 잘리는 마지막 7일, 이벤트 발생일 이후 행은 없음.

## 심어둔 신호

- 신호 1. 이벤트 7일 전부터 W3 발생률 선형 증가, +0.5에서 +4.0건/일. `signal_validation.csv`의 w3 일 평균 0.29 -> 4.25
- 신호 2. 14일 전부터 사용량 분산만 증가, 평균 유지 (lognormal sigma 0.08 -> 0.60). 정규화 표준편차 0.21 -> 0.42
- 함정. 기기 12%는 W7이 평시의 10배로 상시 높지만 이벤트와 무관 (일 평균 3.28 vs normal 0.31, 경험률 0). severe 기기와 안 겹치게 뽑아서 모델은 "W7 많음 = 안전"을 배우고, 06이 이걸 SHAP 방향과 ablation으로 잡음

## 알려진 문제

- test 구간을 "마지막 30일"이라 부르지만 실제 행은 day 150~172의 23일치. 03이 마지막 7일을 버려서. 07의 일평균 경보는 23일로 나눈 값
- 심은 신호 중 사용량 표준편차는 SHAP 1위인데 W3 계열은 7, 22, 25위. 함정 `w7_cnt_7d`가 5위로 더 위. 7일 램프가 7d/14d 합계 feature에서 뭉개지는 것 같은데 확인 안 함
