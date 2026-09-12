# churn-prediction-b2b-ats

[B2B 고객사 이탈 분류: 타깃을 정의한 열이 feature로 남았을 때](https://analysis-archive.vercel.app/analyses/churn-prediction-b2b-ats) 재현 코드.

원본은 실제 B2B ATS 고객사 데이터로 했던 이탈 예측. 여기서는 같은 스키마의 가상 데이터를 만들어 원본 절차를 그대로 돌리고 어디서 누수가 났는지 확인한다 (실제 데이터 없음).

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
python verify.py   # 선택. 루트에서
```

전체 3분 정도 (05가 2분). 경로는 `common.py`에서 파일 위치 기준으로 잡아서 어디서 실행해도 된다.

재현에 쓴 버전: Python 3.12, pandas 2.3, scikit-learn 1.8, LightGBM 4.7, XGBoost 3.4, shap 0.52, lifelines 0.30. 이 조합에서는 `data/`를 지우고 다시 돌려도 결과 CSV가 동일. 다른 버전(특히 Apple Silicon)에서는 XGBoost/RF 행이 셋째 자리에서 흔들리는데 글에서 인용하는 숫자는 LightGBM, LR, oracle 행이라 영향 없음. macOS는 `brew install libomp` 먼저.

## 스크립트

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 고객사 2000사, 계약 이력, 외부데이터 생성. 갱신 확률 `truth_renew_p`와 결합 탈락 편향을 심음 | `data/companies.csv`, `contracts.csv`, `external.csv` |
| 02 | 원본 결합 절차 재현 (fillna 0, 연봉 median, right join, dropna 순). 탈락 편향 집계 | `merged_v1.csv`, `join_bias_*.csv` |
| 03 | 실행일(2024-02-15) 기준 90일 규칙으로 라벨. 원본 feature 30열 | `features_v1.csv`, `future_revenue_leak.csv` |
| 04 | T(실행일 12개월 전) 절단 feature + oracle | `features_v2.csv`, `truth_v2_population.csv` |
| 05 | 원본 RF 회귀 재현, 분류기 4종 × v1/ablation/v2, oracle, SHAP, calibration | `model_compare.csv`, `shap_importance_*.csv` |
| 06 | 실행일 sweep. today를 옮기면 라벨이 얼마나 바뀌는지 | `reference_date_sweep.csv`, `ref_sweep.json` |
| 07 | KM, Cox PH | `km_*.csv`, `cox_summary.csv` |
| 08 | 그림. results CSV만 읽음 | `outputs/figures/*.png` |

라벨 규칙 `churn_label()`과 날짜 상수는 `common.py`에 있고 02, 03, 06이 공유한다.

## 확인할 숫자

- 02: 2000사 중 1201사 결합 (60%). 빠진 799사는 작은 회사에 몰림 (50명 미만은 12%만 결합)
- 03: 이탈률 0.392
- 05: v1 LightGBM AUC 0.978 > oracle(1141) 0.876. 정답을 아는 판정자보다 높으면 누수. v2 LightGBM 0.852 < oracle(688) 0.871, 이게 정상
- 07: concordance 0.664. Cox 계수 부호 6개가 심어둔 방향과 전부 일치

`truth_*` 열은 생성기의 잠재 변수라 채점에만 쓰고 feature에는 안 들어간다. `verify.py`가 이걸 다시 확인한다.

## 데이터

`companies.csv` 2000행, `contracts.csv` 6682행 (다년 계약은 연 단위 세금계산서 행으로 split), `external.csv` 1201행 (규모 클수록 존재. 매출/영업이익 55% 결측, 연봉 30% 결측, 단위 오류 1건). 계약은 2029-06까지 생성돼 있고 스냅샷 절단은 쓰는 쪽에서 `contract_date <= 실행일`로 한다.

`features_v1.csv`의 `rev_2015`, `rev_2024~2027`, `n_contracts`, `total_revenue`, `gubun_*`, `first_year/month`가 계약 이력 파생 열이고 글에서 문제 삼는 부분. 원본이 `rev_2018~2023`은 drop하고 2015와 2024 이후만 남겼는데 이유를 모르겠어서 그대로 재현했다.

`features_v2.csv`의 `remaining_months_T`가 36건 음수인 건 모집단 정의(last_end >= T-90d) 때문이고 의도한 값.

## 원본과 다른 점

PyCaret setup/compare/tune 대신 sklearn KFold, StratifiedKFold, cross_val_predict. 95/5 split, 10-fold, random_state 786은 원본과 같음.

## 알려진 문제

- oracle의 갱신 결정 횟수 k를 12개월 단위로 세서 24/36개월 계약 회사는 k가 과대. oracle AUC가 그만큼 약간 높음. 글은 이 오차보다 큰 차이만 다룸
