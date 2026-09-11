# churn-prediction-b2b-ats

글: [B2B 고객사 이탈 분류: 타깃을 정의한 열이 feature로 남았을 때](https://analysis-archive.vercel.app/analyses/churn-prediction-b2b-ats)

<!-- TODO: 배경 문단 직접 쓰기 -->
<!-- 옛 문단: 채용 관리 솔루션(ATS) 고객사 이탈 분류 프로젝트를 가상데이터로 재현한다. 고객사 2,000사에 연간 갱신 확률(`truth_renew_p`)을 심고 계약 이력을 생성한 뒤, 원본 노트북의 절차(외부데이터 결합 → 실행일 기준 90일 규칙 라벨 → 30열 feature → 회귀로 학습)를 그대로 따라가고, 정답을 알고 다시 채점한다. 그다음 결합 탈락의 편향, 라벨의 기준일, 계약 이력에서 파생된 feature를 각각 고친 버전(시간 절단 v2, 생존분석)을 나란히 놓는다. -->

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
.venv/bin/python verify.py
```

- Linux x86 기준 전체 약 3분, 그중 05가 2분. 스크립트는 어느 위치에서 실행해도 된다. 경로는 `common.py`가 파일 위치 기준으로 잡는다. `verify.py`만 폴더 루트에서 실행한다.
- 환경: Python 3.12.3, pandas 2.3.3, scikit-learn 1.8.0, LightGBM 4.7.0, XGBoost 3.4.1, SHAP 0.52.0, lifelines 0.30.3, matplotlib 3.10.8. 이 환경에서 `data/`를 지우고 다시 돌려도 데이터와 `outputs/results/*.csv`가 byte 단위로 같다. macOS에서는 `brew install libomp`가 먼저 필요하고, 결과가 어디까지 달라지는지는 "남은 것"에 적었다.
- `verify.py`는 pipeline의 일부가 아니다. feature 행렬에 `truth_*` 열이 없는지, v2 5사의 시간 절단을 손으로 다시 계산하면 맞는지, 모델 AUC가 oracle을 넘지 않는지를 다시 확인한다.
- 제대로 돌았는지 확인할 숫자
  - 02의 `n_kept` 1,201, 결합 60.05%. 2,000사 중 799사가 결합에서 빠지고, 빠진 회사는 무작위가 아니라 작은 회사에 몰린다
  - 03의 `churn_rate` 0.3922. 실행일 기준 90일 규칙으로 도출한 이탈률이고 1,201사 중 471사
  - 05의 v1 snapshot LightGBM AUC 0.9779. oracle(1,141) 0.8764보다 높고, 정답을 아는 판정자보다 잘 맞힌다는 것이 누수의 증거다. v2 timecut LightGBM 0.8515는 oracle(688) 0.8712 바로 아래 붙어 있다. 이게 정상이다
  - 07의 concordance 0.6637. 생존 시간 순위를 맞힌 비율이라 분류 AUC와 직접 비교할 수 없다. 위험비의 방향 6개가 심어둔 원인과 전부 일치한다는 쪽이 더 중요하다

## Pipeline

| 스크립트 | 역할 | 출력 |
| --- | --- | --- |
| `01_generate_companies.py` | 고객사·계약 이력·외부데이터 생성. 갱신 확률과 결합 탈락 편향을 심는다 | `data/companies.csv`, `contracts.csv`, `external.csv` |
| `02_external_join.py` | 원본 결합 절차(빈값 0 → 연봉 중앙값 → right join → dropna) 재현, 탈락 편향 계량 | `data/merged_v1.csv`, `results/join_bias_by_size.csv`, `join_bias_summary.csv` |
| `03_labels_features_v1.py` | 실행일(2024-02-15) 기준 90일 규칙으로 라벨 도출, 원본 30열 feature, 열별 라벨 분리도 | `data/features_v1.csv`, `snapshot_v1.csv`, `results/label_stats_v1.csv`, `future_revenue_leak.csv`, `churn_by_first_year.csv` |
| `04_features_v2_timecut.py` | 절단 시점 T(실행일 12개월 전) 기준 feature와 oracle | `data/features_v2.csv`, `truth_v2_population.csv`, `results/label_stats_v2.csv` |
| `05_models.py` | 누수 점검, 원본 회귀 재현, 분류기 4종 × v1 / v1 ablation / v2, oracle, SHAP, calibration, 리스크 리스트 | `results/model_compare.csv`, `shap_importance_v1.csv`, `shap_importance_v2.csv`, `calibration.csv`, `risk_list_compare.csv`, `data/shap_values_*.csv` |
| `06_reference_date_sweep.py` | 실행일 sweep: 라벨이 실행일에 따라 얼마나 바뀌는가 | `results/reference_date_sweep.csv`, `ref_sweep.json`(사이트 `src/data/ref-sweep.json`으로 복사, 위젯 RefDateExplorer) |
| `07_survival.py` | Kaplan-Meier·Cox PH: 이탈을 시간으로 다시 묻기 | `results/km_by_product.csv`, `km_key_points.csv`, `cox_summary.csv` |
| `08_site_figures.py` | 그림 8장을 사이트 톤으로 작도. 값은 02\~07의 CSV에서 읽는다 | `outputs/figures/fig1~8.png`, `outputs/figures/site/*.webp` |

`common.py`에 상수 REF_DATE, CHURN_GRACE_DAYS, CUTOFF_T, DATA_END와 라벨 함수 `churn_label()`이 있다. 02, 03, 06이 모두 이 함수로 라벨을 만든다.

## 데이터 스키마

| 파일 | 행 | 내용 |
| --- | --- | --- |
| `companies.csv` | 2,000 | 회사 속성: `employees`, `firm_age`, `size_class`(SMALL/MEDIUM/LARGE/ETC), `product`(diag / posting / both), `hiring_rate`, `attrition_rate`, `headcount_growth`, `hiring_growth`, `attrition_growth`, **`truth_renew_p`**. 마지막 열은 연간 갱신 확률이고 채점에만 쓴다 |
| `contracts.csv` | 6,682 | 계약 행. `contract_id`, `company_id`, `contract_date`, `start_date`, `end_date`, `term_months`(12/24/36), `invoice_seq`(다년 계약은 연 단위 세금계산서 행으로 분할), `amount`, `product`. 2029-06까지 생성돼 있고 스냅샷은 사용하는 쪽에서 `contract_date <= 실행일`로 자른다 |
| `external.csv` | 1,201 | 외부데이터(건강보험·공시를 흉내낸 것). 규모가 클수록 존재. `company_revenue`·`operating_profit`은 55%가 빈값(비상장), 연봉 3열은 30%가 빈값, `industry_salary`에 단위 오류 이상치 1건(9,010) |
| `merged_v1.csv` | 1,201 | 02의 결합 결과. 원본 절차대로 빈값이 0·중앙값으로 채워져 있다 |
| `snapshot_v1.csv` | 1,201 | 03의 스냅샷 파생 전체(연도별 매출 13열, `gubun` 신규/재계약, `first_contract`, `last_end`, `in_use`, `y`). `truth_*` 열은 뺐다 |
| `features_v1.csv` | 1,201 × 31 | 원본 30열 feature + `y`. `rev_2015`, `rev_2024~2027`, `n_contracts`, `total_revenue`, `gubun_*`, `first_year`, `first_month`가 계약 이력 파생 열 |
| `features_v2.csv` | 688 × 31 | T 시점 고객만. `n_contracts_T`, `tenure_months_T`, `total_revenue_T`, `revenue_last12m_T`, `remaining_months_T`, `current_term`, `product_*` + 외부데이터·기업 속성 + `y` |
| `truth_v2_population.csv` | 688 | v2 모집단의 `truth_renew_p`, `truth_churn_p`(1−p^k), `remaining_months_T`, `y`. 채점 전용 |
| `shap_values_v1.csv`, `shap_values_v2.csv` | 1,141 / 688 | 05의 5-fold OOF SHAP 값 |

`remaining_months_T`는 36개 행에서 음수(최소 −2.9)다. v2 모집단 정의가 "최종 종료일이 T−90일 이후"라서, T 직전에 계약이 끝났지만 아직 이탈로 판정되지 않은 회사가 포함되기 때문이다. 정의상 맞는 값이고 그대로 둔다.

## 심어둔 구조

| 구조 | 생성기 | 확인 |
| --- | --- | --- |
| 연간 갱신 확률 `truth_renew_p`의 원인 6개: 인원 성장률(+), 채용 비율(+), 두 상품 동시 사용(+), 재직 인원(+), 업력(−), 퇴사 비율(−) | 01의 `logit` | `cox_summary.csv`(Cox 위험비의 방향 6개 전부 일치, `spearman_vs_truth` 0.867) |
| 이탈 라벨은 심지 않는다. 계약이 끝날 때마다 `truth_renew_p`로 갱신 여부를 뽑고, 라벨은 03이 원본의 90일 규칙으로 도출한다 | 01의 계약 루프, `_common.churn_label()` | `churn_by_first_year.csv`(최초 계약이 오래될수록 이탈률이 높다: 갱신 결정 횟수의 함수) |
| 외부데이터 존재 확률이 재직 인원에 비례 → 결합 탈락이 작은 회사에 몰림 | 01의 `p_ext` | `join_bias_by_size.csv`(50명 미만 12%만 결합, 1,000명 이상 92%), `join_bias_summary.csv`(탈락 회사 이탈률 45.4% vs 잔류 39.2%) |
| 비상장사의 매출·영업이익 빈값 → 원본 절차의 0 채움이 "매출 0인 회사"로 남김 | 01의 `blank_fin` | `join_bias_summary.csv`의 `share_fin_zero_among_kept` 0.5645 |

## 결과 파일

| 파일 | 내용 | 글에서 |
| --- | --- | --- |
| `join_bias_by_size.csv`, `join_bias_summary.csv` | 규모대별 결합률·이탈률, 결합 전후 이탈률 | fig1과 결합 절 |
| `churn_by_first_year.csv` | 최초 계약 연도별 이탈률과 평균 갱신 결정 횟수 | fig2 |
| `label_stats_v1.csv`, `label_stats_v2.csv` | 행 수·이탈률, v2의 갱신 결정 없는 비율(0.3924)과 oracle AUC | 재현 절 |
| `future_revenue_leak.csv` | 열별 라벨 분리도 | fig3 |
| `model_compare.csv` | 회귀 재현, 분류기 4종 × feature 구성, oracle | 모델 비교 표, fig4 |
| `shap_importance_v1.csv`, `shap_importance_v2.csv` | LightGBM OOF SHAP 비중 | fig5 |
| `risk_list_compare.csv` | 상위 20% 리스크 리스트의 실제 이탈률, 심어둔 이탈 확률, 갱신 결정 없는 비율, 겹침 | 리스크 리스트 표 |
| `calibration.csv` | 8분위 신뢰도 곡선과 Brier | fig8 |
| `reference_date_sweep.csv`, `ref_sweep.json` | 실행일별 이탈률과 라벨 뒤집힘 비율(live / frozen) | fig6, 위젯 RefDateExplorer |
| `km_by_product.csv`, `km_key_points.csv`, `cox_summary.csv` | 상품 구성별 KM 곡선, Cox 위험비·concordance | fig7 |

## 원본과 다른 점

원본의 PyCaret 호출 setup, compare, tune은 scikit-learn의 KFold, StratifiedKFold와 cross_val_predict로 옮겼다. 95/5 분할, 10-fold, 원본 random_state는 같게 맞췄다.

## 남은 것

- oracle의 갱신 결정 횟수 k를 12개월마다 한 번으로 셌다. 24개월, 36개월 계약 회사는 k가 실제보다 크고 oracle AUC가 그만큼 조금 높다. 글은 그 오차보다 큰 차이만 다룬다.
- 결과 CSV는 Linux x86에서 만들었다. macOS Apple Silicon, scikit-learn 1.9.1, numpy 2.5.3에서 다시 돌리면 `model_compare.csv`의 XGBoost와 Random Forest 행이 셋째나 넷째 자리에서 달라지고(예: v1 snapshot XGBoost 0.9792가 0.9794), `shap_importance_v1.csv`에서 SHAP이 0인 `rev_2025`, `rev_2026`, `rev_2027`의 순서가 바뀐다. `data/`의 실수 열도 마지막 자리가 흔들린다. LightGBM, Logistic Regression, oracle 행과 글이 인용하는 숫자는 같다.
- `remaining_months_T`가 36개 행에서 음수인 것은 데이터 스키마 절에 적은 대로 정의상 맞는 값이고 그대로 둔다.
