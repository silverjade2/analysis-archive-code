# churn-prediction-b2b-ats

글: [B2B 고객사 이탈 분류 — 타깃을 정의한 열이 피처로 남았을 때](https://analysis-archive.vercel.app/analyses/churn-prediction-b2b-ats)

채용 관리 솔루션(ATS) 고객사 이탈 분류 프로젝트를 가상데이터로 재현한다. 고객사 2,000사에 연간 갱신 확률(`truth_renew_p`)을 심고 계약 이력을 생성한 뒤, 원본 노트북의 절차(외부데이터 결합 → 실행일 기준 90일 규칙 라벨 → 30열 피처 → 회귀로 학습)를 그대로 따라가고, 정답을 알고 다시 채점한다.

| 스크립트 | 역할 | 출력 |
| --- | --- | --- |
| `01_generate_companies.py` | 고객사·계약 이력·외부데이터 생성. 갱신 확률과 결합 탈락 편향을 심는다 | `data/companies.csv`, `contracts.csv`, `external.csv` |
| `02_external_join.py` | 원본 결합 절차(0 채움 → 중앙값 → dropna) 재현, 탈락 편향 계량 | `data/merged_v1.csv`, `results/join_bias_*.csv` |
| `03_labels_features_v1.py` | 실행일 기준 라벨 도출, 원본 30열 피처, 미래 매출 열의 라벨 분리도 | `data/features_v1.csv`, `results/label_stats_v1.csv`, `future_revenue_leak.csv`, `churn_by_first_year.csv` |
| `04_features_v2_timecut.py` | 절단 시점 T 기준 피처와 오라클 | `data/features_v2.csv`, `truth_v2_population.csv`, `results/label_stats_v2.csv` |
| `05_models.py` | 원본 회귀 재현, 분류기 4종 × v1/v1-ablation/v2, 오라클, SHAP, 캘리브레이션, 리스크 리스트 | `results/model_compare.csv`, `shap_importance_*.csv`, `calibration.csv`, `risk_list_compare.csv` |
| `06_reference_date_sweep.py` | 실행일 스윕 — 라벨이 실행일에 따라 얼마나 바뀌는가 | `results/reference_date_sweep.csv`, `site/ref_sweep.json` |
| `07_survival.py` | Kaplan-Meier·Cox PH — 이탈을 시간으로 다시 묻기 | `results/km_*.csv`, `cox_summary.csv` |
| `08_site_figures.py` | 그림 8장 (사이트 톤) | `outputs/figures/fig*.png` |

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd notebooks && for s in 0*.py; do python "$s"; done
```

시드 고정(`SEED=42`). Linux x86 기준 전체 약 3분(05가 2분). `data/`는 커밋한다. `truth_*` 열은 채점에만 쓰고 피처로는 쓰지 않는다 — 05의 누수 점검이 이를 코드로 확인한다.
`site/RefDateExplorer.jsx`는 사이트에 붙일 인터랙티브 위젯이고 `site/ref_sweep.json`을 읽는다.
