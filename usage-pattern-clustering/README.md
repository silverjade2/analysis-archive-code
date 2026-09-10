# usage-pattern-clustering

글: [사용 패턴 세그멘테이션: 클러스터링의 함정들](https://analysis-archive.vercel.app/analyses/usage-pattern-clustering)

기기 800대의 시간대별 사용 프로파일(요일 7 × 시각 24 = 168개 feature + `total_usage`)에 진짜 군집 4개와 함정 3종을 심어 두고, scaling 없는 1차 시도의 실패, 로그 변환의 트레이드오프, k 선택 지표가 가리킨 k=5의 실체까지 정답을 알고 채점하며 기록한다. 클러스터링은 정답 라벨이 없어 결과가 틀려도 티가 나지 않는다. 그래서 정답을 심었다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

- Apple Silicon 기준 전체 약 12초.
- 환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, scipy 1.18, matplotlib 3.11.
- seed 고정(`SEED=42`). `data/`를 지우고 다시 돌려도 CSV가 byte 단위로 같다.
- 제대로 돌았는지 확인할 숫자: 02의 ARI 0.238과 분산 비율 99.8%, 03의 ARI 0.723(표준화만)·0.897(로그+표준화), 04의 k=5 silhouette 0.412·ARI 0.928.

## Pipeline

| 스크립트 | 역할 | 출력 |
| --- | --- | --- |
| `01_generate_usage_profiles.py` | 가상데이터 생성(진짜 군집 4 + 함정 3종) + 정답 기준 검증 plot | `data/usage_profiles.csv`, `results/true_cluster_sizes.csv`, `figures/fig1` |
| `02_cluster_naive.py` | 1차 시도: scaling 없는 K-means(k=4), 분산 지배 진단 | `results/naive_kmeans_summary.csv`, `naive_crosstab.csv`, `naive_usage_by_pred.csv`, `figures/fig2` |
| `03_cluster_scaled.py` | 2차 시도: 표준화만 vs 로그+표준화(k=4), skewness 진단, 노이즈 15대 추적 | `results/skewness.csv`, `scaling_compare.csv`, `crosstab_standard.csv`, `crosstab_log_standard.csv`, `figures/fig3~4` |
| `04_select_k.py` | k=2\~10 sweep(inertia·silhouette·ARI), k=4/5 대조 | `results/k_sweep.csv`, `k_composition.csv`, `crosstab_k4.csv`, `crosstab_k5.csv`, `figures/fig5~6` |
| `05_profile_clusters.py` | 최종 군집(k=5) z-score 프로파일링과 비즈니스 라벨 | `results/cluster_zscore.csv`, `cluster_labels.csv`, `final_crosstab.csv`, `figures/fig7` |

## 데이터 스키마

`data/usage_profiles.csv` (800행 × 171열).

| 열 | 설명 |
| --- | --- |
| `device_id` | D0000\~D0799 |
| `u_{dow}_h{HH}` × 168 | 요일(mon\~sun) × 시각(00\~23)의 평균 사용 강도, 0\~10 |
| `total_usage` | 전체 사용량 합산 지표, 0\~5000. 셀 합 × 3.5 + 노이즈 |
| `true_cluster` | 정답 라벨(morning / allday_low / night / intermittent / noise). 채점 전용, 클러스터링 입력에 쓰지 않는다 |

## 심어둔 구조

| 구조 | 생성기 | 확인 |
| --- | --- | --- |
| 진짜 군집 4개: morning 230, allday_low 240, night 165, intermittent 150 | `CLUSTER_SIZES`, `*_profile()` | fig1, `true_cluster_sizes.csv` |
| 함정 1. `total_usage`만 스케일이 커서 scaling 없이는 이 축이 거리를 지배 | `TOTAL_SCALE` | `naive_kmeans_summary.csv`의 `var_share_total_usage` 0.998, `naive_usage_by_pred.csv`(예측 군집이 total_usage 구간으로 정확히 갈림) |
| 함정 2. night ↔ intermittent 기기의 35%는 상대 프로파일과 혼합. intermittent의 버스트 시간대도 저녁에 치우침 | `BLUR_FRAC`, `BLUR_ALPHA`, `intermittent_profile()` | `crosstab_k5.csv`(night 군집에 남는 intermittent 28대). 애초 목적(elbow를 애매하게)은 실패했고 글에 그렇게 적었다 |
| 함정 3. 24시간 상시 고강도인 노이즈 소군집 15대 | `noise_profile()` | `scaling_compare.csv`(로그 변환에서 allday_low에 흡수, 순도 6%), `crosstab_k5.csv`(k=5에서 복원) |

## 결과 파일

| 파일 | 내용 | 글에서 |
| --- | --- | --- |
| `naive_kmeans_summary.csv` | ARI, 시간대 feature 분산 합, total_usage 분산, 비율 | 1차 시도 절 (0.238, 497.8, 320,498.9, 99.8%) |
| `naive_crosstab.csv`, `naive_usage_by_pred.csv` | 정답 × 예측 교차표, 예측 군집별 total_usage 범위 | 1차 시도 절의 표와 구간 |
| `skewness.csv` | 시간대 feature skewness 평균·최대, total_usage skewness | 전처리 절 (2.3, 4.1, 5.1) |
| `scaling_compare.csv` | 방법별 ARI, 노이즈 재현율·순도 | 전처리 절 비교 표 |
| `crosstab_standard.csv`, `crosstab_log_standard.csv` | 두 방법의 교차표 | "121대가 allday_low에 흡수", "361대" |
| `k_sweep.csv`, `k_composition.csv` | k별 inertia·silhouette·ARI, k별 군집 구성 | k 선택 표, "k=6은 morning을 109+121로" |
| `crosstab_k4.csv`, `crosstab_k5.csv` | k=4/5 교차표 | fig6, 노이즈 15대의 행방 |
| `cluster_zscore.csv`, `cluster_labels.csv`, `final_crosstab.csv` | 군집별 평균 z-score, 라벨, 최종 교차표 | 라벨링 표와 fig7 |

## 후속 글

세그멘테이션 시리즈 2편(발행 전)은 이 폴더의 생성기를 그대로 복사해 쓴다. 생성기(`01_`)를 고치면 이 폴더의 pipeline 전체를 다시 돌리고, 후속 글 폴더의 `00_generate_usage_profiles.py`도 같은 내용으로 바꾼다.
