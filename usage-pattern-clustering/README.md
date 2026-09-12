# usage-pattern-clustering

[사용 패턴 세그멘테이션: 클러스터링의 함정들](https://analysis-archive.vercel.app/analyses/usage-pattern-clustering) 재현 코드.

원본은 실제 기기 사용 로그로 했던 사용 패턴 세그멘테이션. 클러스터링은 정답이 없어 틀려도 티가 안 나서, 여기서는 정답 군집과 함정을 심은 가상 프로파일을 만들어 같은 절차를 돌리고 어디서 틀리는지 본다 (실제 데이터 없음).

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

전체 12초 정도. 경로는 파일 위치 기준이라 어디서 실행해도 된다.

재현에 쓴 버전: Python 3.14, numpy 2.5, pandas 3.0, scipy 1.18, scikit-learn 1.9, matplotlib 3.11. 이 조합에서는 `data/`를 지우고 다시 돌려도 데이터와 결과 CSV가 동일.

## 스크립트

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 기기 800대 프로파일 생성. 진짜 군집 4 + 함정 3종, 정답 기준 검증 plot | `data/usage_profiles.csv`, `true_cluster_sizes.csv`, `fig1` |
| 02 | 1차 시도. scaling 없는 K-means (k=4), 분산 지배 진단 | `naive_kmeans_summary.csv`, `naive_crosstab.csv`, `fig2` |
| 03 | 2차 시도. 표준화만 vs log1p + 표준화 (k=4), skewness, 노이즈 15대 추적 | `scaling_compare.csv`, `crosstab_*.csv`, `fig3~4` |
| 04 | k=2~10 sweep (inertia, silhouette, ARI), k=4 vs 5 대조 | `k_sweep.csv`, `k_composition.csv`, `fig5~6` |
| 05 | 최종 군집 (k=5) z-score 프로파일과 비즈니스 라벨 | `cluster_zscore.csv`, `cluster_labels.csv`, `fig7` |

## 확인할 숫자

- 02: ARI 0.238, 사실상 랜덤. 분산 비율 99.8%는 거리 계산을 total_usage 한 열이 다 정했다는 뜻
- 03: ARI 0.723 (표준화만), 0.897 (log1p + 표준화). 로그 변환이 낫지만 노이즈 15대가 allday_low에 흡수됨 (순도 6%)
- 04: k=5에서 silhouette 0.412, ARI 0.928. 지표가 고른 k와 정답 k가 같고 노이즈 15대가 독립 군집으로 돌아옴

## 데이터

`usage_profiles.csv` 800행 x 171열. `device_id`(D0000~D0799), `u_{dow}_h{HH}` 168열 (요일 x 시각 평균 사용 강도, 0~10), `total_usage`(0~5000, 셀 합 x 3.5 + noise), `true_cluster`(morning / allday_low / night / intermittent / noise). 정답 열은 채점 전용이고 클러스터링 입력에 안 들어간다.

세그멘테이션 시리즈 2편은 이 폴더의 생성기를 그대로 복사해 쓴다. 01을 고치면 여기 pipeline 전체를 다시 돌리고 2편 폴더의 생성기도 같이 바꿔야 함.

## 심어둔 구조

- 진짜 군집 4개: morning 230, allday_low 240, night 165, intermittent 150
- 함정 1. `total_usage`만 스케일이 커서 scaling 없이는 이 축이 거리를 지배. `naive_kmeans_summary.csv`의 분산 비율 0.998
- 함정 2. night, intermittent 기기의 35%는 상대 프로파일과 혼합. intermittent 버스트 시간대도 저녁에 치우침
- 함정 3. 24시간 상시 고강도 노이즈 15대. 로그 변환에서 allday_low에 흡수됐다가 k=5에서 복원 (`crosstab_k5.csv`)

## 알려진 문제

- 함정 2는 elbow를 애매하게 만들려던 건데 실패. night 군집에 intermittent 28대가 남는 흔적 정도만 (글에 그렇게 적음)
- 05의 라벨은 KMeans 군집 번호에 손으로 붙인 것. seed나 데이터가 바뀌면 번호가 섞이므로 히트맵을 다시 읽고 다시 붙여야 함
