# usage-pattern-clustering

가상데이터로 재현한 "사용 패턴 세그멘테이션: 클러스터링의 함정들" 프로젝트. 글은 [analysis-archive](https://analysis-archive.vercel.app/analyses/usage-pattern-clustering)에 있다.

기기 800대의 시간대별 사용 프로파일(7일 × 24시간)에 진짜 군집 4개와 함정 3종(total_usage 스케일 지배 / 군집 경계 흐림 / 15대 노이즈 소군집)을 심어 두고, 스케일링 없는 1차 시도의 실패, 로그 변환의 트레이드오프, k 선택 지표가 가리킨 k=5의 실체까지를 정답을 알고 채점하며 기록한다. `true_cluster` 열은 채점에만 쓰고 피처로는 쓰지 않는다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in notebooks/0*.py; do .venv/bin/python "$s"; done
```

Apple Silicon 기준 전체 약 12초. 시드 고정(`SEED=42`) — `data/`를 지우고 다시 돌려도 CSV가 바이트 단위로 같다.

환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, scipy 1.18, matplotlib 3.11.

## 파이프라인

| 스크립트 | 역할 |
| --- | --- |
| `01_generate_usage_profiles.py` | 가상데이터 생성 (기기 800대, 진짜 군집 4개 + 함정 3종) + 정답 기준 검증 플롯 |
| `02_cluster_naive.py` | 1차 시도: 스케일링 없는 K-means(k=4) + 분산 지배 진단 |
| `03_cluster_scaled.py` | 2차 시도: 표준화 vs 로그+표준화 비교 (k=4) |
| `04_select_k.py` | k=2~10 스윕 (엘보우·실루엣·ARI) + k=4/5 대조 |
| `05_profile_clusters.py` | 최종 군집(k=5) z-score 프로파일링 + 비즈니스 라벨링 |

그림은 `outputs/figures/fig1~7.png`. 4호 글([intermittent-recovery-limits](../intermittent-recovery-limits))은 이 폴더의 생성기를 그대로 복사해 쓴다.

`data/usage_profiles.csv`는 커밋한다. 생성기(`01_`)를 고치면 파이프라인 전체를 다시 돌려 데이터·결과·글의 숫자를 같은 커밋에 넣고, 4호 폴더의 `00_generate_usage_profiles.py`도 같이 바꾼다.
