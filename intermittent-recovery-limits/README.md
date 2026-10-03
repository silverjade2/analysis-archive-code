# intermittent-recovery-limits

[취업했는지 떠났는지 모르는 고객](https://analysis-archive.vercel.app/analyses/intermittent-recovery-limits) 재현 코드.

구직자 군집화에서 해석이 안 된 두 고객군(취업 완료/이탈이 안 갈리는 비활성, 입력 빈약)을 가상 구직자 6,000명으로 재현함(03~05). 실제 구직 플랫폼 데이터는 없고 구조만 옮김. 00~02는 같은 질문을 정답 구조를 아는 기기 데이터에 대본 통제 실험: [usage-pattern-clustering](../usage-pattern-clustering)의 K-means(k=5)가 어떤 k로도 회수하지 못한 intermittent 28대 추적.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

10초 안쪽. Python 3.14.7, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2. seed 42.

## 스크립트

| | 역할 | 주요 출력 |
|---|---|---|
| 00 | 기기 800대 사용 프로파일 생성. usage-pattern-clustering의 01과 같은 생성기(검증 그림 파일명만 `fig0`) | `data/usage_profiles.csv` |
| 01 | 기준선, GMM, DBSCAN, 버스트 feature 4방법 비교 | `recovery_comparison.csv`, `dbscan_eps_grid.csv`, fig1, fig2 |
| 02 | 해부: blur 복원, 피크 시각, 모양 지표, GMM 사후확률, 강화 feature | `missed_anatomy.csv`, fig3 |
| 03 | 가상 구직자 6,000명. 잠재 유형 4개, 비활성 30%(취업 완료/이탈), 입력 빈약 25% | `data/jobseekers.csv` |
| 04 | k sweep, k=5 구성, 비활성 군집 재분할, 채용 진행 상태로 확인되는 범위, 최근 경력 신입 규칙 | `js_*.csv` |
| 05 | 구직자 그림 | fig4 |
| 06 | 사이트용 그림 5장, CSV에서 읽기만 함 | `outputs/figures/site/` |

02는 00을 모듈로 불러 같은 seed로 난수를 다시 밟고, 생성 때만 있던 혼합(blur) 플래그를 복원함. 그래서 00의 생성 순서를 바꾸면 02도 깨짐. `truth_*` 열은 04의 평가에만 씀.

## 확인할 숫자

- 01: 미회수 28대 회수 기준선 0, GMM 3, DBSCAN 0, 버스트 feature 0. ARI 0.928 / 0.925 / 0.920 / 0.907
- 02: 28대 중 혼합 기기 10대, 피크가 야간 창(21~02시) 25대, 강화 feature로도 2/28
- 04: silhouette 0.283~0.308로 평평. 비활성 군집 재분할 ARI -0.001(K-means, GMM 둘 다). 채용 진행 상태로 확인되는 건 844명 중 161명
- 04: 최근 경력 신입 규칙 recall 0.751, precision 0.683. 놓친 339명은 전부 입력 빈약

## 알려진 문제

- 채용 진행 상태가 취업 완료자의 35%에만 남는다는 건 생성기 가정. 실제 비율은 모름
