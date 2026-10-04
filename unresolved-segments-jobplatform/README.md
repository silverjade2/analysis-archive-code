# unresolved-segments-jobplatform

[취업했는지 떠났는지 모르는 고객](https://analysis-archive.vercel.app/analyses/unresolved-segments-jobplatform) 재현 코드.

구직자 군집화에서 해석이 안 된 두 고객군(취업 완료/이탈이 안 갈리는 비활성, 입력 빈약)을 가상 구직자 6,000명으로 재현함. 실제 구직 플랫폼 데이터는 없고 구조만 옮김.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

10초 안쪽. Python 3.14.7, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2. seed 42.

## 스크립트

| | 역할 | 주요 출력 |
|---|---|---|
| 01 | 가상 구직자 6,000명. 잠재 유형 4개, 비활성 30%(취업 완료/이탈), 입력 빈약 25% | `data/jobseekers.csv` |
| 02 | k sweep, k=5 구성, 비활성 군집 재분할, 채용 진행 상태로 확인되는 범위, 최근 경력 신입 규칙 | `js_*.csv` |
| 03 | 분석 그림 | fig1 |
| 04 | 사이트용 그림 4장, CSV에서 읽기만 함 | `outputs/figures/site/` |

`truth_*` 열은 02의 평가에만 쓰고 군집 feature에는 넣지 않음. 비활성 군집 재분할은 군집 전체로 학습하고, 평가만 취업 완료/이탈 행에서 함.

## 확인할 숫자

- silhouette 0.283~0.308로 평평. k=5는 입력 빈약 사용자가 활동 수준별로 갈라지는 가장 작은 k
- 비활성 군집 재분할 ARI -0.001(K-means, GMM 둘 다). 채용 진행 상태로 확인되는 건 844명 중 161명
- 최근 경력 신입 규칙 recall 0.751, precision 0.683. 놓친 339명은 전부 입력 빈약

## 알려진 문제

- 채용 진행 상태가 취업 완료자의 35%에만 남는다는 건 생성기 가정. 실제 비율은 모름
- 군집 이름(04의 `CLUSTER_NAMES`)은 seed 42의 군집 번호에 붙인 것이라 seed를 바꾸면 다시 붙여야 함
