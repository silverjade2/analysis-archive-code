# analysis-archive-code

[analysis-archive](https://analysis-archive.vercel.app) 글에 딸린 재현 코드. 글 하나 = 폴더 하나이고, 폴더명은 글의 slug와 같다.

| 폴더 | 글 |
| --- | --- |
| [`precursor-prediction-imbalanced/`](./precursor-prediction-imbalanced) | [이상 이벤트 예측: 불균형 데이터에서 전조 신호 모델링](https://analysis-archive.vercel.app/analyses/precursor-prediction-imbalanced) |
| [`usage-pattern-clustering/`](./usage-pattern-clustering) | [사용 패턴 세그멘테이션: 클러스터링의 함정들](https://analysis-archive.vercel.app/analyses/usage-pattern-clustering) |
| [`auc-illusion-imbalanced/`](./auc-illusion-imbalanced) | 불균형 분류에서 AUC 0.94의 착시: 지표가 만드는 안심 (draft) |
| [`intermittent-recovery-limits/`](./intermittent-recovery-limits) | 피처 표현의 한계: 2호에서 못 찾은 28대의 행방 (draft) |
| [`loyal-user-prediction-jobplatform/`](./loyal-user-prediction-jobplatform) | [핵심 유저 전환 예측: 저니맵이 가리킨 병목을 모델로 검증하기](https://analysis-archive.vercel.app/analyses/loyal-user-prediction-jobplatform) |
| [`depression-emotion-multimodal-mentalhealth/`](./depression-emotion-multimodal-mentalhealth) | [직장인 번아웃 조기 탐지: 발화로 우울을 분류하는 모델이 실제로 배운 것](https://analysis-archive.vercel.app/analyses/depression-emotion-multimodal-mentalhealth) |
| [`churn-prediction-b2b-ats/`](./churn-prediction-b2b-ats) | [B2B 고객사 이탈 분류: 타깃을 정의한 열이 feature로 남았을 때](https://analysis-archive.vercel.app/analyses/churn-prediction-b2b-ats) |

## 폴더 구조

```
<slug>/
  notebooks/        NN_역할.py — 파이프라인 순서대로 번호. 경로는 파일 위치 기준이라 어디서 실행해도 된다
  data/             생성 데이터 (커밋)
  outputs/figures/  그림 figN_내용.png (사이트용 webp는 site/ 하위)
  outputs/results/  결과 CSV·JSON — 글의 재현값은 전부 여기서 나온다
  README.md         실행 명령, 소요 시간, 환경, 파이프라인 표
  requirements.txt
```

각 폴더의 README에 실행 명령과 소요 시간이 있다. 시드는 고정이고, 생성 데이터(`data/`)는 커밋한다 — 코드를 돌리지 않아도 데이터를 열어볼 수 있게 하기 위해서다. 생성기를 고치면 파이프라인 전체를 다시 돌려 데이터·결과·글의 숫자를 같은 커밋에 넣는다.
