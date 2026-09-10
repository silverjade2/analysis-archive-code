# analysis-archive-code

Reproduction code for the articles on [analysis-archive](https://analysis-archive.vercel.app). Each folder rebuilds one past project on synthetic data with the same schema and re-scores the original procedure against planted ground truth. No real data is included.

글 하나 = 폴더 하나, 폴더명 = 글의 slug. 배경과 해석은 글에 있고 여기에는 코드·데이터·결과만 둔다.

| 폴더 | 글 | 질문 | 소요 시간 |
| --- | --- | --- | --- |
| [`precursor-prediction-imbalanced/`](./precursor-prediction-imbalanced) | [이상 이벤트 예측: 불균형 데이터에서 전조 신호 모델링](https://analysis-archive.vercel.app/analyses/precursor-prediction-imbalanced) | 양성 0.36%에서 분할·지표·feature 검증을 어떻게 해야 "좋아 보이는 숫자"에 속지 않는가 | 35초 |
| [`usage-pattern-clustering/`](./usage-pattern-clustering) | [사용 패턴 세그멘테이션: 클러스터링의 함정들](https://analysis-archive.vercel.app/analyses/usage-pattern-clustering) | 정답 군집을 심어 두면 scaling·로그 변환·k 선택은 각각 무엇을 살리고 무엇을 지우는가 | 12초 |
| [`loyal-user-prediction-jobplatform/`](./loyal-user-prediction-jobplatform) | [핵심 유저 전환 예측: 저니맵이 가리킨 병목을 모델로 검증하기](https://analysis-archive.vercel.app/analyses/loyal-user-prediction-jobplatform) | 타깃이 정해진 뒤의 행동이 feature에 들어가면 모델은 무엇을 배우는가 | 5분 |
| [`depression-emotion-multimodal-mentalhealth/`](./depression-emotion-multimodal-mentalhealth) | [직장인 번아웃 조기 탐지: 발화로 우울을 분류하는 모델이 실제로 배운 것](https://analysis-archive.vercel.app/analyses/depression-emotion-multimodal-mentalhealth) | 라벨 1과 0이 다른 출처에서만 올 때 분류기는 무엇을 구분하는가, multimodal 비교는 언제 성립하는가 | 3분 40초 |
| [`churn-prediction-b2b-ats/`](./churn-prediction-b2b-ats) | [B2B 고객사 이탈 분류: 타깃을 정의한 열이 feature로 남았을 때](https://analysis-archive.vercel.app/analyses/churn-prediction-b2b-ats) | 실행일에 매달린 라벨과 계약 이력 파생 feature를 걷어내면 무엇이 남는가 | 3분 |

## 폴더 구조

```
<slug>/
  scripts/          NN_역할.py, pipeline 순서. 어디서 실행해도 된다
  data/             생성 데이터 (커밋)
  outputs/figures/  그림. site/ 하위는 사이트용 webp
  outputs/results/  결과 CSV·JSON. 글의 재현값은 전부 여기서 나온다
  README.md         실행, pipeline 표, 데이터 스키마, 심어둔 구조, 결과 파일 표
  requirements.txt
```

`truth_*` 열은 생성기의 잠재 변수다. 채점에만 쓰고 feature로는 쓰지 않는다. oracle은 잠재 변수를 아는 판정자의 점수이고, 모델이 oracle을 넘으면 답을 본 것이다.

## 실행

```bash
cd <slug>
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

- seed 고정(`SEED=42`). 폴더 README에 적힌 버전에서는 `data/`를 지우고 다시 돌려도 데이터와 결과 CSV가 byte 단위로 같다. 다른 버전에서는 마지막 자리가 달라질 수 있다.
- Python 3.12 이상. LightGBM·XGBoost를 쓰는 폴더는 macOS에서 `brew install libomp`.
- `draft: true`인 글의 코드는 발행 전까지 별도 private 저장소에 있다.

## 라이선스

코드는 MIT([LICENSE](./LICENSE)). 글과 그림은 사이트를 따른다.
