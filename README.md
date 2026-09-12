# analysis-archive-code

[analysis-archive](https://analysis-archive.vercel.app) 글의 재현 코드. 글 하나 = 폴더 하나, 폴더명 = 글 slug.

폴더마다 옛 프로젝트 하나를 같은 스키마의 가상 데이터로 다시 만들고, 원래 분석이 걸려 넘어진 구조를 심어둔 뒤 원본 절차를 그대로 돌려 심어둔 정답과 대조한다. 실제 데이터는 없음. 배경과 해석은 글에 있고 여기엔 숫자를 다시 내는 데 필요한 것만.

| 폴더 | 글 | 소요 |
| --- | --- | --- |
| [`precursor-prediction-imbalanced/`](./precursor-prediction-imbalanced) | [이상 이벤트 예측: 불균형 데이터에서 전조 신호 모델링](https://analysis-archive.vercel.app/analyses/precursor-prediction-imbalanced) | 35초 |
| [`usage-pattern-clustering/`](./usage-pattern-clustering) | [사용 패턴 세그멘테이션: 클러스터링의 함정들](https://analysis-archive.vercel.app/analyses/usage-pattern-clustering) | 12초 |
| [`loyal-user-prediction-jobplatform/`](./loyal-user-prediction-jobplatform) | [핵심 유저 전환 예측: 저니맵이 가리킨 병목을 모델로 검증하기](https://analysis-archive.vercel.app/analyses/loyal-user-prediction-jobplatform) | 5분 |
| [`depression-emotion-multimodal-mentalhealth/`](./depression-emotion-multimodal-mentalhealth) | [직장인 번아웃 조기 탐지: 발화로 우울을 분류하는 모델이 실제로 배운 것](https://analysis-archive.vercel.app/analyses/depression-emotion-multimodal-mentalhealth) | 3분 40초 |
| [`churn-prediction-b2b-ats/`](./churn-prediction-b2b-ats) | [B2B 고객사 이탈 분류: 타깃을 정의한 열이 feature로 남았을 때](https://analysis-archive.vercel.app/analyses/churn-prediction-b2b-ats) | 3분 |
| [`wellness-score-sigmoid-design/`](./wellness-score-sigmoid-design) | [지금 바이탈 어때요? rPPG Vital sign Data로 만드는 Wellness score: 의료 데이터를 이해할 수 있는 웰니스 언어로](https://analysis-archive.vercel.app/analyses/wellness-score-sigmoid-design) | 10초 |

## 실행

```bash
cd <slug>
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

Python 3.12 이상. LightGBM이나 XGBoost 쓰는 폴더는 macOS에서 `brew install libomp` 먼저. `requirements.txt`는 하한만 적었고 재현에 쓴 버전은 폴더 README에 있다. 그 버전에서는 `data/`를 지우고 다시 돌려도 결과 CSV가 같고, 다른 버전에서 어디가 흔들리는지도 폴더 README에.

## 폴더 구조

```
<slug>/
  scripts/               NN_역할.py, 번호 순서. 어디서 실행해도 됨
  data/                  생성 데이터
  outputs/results/       결과 CSV, JSON. 글의 숫자는 전부 여기서
  outputs/figures/       그림 (site/는 사이트용 webp, 일부 폴더)
  README.md              실행, 스크립트 표, 확인할 숫자, 알려진 문제
```

`truth_*` 열은 생성기의 잠재 변수라 채점에만 쓰고 feature에는 안 넣는다. oracle은 그 변수를 아는 판정자의 점수. 모델이 oracle을 넘으면 답을 본 것.

draft 글의 코드는 발행 전까지 별도 private 저장소에 있음.

## 라이선스

코드는 MIT ([LICENSE](./LICENSE)). 글과 그림은 사이트를 따름.
