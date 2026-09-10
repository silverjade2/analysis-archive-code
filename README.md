# analysis-archive-code

Reproduction code for the articles on [analysis-archive](https://analysis-archive.vercel.app), a site where I revisit past data analysis and modeling projects. Each folder rebuilds one project on **synthetic data** with the same schema, plants the signals and traps we actually ran into, and re-scores the original procedure against the planted ground truth. No real service data is included. Written in Korean; the folder READMEs carry the details.

[analysis-archive](https://analysis-archive.vercel.app)에 올린 분석 회고 글의 재현 코드. 글은 예전에 했던 데이터 분석·모델링 프로젝트를 되짚는 것이고, 이 저장소는 그 프로젝트를 같은 스키마의 **가상데이터**로 다시 돌린 코드다. 실제 서비스의 데이터는 한 행도 들어 있지 않고, 회사명·서비스명·실명도 쓰지 않는다. 가상데이터를 쓰는 이유는 정답을 알고 채점하기 위해서다. 생성기가 신호·함정·잠재 변수를 심고, 원래 프로젝트의 절차를 그대로 따라간 뒤, 심어둔 정답으로 그 절차가 무엇을 맞히고 무엇을 놓쳤는지 잰다. 만든 사람과 배경은 사이트의 [소개 페이지](https://analysis-archive.vercel.app/about)에 있다.

글 하나 = 폴더 하나. 폴더명은 글의 slug와 같다.

| 폴더 | 글 | 질문 | 소요 시간 |
| --- | --- | --- | --- |
| [`precursor-prediction-imbalanced/`](./precursor-prediction-imbalanced) | [이상 이벤트 예측: 불균형 데이터에서 전조 신호 모델링](https://analysis-archive.vercel.app/analyses/precursor-prediction-imbalanced) | 양성 0.36%에서 분할·지표·feature 검증을 어떻게 해야 "좋아 보이는 숫자"에 속지 않는가 | 35초 |
| [`usage-pattern-clustering/`](./usage-pattern-clustering) | [사용 패턴 세그멘테이션: 클러스터링의 함정들](https://analysis-archive.vercel.app/analyses/usage-pattern-clustering) | 정답 군집을 심어 두면 scaling·로그 변환·k 선택은 각각 무엇을 살리고 무엇을 지우는가 | 12초 |
| [`loyal-user-prediction-jobplatform/`](./loyal-user-prediction-jobplatform) | [핵심 유저 전환 예측: 저니맵이 가리킨 병목을 모델로 검증하기](https://analysis-archive.vercel.app/analyses/loyal-user-prediction-jobplatform) | 타깃이 정해진 뒤의 행동이 feature에 들어가면 모델은 무엇을 배우는가 | 5분 |
| [`depression-emotion-multimodal-mentalhealth/`](./depression-emotion-multimodal-mentalhealth) | [직장인 번아웃 조기 탐지: 발화로 우울을 분류하는 모델이 실제로 배운 것](https://analysis-archive.vercel.app/analyses/depression-emotion-multimodal-mentalhealth) | 라벨 1과 0이 다른 출처에서만 올 때 분류기는 무엇을 구분하는가, multimodal 비교는 언제 성립하는가 | 3분 40초 |
| [`churn-prediction-b2b-ats/`](./churn-prediction-b2b-ats) | [B2B 고객사 이탈 분류: 타깃을 정의한 열이 feature로 남았을 때](https://analysis-archive.vercel.app/analyses/churn-prediction-b2b-ats) | 실행일에 매달린 라벨과 계약 이력 파생 feature를 걷어내면 무엇이 남는가 | 3분 |

소요 시간은 폴더 README에 적힌 실측이다(Apple Silicon, churn은 Linux x86). 사이트에서 `draft: true`인 글의 코드는 이 저장소에 없다. 별도 private 저장소에서 작업하고, 글을 발행할 때 폴더를 여기로 옮긴다.

## 읽는 법

- `scripts/01_*.py`가 생성기다. 심어둔 구조는 여기 상수와 주석에 있고, 폴더 README의 "심어둔 구조" 표가 그것을 어느 결과 파일에서 확인하는지 가리킨다.
- `truth_*` 열은 생성기의 잠재 변수다. 채점에만 쓰고 feature로는 쓰지 않으며, pipeline 안의 누수 점검이 그것을 코드로 확인한다.
- oracle은 잠재 변수를 아는 판정자의 점수다. 모델이 oracle을 넘으면 답을 본 것이다.
- 글의 재현값은 전부 `outputs/results/*.csv`에서 나온다. 어느 숫자가 어느 파일에서 왔는지는 폴더 README의 "결과 파일" 표에 있다.
- 그림은 분석 스크립트의 원본(`outputs/figures/*.png`)과 사이트 톤으로 다시 그린 것(`outputs/figures/site/*.webp`) 두 벌이다. 후자는 값을 CSV에서 읽고 다시 계산하지 않는다.

## 폴더 구조

```
<slug>/
  scripts/          NN_역할.py, pipeline 순서로 번호. 경로는 파일 위치 기준이라 어디서 실행해도 된다
  data/             생성 데이터 (커밋)
  outputs/figures/  분석 스크립트의 그림. site/ 하위는 사이트용 webp
  outputs/results/  결과 CSV·JSON. 글의 재현값은 전부 여기서 나온다
  README.md         실행 명령·소요 시간·환경, pipeline 표, 데이터 스키마, 심어둔 구조, 결과 파일 표
  requirements.txt
```

## 검증 규약

- seed 고정(`SEED=42`). 폴더 README에 기록된 라이브러리 버전에서는 `data/`를 지우고 다시 돌려도 데이터와 `outputs/results/*.csv`가 byte 단위로 같다. 다른 버전에서는 마지막 자리가 달라질 수 있다.
- 누수 점검: feature 행렬에 타깃과 `truth_*` 열이 없는지 코드로 확인한다. 시간 절단 feature가 있는 폴더는 절단일 이후 이벤트가 집계에 안 들어가는 것을 표본으로 검산한다.
- 생성기를 고치면 pipeline 전체를 다시 돌려 데이터·결과·글의 숫자를 같은 커밋에 넣는다. 코드와 데이터가 어긋난 상태로 두지 않는다.
- 생성 데이터(`data/`)는 커밋한다. 코드를 돌리지 않아도 데이터를 열어볼 수 있게 하기 위해서다. 모델 파일(`*.joblib`)은 커밋하지 않는다.

## 실행 환경

폴더마다 `requirements.txt`와 실행 명령이 있다. 공통은 아래와 같다.

```bash
cd <slug>
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

- Python 3.12 이상. 기록된 값은 3.14(Apple Silicon)와 3.12(Linux x86)에서 나왔다. `requirements.txt`는 폴더에 따라 하한만 두거나(`>=`) 버전을 고정한다(`==`). 숫자를 그대로 재현하려면 폴더 README의 버전을 쓴다.
- LightGBM·XGBoost를 쓰는 폴더는 macOS에서 `brew install libomp`가 필요하다.
- 그림의 한글 폰트는 Pretendard이고, 없으면 시스템 한글 폰트로 대체된다.

## 라이선스

코드는 MIT([LICENSE](./LICENSE)). 글의 본문과 그림은 사이트를 따른다.
