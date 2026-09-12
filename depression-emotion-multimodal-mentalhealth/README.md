# depression-emotion-multimodal-mentalhealth

[직장인 번아웃 조기 탐지: 발화로 우울을 분류하는 모델이 실제로 배운 것](https://analysis-archive.vercel.app/analyses/depression-emotion-multimodal-mentalhealth) 재현 코드.

원본은 상담 스크립트와 감정 대화 음성 데이터로 했던 우울 분류 + multimodal 감정 분류. 여기서는 두 출처를 흉내낸 가상 발화를 만들어 원 프로젝트의 실험 구조를 그대로 돌리고, accuracy 0.98짜리 모델이 실제로 뭘 배웠는지 본다 (실제 데이터 없음). PLM fine-tuning은 재현하지 않고 char n-gram TF-IDF + LR로 대신한다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

01~06이 3분 40초 정도 (1 core), 07과 08은 몇 초. 경로는 파일 위치 기준이라 어디서 실행해도 된다.

재현에 쓴 버전: Python 3.14, numpy 2.4, pandas 3.0, scikit-learn 1.8, scipy 1.17, matplotlib 3.10. 이 조합에서는 `data/`를 지우고 다시 돌려도 데이터와 결과 CSV가 동일. `requirements.txt`에는 하한만.

## 스크립트

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 일상 발화 19374건 + 상담 발화 20000건 (증상 16000 + 일상 의도 4000) 생성. 구조 3종 주입 | `data/utterances.csv` |
| 02 | EDA. 출처별 문체, 라벨 x 출처 칸, 상담 의도, 화자 편중, 음성 PCA | `eda_*.csv`, `fig1`, `fig7` |
| 03 | v1 원 방식. 상담 증상 = 1, 일상 = 0, 상담 일상 의도는 학습 제외. 뺀 4000건 채점, oracle | `depression_v1_*.csv`, `depression_oracle.csv` |
| 04 | v2. 상담 일상 의도를 라벨 0으로 포함, 같은 테스트셋에서 v1과 비교 | `depression_v1_v2_compare.csv`, `fig2~3` |
| 05 | multimodal vs 텍스트를 프로토콜 3개로. N 곡선 (seed 3), 화자 사전확률 baseline, oracle | `multimodal_*.csv`, `fig4~5` |
| 06 | 감정별 global feature, 예문 토큰 coverage, local 기여 (SHAP plot 흉내), 혼동행렬 차이, 채널 가중치 | `emotion_top_features_by_class.csv`, `shap_*.csv`, `confusion_*.csv`, `fig6`, `fig8` |
| 07 | 그림 1~8 사이트 톤. CSV만 읽음 | `outputs/figures/site/*.webp` |
| 08 | N 곡선 CSV 두 개를 위젯 JSON으로 | `multimodal_n_curve.json` (사이트 `src/data/multimodal-n-curve.json`) |

`models.py`에 모델 셋. 텍스트는 char n-gram(2~4) TF-IDF + LR, 음성은 표준화 + LR, multimodal은 두 채널의 OOF 클래스 확률을 이어 붙여 상위 LR이 결합하는 late fusion. `common.py`에 경로, 원본 기록의 건수와 감정 비중, 우울 진단용 공통 분할 `depression_split()`. `sitestyle.py`는 loyal 폴더에 같은 파일이 있어서 한쪽 고치면 다른 쪽도.

## 확인할 숫자

- 01: shape (39374, 30). 일상 19,374 + 상담 20,000
- 03: `test_as_original` accuracy 0.994 (학습 라벨), 0.975 (실제 우울). 문제는 학습에서 뺀 4000건의 우울 예측률 0.943. 우울 아닌 상담 발화 열에 아홉을 우울로 찍음, 배운 건 우울이 아니라 상담 문체
- 04: v2 `test_counsel_only` accuracy 0.903, v1은 0.806. 4000건을 넣으면 상담 발화 안에서 우울과 아닌 것을 가르기 시작
- 05: (c) 화자 분리 전체에서 multimodal 0.818 vs 텍스트 0.808. 조건을 맞추면 음성이 더하는 건 1%p 남짓, 원 방식 (a)의 2%p 차이는 절반이 비교 조건에서 옴

## 데이터

`utterances.csv` 39,374행 x 30열. 원본의 두 데이터셋을 같은 열로 합침. `source`(daily / counsel), `text`(어휘 풀에서 토큰을 뽑아 이어 붙인 것, 문장으로 읽히진 않음), `speaker_id`와 `emotion_label`(일상만. 화자 150명, 감정 7클래스), `intent`(상담만. `정신증상/...` 8종 또는 `일상/...` 6종), 음성 feature 20열(일상만. zcr, rms, mfcc1~13, chroma, duration을 흉내낸 합성값), `truth_depressed`(상담 증상 1, 상담 일상 0, 일상 sadness의 25%가 1), `truth_*_informative` 3열(oracle 계산용).

`truth_*` 4열은 채점 전용이고 feature에 안 들어간다. 상담 발화의 음성 열과 화자 열은 빈값 (원본에도 없었음).

## 심어둔 구조

- A. 문체(어미, 관용구)는 출처를 따르고 5%만 어긋남. 우울 내용 토큰은 `truth_depressed`를 따름. `eda_source_style.csv`의 경어체 비율 0.947 vs 0.051, 실제 우울 비율 0.800 vs 0.035
- B. 화자마다 음색 feature(MFCC 7~13, chroma, duration)에 지문, 감정 분포 편향 (Dirichlet 농도 0.4). 감정 신호는 에너지/피치 계열에, 고각성 감정에서 크게. 화자 사전확률 baseline 0.500, 화자 분리 시 0.228 (`multimodal_protocols.csv`)
- C. 텍스트 토큰이 정답 감정을 가리킬 확률 0.82, 음성 신호가 실릴 확률 0.70, 우울 내용 토큰이 정답일 확률 0.80. oracle 0.977 (우울), 0.94 안팎 (감정). 어느 모델도 oracle을 안 넘음

## 알려진 문제

- 토큰 조합 생성기라 완전히 같은 텍스트가 1,769건 (4.5%). 우울 진단 테스트 11,813건 중 729건은 학습셋에 같은 텍스트가 있음 (대부분 짧은 상담 발화). v1/v2, 프로토콜 (a)(b)(c)는 같은 분할 안 비교라 결론엔 영향 없지만 절대 accuracy는 그만큼 후할 수 있음
- 일상 대화 안의 실제 우울 발화는 v2도 거의 못 잡음 (테스트 224건, recall 0.03). 원 데이터에 그 라벨이 없어 0으로 학습되기 때문. 글은 이 숫자를 안 다룸
