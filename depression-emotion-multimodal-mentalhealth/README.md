# depression-emotion-multimodal-mentalhealth

글: [직장인 번아웃 조기 탐지: 발화로 우울을 분류하는 모델이 실제로 배운 것](https://analysis-archive.vercel.app/analyses/depression-emotion-multimodal-mentalhealth)

<!-- TODO: 배경 문단 직접 쓰기 -->
<!-- 옛 문단: 원본 프로젝트(2024 상반기, 대학원 팀 프로젝트)는 공개 감정 대화 음성 데이터(일상 발화)와 정신건강 상담 스크립트(우울 발화)를 합쳐 우울 진단 분류기와 7클래스 감정 분류기를 만들고, 텍스트 PLM 3종과 음성·텍스트 multimodal을 비교했다. 이 폴더는 그 두 비교를 같은 스키마의 가상데이터로 다시 돌리되, 당시 설계의 문제 두 가지를 데이터에 그대로 심어 놓고 조건을 맞춘 버전과 비교한다. (1) 라벨 1과 0이 서로 다른 출처에서만 와서 문체가 곧 라벨이었고, 그걸 통제할 상담 일상 의도 발화 4,000건은 학습에서 뺐다. (2) multimodal과 텍스트를 다른 N·다른 테스트셋에서 비교했다. -->

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

- Apple Silicon 1 core 기준 01에서 06까지 약 3분 40초, 07과 08은 수 초. 스크립트는 어느 위치에서 실행해도 된다. Python이 스크립트가 있는 폴더를 sys.path 맨 앞에 넣으므로 `common.py`와 `models.py`를 그대로 import한다.
- 환경: Python 3.14.6, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8. 이 버전에서 `data/`를 지우고 다시 돌려도 `data/`와 `outputs/results/*.csv`가 byte 단위로 같다. `requirements.txt`에는 하한만 적었다.
- 제대로 돌았는지 확인할 숫자
  - 01의 shape (39374, 30). 일상 19,374건에 상담 20,000건
  - 03의 `test_as_original` accuracy 0.9944(학습 라벨), 0.9745(실제 우울). 원 방식대로 채점하면 거의 완벽하고 실제 우울로 채점해도 2%p만 빠진다. 학습에서 뺀 4,000건의 우울 예측률 0.9432이 문제다. 우울이 아닌 상담 발화 열에 아홉을 우울로 찍는다. 모델이 배운 것은 우울이 아니라 상담 문체다
  - 04의 v2 `test_counsel_only` accuracy 0.9030. v1의 0.8057에서 오른 값이고, 4,000건을 넣으면 상담 발화 안에서 우울과 아닌 것을 가르기 시작한다는 뜻
  - 05의 (c) 화자 분리 전체 multimodal 0.8180 vs 텍스트 0.8079. 조건을 맞추면 음성이 더하는 것은 1%p 남짓이다. 원 방식 (a)의 2%p 차이는 절반이 비교 조건에서 왔다

## Pipeline

| 스크립트 | 역할 | 출력 |
| --- | --- | --- |
| `01_generate_utterances.py` | 일상 발화 19,374건 + 상담 발화 20,000건(증상 16,000 + 일상 의도 4,000) 생성. 구조 3종 주입 | `data/utterances.csv` |
| `02_eda.py` | 출처별 문체 통계, 라벨 × 출처 구성, 상담 의도 분포, 화자 편중, 음성 feature PCA | `results/eda_*.csv`, `figures/fig1`, `fig7` |
| `03_depression_v1_source.py` | 원 방식: 상담 증상 = 1, 일상 = 0, 상담 일상 의도는 학습에서 제외. 학습에서 뺀 4,000건에 모델을 넣어 채점. oracle | `results/depression_v1_*.csv`, `depression_oracle.csv` |
| `04_depression_v2_intent.py` | 상담 일상 의도를 라벨 0으로 포함. v1과 같은 테스트셋에서 비교 | `results/depression_v1_v2_compare.csv`, `depression_v2_top_features.csv`, `depression_feature_kinds.csv`, `figures/fig2~3` |
| `05_multimodal_protocols.py` | multimodal vs 텍스트를 프로토콜 3개(원 방식 / 같은 subset / 화자 분리)로 비교. N 곡선(seed 3개), 화자 사전확률 baseline, oracle | `results/multimodal_*.csv`, `figures/fig4~5` |
| `06_feature_attribution.py` | 감정별 global 상위 feature, 예문 토큰의 coverage, local 토큰 기여(SHAP plot 재현), 혼동행렬 차이, 융합 채널 가중치 | `results/emotion_top_features_by_class.csv`, `shap_*.csv`, `confusion_*.csv`, `fusion_channel_weight.csv`, `figures/fig6`, `fig8` |
| `07_site_figures.py` | 그림 1\~8을 사이트 톤으로 작도. 값은 02\~06의 CSV에서 읽고 다시 계산하지 않는다 | `outputs/figures/site/*.webp` |
| `08_widget_data.py` | 05의 N 곡선 CSV 두 개를 위젯용 JSON으로 | `results/multimodal_n_curve.json`(사이트 `src/data/multimodal-n-curve.json`으로 복사, 위젯 MultimodalNCurveExplorer) |

`models.py`에 모델 셋이 있다. 텍스트는 char n-gram 2에서 4의 TF-IDF에 Logistic Regression, 음성은 표준화에 LR, multimodal은 두 채널의 out-of-fold 클래스 확률을 이어 붙여 상위 LR이 결합하는 late fusion이다. PLM fine-tuning은 재현하지 않는다. 재현 대상은 실험 구조다. `common.py`에 경로, 원본 기록의 건수와 감정 비중, 우울 진단용 공통 분할 `depression_split()`이 있다. `sitestyle.py`는 loyal 폴더에 같은 파일이 있어 한쪽을 고치면 다른 쪽도 같이 고쳐야 한다.

## 데이터 스키마

`data/utterances.csv` (39,374행 × 30열). 원본의 두 데이터셋을 같은 열로 합쳤다.

| 열 | 설명 |
| --- | --- |
| `utt_id`, `source` | `source`는 daily(일상 대화) / counsel(상담 스크립트) |
| `text` | 어휘 풀에서 토큰을 뽑아 이어 붙인 발화. 문장으로 읽히지는 않는다. 재현 대상은 "어떤 토큰이 어떤 변수를 따르는가" |
| `speaker_id`, `emotion_label` | 일상 발화만. 화자 150명, 감정 7클래스(happiness / angry / neutral / sadness / disgust / surprise / fear) |
| `intent` | 상담 발화만. `정신증상/…`(증상 의도, 8종) 또는 `일상/…`(일상 의도, 6종) |
| 음성 feature 20열 | 일상 발화만. `zcr_mean`, `zcr_std`, `rms_mean`, `rms_std`, `mfcc1..13_mean`, `chroma_mean`, `chroma_std`, `duration`. 원본의 ZCR/RMS/MFCC/Chroma 추출을 흉내낸 합성값 |
| `truth_depressed` | 실제 우울 여부. 상담 증상 발화는 1, 상담 일상 발화는 0, 일상 sadness 발화의 25%가 1 |
| `truth_text_informative_dep`, `truth_text_informative_emo`, `truth_audio_informative` | 그 발화의 텍스트 내용 토큰·감정 토큰·음성 신호가 정답을 가리키는지. oracle 계산용 |

`truth_*` 4열은 생성기의 잠재 변수다. 채점에만 쓰고 feature로는 쓰지 않는다. 상담 발화의 음성 열과 화자 열은 빈값이다. 원본에도 없었다.

## 심어둔 구조

| 구조 | 생성기 | 확인 |
| --- | --- | --- |
| A. 문체(어미, 상담 관용구)는 출처를 따르고 5%는 어긋난다. 우울 내용 토큰은 `truth_depressed`를 따른다 | `build_text()`, `P_STYLE_CROSS`, `Q_DEP_TEXT` | `eda_source_style.csv`(경어체 비율 0.947 vs 0.051, 실제 우울 비율 0.800 vs 0.035), `depression_v1_by_group.csv`(학습에서 뺀 4,000건의 94.3%를 우울로 예측) |
| B. 화자마다 음색 feature(MFCC 7\~13, chroma, duration)에 지문이 있고 감정 분포가 편향돼 있다(Dirichlet 농도 0.4). 감정 신호는 에너지·피치 계열에 실리고 고각성 감정(angry, fear, surprise)에서 크다 | `SPEAKER_FP_*`, `AUDIO_OFFSET`, `SPEAKER_EMO_CONC` | `eda_speaker_bias.csv`(최빈 감정 비율 중앙값 0.498), `multimodal_protocols.csv`(화자 사전확률 baseline 0.4999, 화자 분리 시 0.228), `multimodal_per_class_gain_speaker_split.csv`(angry +0.0575) |
| C. 텍스트 토큰이 정답 감정을 가리킬 확률 0.82, 음성 신호가 실릴 확률 0.70, 우울 내용 토큰이 정답을 가리킬 확률 0.80 | `Q_EMO_TEXT`, `Q_EMO_AUDIO`, `Q_DEP_TEXT` | `depression_oracle.csv`(0.9774), `multimodal_protocols.csv`의 oracle 행(0.94 안팎). 어느 모델도 oracle을 넘지 않는다 |

## 결과 파일

| 파일 | 내용 | 글에서 |
| --- | --- | --- |
| `eda_source_style.csv`, `eda_label_source_cells.csv`, `eda_counsel_intent.csv`, `eda_emotion_share.csv` | 출처별 문체·우울 비율, 라벨 × 출처 칸, 상담 의도 분포, 감정 비중 | 데이터 절 표, fig1 |
| `eda_speaker_bias.csv`, `eda_audio_pca.csv` | 화자 편중, 음성 PCA 설명 분산 | fig7 캡션 |
| `depression_v1_metrics.csv`, `depression_v1_by_group.csv` | v1의 채점 기준별 성능, 그룹별 우울 예측률 | 우울 진단 절의 표 2개 |
| `depression_v1_top_features.csv`, `depression_v2_top_features.csv`, `depression_feature_kinds.csv` | 우울 방향 상위 char n-gram과 토큰 종류 | fig3, 계수 인용 |
| `depression_v1_v2_compare.csv`, `depression_oracle.csv` | v1 / v2 비교, oracle | v2 표 |
| `multimodal_protocols.csv` | 프로토콜 (a)(b)(c)와 oracle, 화자 사전확률 | 프로토콜 표, fig4 |
| `multimodal_n_curve.csv`, `multimodal_n_curve_gain.csv`, `multimodal_n_curve.json` | N × 분할 × 모델 곡선(seed 3개 평균·표준편차) | fig5, 위젯 |
| `multimodal_per_class_gain_speaker_split.csv`, `confusion_*.csv`, `fusion_channel_weight.csv` | 감정별 recall 차이, 혼동행렬 차이, 채널 가중치 | fig6, "음성 채널에 54.4%" |
| `emotion_top_features_by_class.csv`, `shap_example_token_coverage.csv`, `shap_like_local_example.csv` | 감정별 global feature, 예문 토큰 coverage, 예문 한 건의 토큰별 기여 | 해석 절, fig8 |

## 남은 것

- 텍스트 분류기의 상위 feature 표에서 계수가 완전히 같은 char n-gram, 예를 들어 항상 같이 나오는 '울'과 '울었'은 정렬 동률이라 대표 n-gram이 플랫폼에 따라 다를 수 있다. 계수와 순위는 같고 표기만 다르다.
- 토큰 조합 생성기라 완전히 같은 텍스트가 1,769건(4.5%) 있다. 우울 진단 분할에서는 테스트 11,813건 중 729건(6.2%)이 학습셋에 같은 텍스트가 있고, 대부분 짧은 상담 발화다. 감정 분류 분할에서는 5,813건 중 41건(0.7%)이고 그중 라벨까지 같은 것은 68%다. v1과 v2, 프로토콜 (a)(b)(c)는 같은 분할 안에서의 비교라 결론에 영향은 없지만, 절대 accuracy는 그만큼 후하게 잡혀 있을 수 있다.
- 일상 대화 안의 실제 우울 발화는 v2에서도 거의 못 잡는다. 테스트 224건 중 recall 0.03. 원 데이터에 그 라벨이 없어 0으로 학습되기 때문이고, 4,000건을 되살려도 이건 그대로다. 글은 이 숫자를 다루지 않는다.
- 07의 `EMO_KO`는 감정 이름을 자기 자신으로 매핑하는 빈 dict다. 그림 라벨을 영어로 바꿀 때 한국어 매핑만 지우고 껍데기가 남았다. 쓰는 곳도 없다.
