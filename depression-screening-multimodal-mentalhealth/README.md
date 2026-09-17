# depression-screening-multimodal-mentalhealth

[직장인 번아웃 조기 탐지: 발화로 우울을 분류하는 모델이 실제로 배운 것](https://analysis-archive.vercel.app/analyses/depression-screening-multimodal-mentalhealth) 재현 코드.

원본(2024 상반기)은 AI-Hub 공개 데이터 두 개로 한국어 우울 분류 모델과 감정 분류 모델을 만든 프로젝트. 여기서는 그 데이터를 쓰지 않고 스키마만 가져와 같은 모양의 가상 데이터를 만든 뒤, 스크리닝 파이프라인을 처음부터 다시 돌려 모델이 무엇을 배우는지 본다 (실제 데이터 없음. 원본 D1은 지정 분석 환경에서만 접근 가능한 보건의료 데이터).

- D1 정신건강진단 및 예측을 위한 멀티모달 데이터 (전남대학교병원, 2021): 참가자 2,000명 x 음성 과제 8개, PHQ-9와 2단계 클래스, 환자군/대조군. 공개 기준 환자군 56.5%, 우울 클래스 41.8%, 여성 64.7%
- D2 감성 대화 말뭉치 (미디어젠, 2020): 6대 감정과 상황, 연령대가 붙은 대화 코퍼스

생성 데이터는 녹음 16,000건(2,000명 x 8과제)과 감정 문장 24,000개.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd pipeline && for s in 0*.py 1*.py 9*.py; do ../.venv/bin/python "$s"; done
```

Apple Silicon 기준 35분쯤. 대부분 06(시드 3 x 분할 2 x 모델 4 교차검증). 스크립트는 `pipeline/` 안에서 실행(`_path.py`가 `src/`를 경로에 넣음).

환경: Python 3.14.6, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8, kiwipiepy 0.23.2, Pillow 12.3.0 (macOS 15, arm64). 이 조합에서 `data/`와 `outputs/`를 지우고 다시 돌려도 CSV와 그림이 byte 단위로 같음. `data/`는 커밋해 둠. 생성기를 고치면 전체를 다시 돌려 데이터와 결과를 같은 커밋에.

## 구성

`src/`에 생성기와 모델(config, lexicon, textgen, audiogen, features, models, evalutil), `pipeline/`에 번호 순 스크립트.

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 참가자 2,000명. 그룹, 성별, 나이, 직장인 여부, PHQ-9, 라벨 | `data/participants.csv`, `cohort_summary.csv` |
| 02 | 녹음 16,000건. STT 텍스트 + 음향 피처 20차원 | `data/recordings.csv` |
| 03 | 감정 문장 24,000개 (6감정 x 상황 7종) | `data/emotion_corpus.csv` |
| 04 | Kiwi 형태소 토큰 캐시 | `data/*_tokens.csv` |
| 05 | 코호트 구조, 과제별 통계, 치료 언급률, 음향 피처 분산 분해 | `eda_*.csv` |
| 06 | 녹음 랜덤 vs 참가자 분할 x 텍스트 / 음성 GBM / 음성 k-NN / 융합, 시드 3개 | `split_comparison*.csv` |
| 07 | 과제 하나씩만 쓴 AUC, 참가자 집계 범위 | `task_ablation.csv`, `task_aggregation.csv` |
| 08 | 환자군 라벨 vs PHQ 라벨, 치료 어휘 마스킹 | `label_definition_*.csv`, `top_features_*.csv` |
| 09 | 감정 분류기 성능과 스크리닝 전이 | `emotion_*.csv` |
| 10 | 상위 피처 유형, 하위군 AUC(부트스트랩 CI), 보정 | `subgroup_auc.csv`, `calibration_deciles.csv` |
| 11 | 원본 톤 그림 8장 | `outputs/figures/*.png` |
| 12 | 같은 8장 사이트 톤 (`sitestyle.py`, CSV만 읽음) | `outputs/figures/site/*.webp` |
| 13 | 위젯 데이터 | `split_comparison.json` |
| 99 | 누수 점검, 상한 확인, 글에 쓸 숫자 manifest | `number_manifest.csv` |

## 확인할 숫자

- 06: speaker_prior의 참가자 AUC 1.0 (녹음 랜덤 분할이면 참가자 신원만으로 다 맞힌다는 뜻). 음성 k-NN 참가자 AUC 0.927(녹음 랜덤) -> 0.839(참가자 분리), 텍스트는 0.931 -> 0.930
- 01: 환자군인데 비우울 351명, 대조군인데 우울 51명
- 09: 감정 전이 참가자 AUC 0.626 (아래 Kiwi 차이 참고)

`truth_*` 컬럼(템플릿 가중치, 치료 언급 여부)은 생성기의 잠재 변수라 채점에만 쓰고 피처에는 안 넣음. 99가 매 실행 확인.

## 심어둔 구조

- 참가자 누수. 한 사람이 8건 녹음, 음색 계열 피처(MFCC, f0)는 사람 안에서 안정적. 녹음 단위 랜덤 분할이면 같은 사람이 학습과 평가 양쪽에
- 그룹 != 라벨. 환자군에도 치료 중이라 PHQ-9 낮은 사람, 대조군에도 높은 사람. 환자군은 면담에서 약, 진료, 선생님 같은 치료 맥락을 자주 말해서 라벨을 환자군 여부로 두면 모델이 그 어휘를 배움
- 과제별 신호. 숫자 세기와 낭독은 텍스트가 전원 같아 텍스트 채널에 정보 없음, 서술 과제 5개는 반대
- 신호 강도 상수(`config.py`의 TEXT_SIGNAL, AUDIO_SIGNAL, 화자 지문, 치료 언급 확률)는 원본 프로젝트에서 나온 텍스트, 음성 AUC 수준에 맞춰 조정한 값

## 알려진 문제

- Kiwi 형태소 결과가 플랫폼에 따라 다름. 같은 kiwipiepy 0.23.2인데 Linux(Python 3.12) 토큰 캐시와 macOS 캐시가 16,000건 중 1,179건에서 달랐음("창 밖"이 "창밖"으로 붙는 식). 01~03은 같고 04부터 갈려서 텍스트 AUC는 셋째 자리, 감정 전이 AUC는 0.611 -> 0.626으로 둘째 자리에서 움직임. 커밋된 `data/`, `outputs/`는 macOS 기준
- 11은 Noto Sans CJK를 찾고 없으면 AppleGothic. 둘 다 없으면 한글이 깨짐
