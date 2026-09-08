# depression-emotion-multimodal-mentalhealth

가상데이터로 재현한 "한국어 우울·감정 분류 (텍스트 + 음성)" 프로젝트. 글은 [analysis-archive](https://analysis-archive.vercel.app)에 있다.

원본 프로젝트(2024 상반기, 대학원 팀 프로젝트)는 공개 감정 대화 음성 데이터(일상 발화)와 정신건강 상담 스크립트(우울 발화)를 합쳐 우울 진단 분류기와 7클래스 감정 분류기를 만들고, 텍스트 PLM 3종과 음성·텍스트 멀티모달을 비교했다. 이 저장소는 그 두 비교를 같은 스키마의 가상데이터로 다시 돌리되, 당시 설계의 문제 두 가지((1) 라벨 1과 0이 서로 다른 출처에서만 와서 문체가 곧 라벨이었고, 그걸 통제할 상담 일상 의도 발화는 학습에서 뺐다, (2) 멀티모달과 텍스트를 다른 N·다른 테스트셋에서 비교했다)를 데이터에 그대로 심어 놓고, 조건을 맞춘 버전과 비교한다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd notebooks && for s in 0*.py; do ../.venv/bin/python "$s"; done
```

Apple Silicon 1 core 기준 01~06 약 3분 40초, 07·08은 수 초. 시드 고정(`SEED=42`). 스크립트는 `notebooks/` 안에서 실행한다(`common.py`를 같은 폴더에서 import). 환경: Python 3.14.6, `requirements.txt`의 고정 버전(numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8). 두 번 실행해 `data/`와 `outputs/results/*.csv`가 byte 단위로 같은 것을 확인했다.

## 파이프라인

| 스크립트 | 역할 |
| --- | --- |
| `01_generate_utterances.py` | 일상 발화 19,374건 + 상담 발화 20,000건(증상 16,000 + 일상 의도 4,000) 생성. 문체↔출처 / 우울 내용↔`truth_depressed` 분리, 화자 지문·화자별 감정 편향, 채널별 정보성 플래그 주입 |
| `02_eda.py` | 출처별 문체 통계, 라벨×출처 구성, 상담 의도 분포, 화자 편중, 음성 피처 PCA |
| `03_depression_v1_source.py` | 원 방식: 상담 증상 = 1, 일상 = 0, 상담 일상 의도는 학습에서 제외. 학습에서 뺀 4,000건에 모델을 넣어 채점 |
| `04_depression_v2_intent.py` | 상담 일상 의도를 라벨 0으로 학습에 포함. v1과 같은 테스트셋에서 비교 |
| `05_multimodal_protocols.py` | 멀티모달 vs 텍스트를 프로토콜 3개(원 방식 / 같은 서브셋 / 화자 분리)로 비교. N 곡선, 화자 사전확률 베이스라인, 오라클 |
| `06_feature_attribution.py` | 감정별 전역 상위 피처, SHAP 예문 토큰의 전역 커버리지, 로컬 토큰 기여(SHAP 플롯 재현), 혼동행렬 차이, 융합 채널 가중치 |
| `07_site_figures.py` | 그림 1~8을 사이트 톤(Pretendard, 사이트 팔레트, webp)으로 다시 작도. 값은 02~06의 CSV에서 읽고 다시 계산하지 않는다(`sitestyle.py` 공용). 출력 `outputs/figures/site/` |
| `08_widget_data.py` | 05의 N 곡선 CSV 두 개를 글의 인터랙티브 위젯용 JSON(`outputs/results/multimodal_n_curve.json`)으로. 사이트의 `src/data/multimodal-n-curve.json`에 복사해 정적 임베드 |

결과는 `outputs/results/*.csv`, 그림은 `outputs/figures/*.png`(02~06 원본 작도)와 `outputs/figures/site/*.webp`(사이트용). `models.py`에 텍스트(char n-gram TF-IDF + LR), 음성(표준화 + LR), 멀티모달(채널 확률 late fusion) 분류기가 있다. PLM fine-tuning은 재현하지 않는다. 재현 대상은 실험 구조다.

## 스키마

`data/utterances.csv` 한 파일. 원본의 두 데이터셋을 같은 열로 합쳤다.

- 공통: `utt_id`, `source`(daily/counsel), `text`
- 일상 발화만: `speaker_id`, `emotion_label`(7클래스), 음성 피처 20열(`zcr_*`, `rms_*`, `mfcc1..13_mean`, `chroma_*`, `duration`. 원본의 ZCR/RMS/MFCC/Chroma 추출을 흉내낸 합성값)
- 상담 발화만: `intent`(정신증상/… 또는 일상/…)
- `truth_*` 4열은 생성기의 잠재 변수. 채점에만 쓰고 피처로는 쓰지 않는다.

`data/`는 커밋한다. 코드를 돌리지 않아도 생성 데이터를 열어볼 수 있게 하기 위해서다. 생성기(`01_generate_utterances.py`)를 고치면 파이프라인 전체를 다시 돌려 데이터와 결과를 같은 커밋에 넣는다.

## 알려진 한계

- 텍스트 분류기의 상위 피처 표에서 계수가 완전히 같은 char n-gram(항상 같이 나오는 n-gram, 예: '울'/'울었')은 정렬 동률이라 대표 n-gram이 플랫폼에 따라 다를 수 있다. 계수와 순위는 같고 표기만 다르다.
