# wellness-score-sigmoid-design

[지금 바이탈 어때요? rPPG Vital sign Data로 만드는 Wellness score: 의료 데이터를 이해할 수 있는 웰니스 언어로](https://analysis-archive.vercel.app/analyses/wellness-score-sigmoid-design) 재현 코드.

원본은 웨어러블 바이탈 6항목(체온, 맥박, 산소포화도, 스트레스, 수축기/이완기 혈압)으로 100점 웰니스 점수와 4단계 스테이지를 내던 프로토타입 노트북. 여기서는 그 점수 함수와 단계 규칙을 그대로 옮겨 합성 데이터에 돌리고, 정규화 시그모이드 + 단계 캡으로 바꾼 버전과 나란히 놓는다 (실제 사용자 데이터 없음, 원본 params.json도 없음).

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

전체 10초 안팎. 경로는 `common.py`가 파일 위치 기준으로 잡아서 어디서 실행해도 됨.

재현에 쓴 버전: Python 3.14.6, numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1, pillow 12.3. `data/`를 지우고 다시 돌려도 CSV 동일. 순수 numpy 산술이라 다른 버전에서도 흔들릴 자리가 거의 없음.

## 스크립트

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 합성 1000명 (항목별 독립, 정상 70 / 경계~임계치 20 / 임계치 밖 8 / 결측 2%) + 엣지 케이스 9건 | `data/synthetic_vitals.csv`, `edge_cases.csv` |
| 02 | 점수 함수 곡선. 경계 바로 밖 값의 점수 표, 재설계 k=4/6/8 곡선 | `boundary_steps.csv`, `item_curves.csv`, `k_curves.csv` |
| 03 | 채점. 원본 점수, 보정 점수, 재설계 점수, 항목 단계 두 벌, 최종 단계. 엣지 케이스 assert | `data/scored_synthetic.csv`, `edge_results.csv` |
| 04 | 단계 분포, 일치율, 희석/과판정 집계, k 민감도, 결측 정책 | `agreement.csv`, `stage_crosstab.csv`, `dilution_groups.csv`, `penalty_overjudge.csv`, `k_sensitivity.csv` |
| 05 | 그림 5장. results CSV만 읽음 | `outputs/figures/site/fig1~5.webp` |

점수 함수와 단계 규칙은 전부 `scoring.py`. 파라미터는 `common.py`의 PARAMS.

## 파라미터

원본 params.json은 손에 없어서 설계 문서 표를 기본으로 두고 테스트 출력에서 역산되는 두 가지만 반영함. pulse 정상 상한은 표의 120이 아니라 100 (115가 주의로 찍혔음), spo2 가중치는 전체의 20% (spo2만 0점인 사용자의 종합이 80.00). 나머지 가중치(체온 1.5, 맥박 1.5, spo2 2, 스트레스 1, 수축기 2, 이완기 2)는 임의. 재설계 k는 6으로 통일했고 원본의 항목별 k(3~6)는 원 단위용이라 그대로 못 씀.

## 확인할 숫자

- 02: 체온 37.51℃의 원본 항목 점수 48.75, 38.5℃에서 0.67. 재설계는 각각 99.88, 67.78
- 04: 6항목 완비 882명 기준 점수 단계와 항목 단계 일치율. 원본 0.353, 보정 후 0.831, 재설계 0.190, 재설계 + 캡 0.999
- 04: 주의 항목만 있는 420명 중 원본 점수 50~66.7이 123명, 이들은 ×0.75만으로 경고
- 04: 경고 항목 하나에 나머지 정상인 85명은 재설계 점수 최저 80.00, 전원 점수 단계 안정

재설계가 원본보다 일치율이 낮은 건 버그가 아님. 점수가 관대해질수록 항목 단계와 멀어지고, 그래서 캡이 필요하다는 게 글의 요지.

## 알려진 문제

- 합성 분포가 항목별 독립이라 6항목 모두 정상인 사람이 12%뿐. 실제 사용자 분포와 다르고, 일치율 절대값은 이 분포에 묶여 있음
- 반올림 때문에 구간 경계 바로 옆에서 뽑힌 값이 옆 구간으로 넘어가는 행이 있음. 상태는 03이 값 기준으로 다시 판정하므로 결과에는 문제 없음
