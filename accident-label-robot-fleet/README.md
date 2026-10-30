# accident-label-robot-fleet

[기기 사고 사전 탐지 2편: 넘어짐 알림 중 무엇이 사고인가](https://analysis-archive.vercel.app/analyses/accident-label-robot-fleet) 재현 코드.

1편에서 사고로 센 전도, 낙하, 들림 알림을 알림 앞뒤의 가동 기록과 주행 기록으로 네 유형(A 실제 넘어짐, B 주행 실패 뒤 사용자 개입, C 관리 조작, D 판정 불가)으로 가르고, A, B 사건의 재발 위험 곡선으로 재발을 접는 기간 K를 다시 정한 분석의 재현. 실제 로그는 없고 구조(유형 구성, 시각대별 평소 멈춤, 기기마다 다른 취약성, 상담이 붙는 방식)만 옮김.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

1분 안쪽. Python 3.14, numpy 2.5.4, pandas 3.1.0, matplotlib 3.11.2. 같은 환경에서 두 번 돌려 생성 데이터와 결과 CSV가 byte 단위로 같았음.

## 스크립트

| | 역할 | 주요 출력 |
|---|---|---|
| 01 | 기기 3,000대, 10개월. 가동 세션, 알림과 주행 기록, 상담 기록 생성 | `data/devices.csv`, `sessions.csv.gz`, `events.csv.gz`, `consults.csv` |
| 02 | 알림을 사건으로 묶고 유형 A~D 판정, 창 민감도, 코드 분리 전후 | `incidents.csv`, `type_share.csv`, `type_sensitivity.csv`, `type_by_period.csv` |
| 03 | 달 x 시각별 평소 멈춤, 달별 A 신호 배수와 기대를 넘는 몫 | `stop_baseline.csv`, `a_signal_monthly.csv` |
| 04 | A, B 재발 위험 곡선, 평소 발생률과 장기 수준, K 후보, K별 기준 시점, 코드별 30일 안 재발 | `recurrence_curve.csv`, `recurrence_summary.csv`, `k_anchors.csv`, `funnel.csv`, `repeat_by_code.csv` |
| 05 | 상담 에러코드 일치율, 유형별 상담 연결률과 A / C | `consult_match.csv`, `consult_link.csv`, `consult_a_over_c.csv` |
| 06 | 알림 사건 비율을 달력일, 가동일, 가동 시간으로 | `exposure_monthly.csv`, `exposure_change.csv` |
| 07 | ground truth 평가 | `eval_*.csv` |
| 08 | 그림 6장 | `outputs/figures/site/fig1~6` |

`truth_*` 열은 07만 읽음. 02~06은 시각, 코드, 세션만 봄.

판단 기준은 결과 전에 정해 `common.py`와 각 스크립트 상수에 둠: A 신호는 달별 배수 2배, 창 민감도 10%p, K는 7일 평균 재발률이 평소 발생률 아래로 처음 내려오는 날(하한 7일, 안 내려오면 경과 31~60일 수준 기준), 상담 연결률 A / C 1.5.
