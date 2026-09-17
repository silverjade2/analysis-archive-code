# ota-reboot-spike-iot

[사용량이 튀는 날은 정말 많이 쓴 날인가: IoT 가전 fleet의 OTA 재부팅 스파이크](https://analysis-archive.vercel.app/analyses/ota-reboot-spike-iot) 재현 코드.

원본은 실제 fleet의 일별 KPI 스파이크 원인 분석. 여기서는 기기 2,000대의 세션 로그를 가상으로 만들어 OTA 배포일 4개를 심고, 당시 쓴 판정 규칙(첫 시작 시각 집중 비율 0.5)이 그 4일을 잡는지 채점한다 (실제 데이터 없음).

## 실행

```bash
pip install -r requirements.txt
for s in scripts/0*.py; do python "$s"; done
```

1분 안쪽. 경로는 `scripts/common.py`에서 파일 위치 기준으로 잡아 어디서 실행해도 됨.

재현에 쓴 버전: numpy 2.4.4, pandas 3.0.2, matplotlib 3.10.8. Python 3.12.3(Linux)과 3.14.6(macOS) 둘 다 `data/`를 지우고 다시 돌려도 결과 CSV가 byte 단위로 같았음. png는 폰트 메타데이터가 달라 diff 대상이 아님.

## 스크립트

| | 역할 | 주요 출력 |
|---|---|---|
| 01 | 기기, 세션 로그 생성. 올타임 25%, 스케줄 5%, 설치 배치 2회, OTA 4회를 심음 | `data/devices.csv`, `sessions.csv`, `ota_days.csv` |
| 02 | 일별 집계 두 벌. 시작일 귀속과 자정 분할 | `daily_startdate.csv`, `daily_midnight.csv` |
| 03 | MAD 기준 z-score로 스파이크 검출, 신규/재활성/기존 분해, 요일과 시작 시각 프로파일 | `spike_days.csv`, `spike_decomposition.csv`, `hypothesis_table.csv` |
| 04 | 정답 대비 채점. 임계 민감도, carry-over 비교, OTA 직후 세션의 경로 비중 | `threshold_sensitivity.csv`, `path_share.csv`, `summary.csv` |
| 05 | 그림 | `outputs/figures/fig1~5` |

03의 판정 기준(MAD 기준 z 3.0, 신규 3일, 재활성 7일 간격, 집중 비율 0.5)과 04의 15분 창은 운영 분석에서 쓴 값을 그대로 옮긴 것. 03은 `is_ota`, `truth_*`를 읽지 않고 읽는 건 채점 04와 그림 05(OTA일 점 찍기, 경로별 히스토그램)뿐.

## 확인할 숫자

- 02: 평소 off_n 중앙값 405 (시작일 귀속), 634 (자정 분할). 자정 분할이 높은 건 올타임 기기 carry-over
- 03: 스파이크 4일, 전부 14시 최빈, 집중 비율 0.493~0.516. 평일 최대 0.208
- 04: 임계 0.5에서 3/4 검출, 0.4에서 4/4, 오탐 0. 0.5가 경계선이라는 게 글의 논지 중 하나
- 04: OTA 직후 15분(운영 분석과 같은 창) 세션 중 재개 58~64%, 오류 가동 35~41%. 생성기에 넣은 값이지 실제가 아님

## 알려진 문제

- 올타임 기기의 이동 기능 사용은 하루 확률 0.3 고정이라 실제보다 단순. 집중 비율 계산에는 영향 없음
- 첫날(06-02) 자정 분할 off_n이 워밍업 10일치에 의존. 시작일 귀속에는 영향 없음
