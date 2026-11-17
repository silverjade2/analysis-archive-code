# personal-warehouse-duckdb

[개인 데이터 창고: 웨어하우스 설계를 1인 규모로 줄이면 뭐가 남나](https://analysis-archive.vercel.app/analyses/personal-warehouse-duckdb) 재현 코드.

맥 한 대의 Parquet, DuckDB 창고에 소스 셋(스코어 시트, iCalendar 캘린더, 설문 응답)을 붙인 첫날의 재현. 실제 소스는 없음. 응답 형식(Sheets values 응답에서 잘려 오는 행 끝 빈 칸, VCALENDAR 의 RRULE, EXDATE, RECURRENCE-ID, 식별 열이 있는 설문 원본과 DB 쪽 view)만 옮겨 가짜 소스를 만들고, 원래 수집기 순서대로 읽어 raw parquet 로 저장한 뒤 stg, mart SQL 을 같은 파일 순서로 돌려 생성기가 기록한 truth 와 대조한다. 4주 운영 기록(실행, 실패, 호출 수, 실행 시간, 파일 크기)은 재현 대상이 아님.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

3초 안쪽. Python 3.11.16, duckdb 1.5.6, pyarrow 26.0.0, pandas 3.0.6, numpy 2.4.6, icalendar 7.3.0, recurring-ical-events 3.8.2 (운영 코드와 같은 조합). 이 조합에서 `data/` 와 `outputs/` 를 지우고 다시 돌려도 `sql_seconds.csv` 와 `data/warehouse.duckdb` 를 뺀 26개 파일이 byte 단위로 같았음. `data/warehouse.duckdb` 는 git 에 넣지 않음.

## 스크립트

| | 역할 | 주요 출력 |
|---|---|---|
| 01 | 가짜 소스. 스코어 시트 탭 4개(라운드 14, 그중 삭제 6), 캘린더 6개의 VCALENDAR(반복 일정 3개, 취소 1, 시간대 없는 값과 UTC 값, 조회 기간 밖 2), 설문 원본 352행(테스트 12) | `data/source/`, `truth_*.csv` |
| 02 | 수집. 시트는 토큰, 탭 목록, batchGet. 캘린더는 사용자 정보, 캘린더 목록, 캘린더마다 조회. 설문은 접속, view 조회 후 메모리에서 (차원, 값) 건수로 집계. 선언 밖 컬럼이 오면 실패. 반복 판정은 두 방식을 같이 남김 | `data/raw/<table>/2026-10-20.parquet`, `calls.jsonl`, `collect.csv`, `calls.csv`, `calendar_rules.csv` |
| 03 | raw view(날짜별 파일 glob, 최신 파일 하나) -> `sql/stg` -> `sql/mart` 번호순. mart 를 CSV 로, 실행 시간은 5번 | `sql_steps.csv`, `sql_seconds.csv`, mart 5개 |
| 04 | ground truth 평가. 스코어 행이 빠진 이유와 타수 합, 캘린더 회차와 반복 판정과 주별 시간, 설문 건수를 원본에서 다시 센 값과 식별 값 | `eval_checks.csv`, `numbers.csv` |

`truth_*.csv` 는 04 만 읽음. `sql/` 은 운영 코드의 SQL 에서 골프장, 플레이어 기본 이름 상수만 가상 값으로 바꿈. 동작이 다른 곳은 `mart/030_calendar.sql` 의 미래 주 기준일 하나 (`current_date` -> 마지막 수집일, 결과가 실행 날짜와 무관하게). 수집 쪽은 첫날 한 번 성공하는 경로만 옮겼고 재시도, 제한 시간, 건너뜀, `_meta.json` 실행 기록, 같은 날 조회 기간 유지는 없음. 가짜 캘린더 서버는 기간 조건 없이 전부 돌려주고(실제 CalDAV 는 서버가 기간으로 거름) 전개 단계가 같은 기간으로 잘라 회차는 같음.

판단 기준은 `common.py` 에: 첫 수집일 2026-10-20, 캘린더 조회 기간 지난 56일과 앞으로 90일, 소스별 하루 호출 상한(캘린더 60, 시트 20, 설문 20).

## 확인할 숫자

- 02: 시트 점수 660행. 호출은 시트 3, 캘린더 8(캘린더 6개, 일정 있는 캘린더 5개), 설문 2
- 02: 캘린더 회차 73(종일 9). 반복 판정은 회차에 RECURRENCE-ID 가 있으면 반복 73, 원본 VEVENT 에 RRULE 이나 회차 수정이 있으면 반복 12
- 02: 설문 응답 340(테스트 12 제외) -> 집계 228행, 차원 8개
- 03: stg 점수 459행(삭제 라운드 6개의 192행, 빈 칸 9행 제외). mart 타수 합 2,418 = 시트 합
- 03: stg, mart SQL 7파일 0.03초 안팎(5번 중앙값, 머신마다 다름)
- 04: `eval_checks.csv` 18개 전부 truth 와 같음

## 알려진 문제

- 빈 칸 비율(1.5%), 플레이어별 타수, 일정 수, MBTI 비율은 생성기에서 정한 값. 실제 소스에서 센 값이 아님
- 설문 집계 코드는 mbti 가 비면 `letters_matched` 를 `(없음)` 대신 0 으로 셈. 빈 값을 `(없음)` 으로 바꾼 뒤 글자 수를 세는데 `(없음)` 이 4글자라서. 원본 열이 NOT NULL 이라 가상데이터에도 넣지 않았고 코드는 운영 그대로 둠
- 일정을 전부 지운 날의 snapshot(`calendar_window` 가 따로 있는 이유)은 재현하지 않음. snapshot 은 첫날 하나뿐
