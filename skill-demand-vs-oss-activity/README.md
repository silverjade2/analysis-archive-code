# skill-demand-vs-oss-activity

글: [찾는 만큼 쓰이지 않는다](https://analysis-archive.vercel.app/analyses/skill-demand-vs-oss-activity)

다른 폴더와 달리 가상데이터가 아니고, 공고 원문 없이 기술별 집계만 둔다

## 구조

| 경로 | 역할 |
| --- | --- |
| `scripts/01_fetch_activity.py` | ClickHouse 공개 서버에서 GitHub 이벤트와 PyPI 다운로드를 기술별 주 단위로 받아 `data/*.parquet`에 씀 |
| `snowflake/01_load_jobkko.sql` | 공고 parquet를 stage에 올리고 COPY INTO (공고 원문이라 파일은 저장소에 없음) |
| `snowflake/02_load_activity.sql` | `data/github_weekly.parquet`, `data/pypi_weekly.parquet`를 stage에 올리고 COPY INTO |
| `snowflake/04_dbt_git_setup.sql` | GitHub 저장소를 Snowflake Workspace로 가져오는 API integration (비공개 저장소면 토큰 secret도) |
| `dbt/` | seed `skills.csv`(기술 사전 60개), 모델 7개, 테스트 11개. mart는 `skill_demand_vs_activity` |
| `dashboard/app.py` | Streamlit in Snowflake 앱 |
| `scripts/02_widget_data.py` | 사이트 위젯 json 두 개를 `outputs/results/`에 씀 |

```bash
python scripts/01_fetch_activity.py --start 2023-01-02 --end 2026-11-02   # 공개 서버 쿼리 두 번, 수 초
python scripts/02_widget_data.py                                          # 1초 안쪽
```

`01`은 `data/`의 parquet 두 개를 덮어쓴다. 2026-11-04에 다시 받아 저장된 파일과 같은 것을 확인했다(공개 서버가 과거 값을 고치면 달라질 수 있다).

dbt는 Snowsight에서 `snowflake/01`, `02`, `04`를 실행한 뒤 Workspace에서 build:

```sql
EXECUTE DBT PROJECT FROM WORKSPACE user$.public."analysis-archive-code" PROJECT_ROOT = 'skill-demand-vs-oss-activity/dbt' ARGS = 'build --target dev'
```

## 데이터

| 파일 | 내용 |
| --- | --- |
| `data/github_weekly.parquet` | 저장소 49개 주간 GitHub 이벤트(봇 계정 제외), 2023-01-02 주\~2026-10-26 주. staging에서 자르기 전 원본 |
| `data/pypi_weekly.parquet` | 패키지 46개 주간 다운로드, 같은 기간 |
| `data/SKILL_DEMAND_VS_ACTIVITY.csv` | dbt mart를 내려받은 것. 기술 60개의 공고 그룹 등장률, 최근 13주 다운로드, 3년 배수, 사분면 |
| `data/STG_PYPI_WEEKLY.csv` | dbt staging `stg_pypi_weekly`를 내려받은 것 |

등장률은 직접 운영하는 채용 공고 수집기의 데이터 직무 공고 791건(공고 그룹 537개)에서 센 값이고, 공고 원문과 그 적재 테이블은 저장소에 없다. `02`는 위 csv 두 개와 GitHub parquet만 읽는다.

## 위젯 json

- `skill-quadrant.json`: 측정 기술 32개 사분면 값과 패키지별 주간 다운로드. 13주 합을 패키지 합으로 다시 내 mart 값과 같은지 assert
- `event-break.json`: 기술별 분기 GitHub 스타(WatchEvent)와 PyPI 다운로드, 2024년 분기 평균 = 100. 다 차지 않은 2026-Q4는 뺌
- Kubernetes는 사전과 dbt에서 strong이지만 json에서는 proxy (Python 클라이언트 패키지)

Python 3.12.14, pandas 2.3.3, pyarrow 26.0.0에서 두 번 실행해 json byte 동일.

## 확인할 숫자

- 측정 32개, strong 10, proxy 22
- 중앙값: 등장률 0.049348, 최근 13주 다운로드 171,189,083, 3년 배수 4.6575(31개)
- 저장소 49개 분기 스타 합: 2024년 48,318\~53,044, 2026-Q3 2,336
- `01` 결과 행 수: GitHub 44,353, PyPI 9,051
