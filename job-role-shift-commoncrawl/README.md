# job-role-shift-commoncrawl

글: [데이터 직무 공고는 줄었는데 왜 DS는 늘었나](https://analysis-archive.vercel.app/analyses/job-role-shift-commoncrawl)

가상데이터가 아니다. 공고 원문은 저장소에 없고 집계만 둔다

원문이 필요한 단계는 코드만 있다. 저장소만으로 도는 것은 앱과 집계에서 만드는 표

## 구조

| 경로 | 역할 | 입력 |
| --- | --- | --- |
| `scripts/01_wdc_parse.py` | WDC N-Quads를 공고 parquet로 (PySpark) | `work/wdc/` |
| `scripts/02_build_wdc_dedup.py` | 제목 규칙으로 직무군, 중복과 오래된 공고 제거, 기준 데이터 | `work/wdc_jobs/` |
| `scripts/03_role_share.py` | 미국 공고 감소 분해, JOLTS, Indeed 대조 | `work/wdc_jobs/`, `data/external/` |
| `scripts/04_skill_rates.py` | 기술 96개의 회차 x 직무군 등장률 | 기준 데이터 |
| `scripts/05_label_sample.py` | 라벨 표본, 평가 표본 추출 | 기준 데이터, 먼저 붙인 라벨 713건 |
| `scripts/06_merge_labels.py` | 라벨 조각을 합침 (`train`, `test`) | `work/task1/labels_*/out/` |
| `scripts/07_prepare_train.py` | 학습용 표와 교차 검증 group | 기준 데이터, 라벨 |
| `scripts/08_train.py` | 교차 검증 (`cv`), 최종 모델 (`final`) | 학습용 표 |
| `scripts/09_apply_spark.py` | 모델을 기준 데이터 143,762건에 적용 | 기준 데이터, 모델 |
| `scripts/10_eval_test.py` | 평가 표본으로 한 번만 평가, 보정. `data/eval_labels.csv`, `task_pred.parquet`도 씀 | 예측, 평가 라벨과 본문 |
| `scripts/11_share_final.py` | 최종 표: 유형 + 묶음, 전체, 영어, 비영어 | `data/eval_labels.csv`, `task_pred.parquet` |
| `scripts/12_recheck.py` | 주 대상 기준 F1, 회차 차이 구간 | 예측, 평가 라벨 |
| `scripts/13_article_numbers.py` | 글의 업무 구성, 기술 추세 민감도, 국내 비교, 직함 변경 상한 표 | `outputs/results/`, 기준 데이터, 라벨 |
| `scripts/14_dbt_seed.py` | 최종 표에서 dbt seed `task_share_wdc.csv` | `outputs/results/` |
| `scripts/15_widget_json.py` | 사이트 위젯 json 3개 | `outputs/results/` |
| `dbt/` | seed 3개, 모델 12개, 테스트 28개. mart는 `skill_demand_vs_activity`, `task_mix`, `task_group_mix` | Snowflake 적재 표 |
| `snowflake/` | 공고, 활동량, 라벨을 stage에 올리고 COPY INTO | 원문 파일은 저장소에 없음 |
| `app/app.py` | Streamlit 앱. `data/`의 csv만 읽음 | `data/` |

`common.py`(경로, 업무 유형, 공용 함수), `roles.py`(직무군 규칙), `candidates.py`(기술 후보 96개), `model_core.py`(feature와 점수), `spark_local.py`는 스크립트가 함께 쓴다.

## 실행

이 저장소만으로 도는 것:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/11_share_final.py
.venv/bin/python scripts/13_article_numbers.py task_mix label_mix rename_bound
.venv/bin/python scripts/14_dbt_seed.py
.venv/bin/python scripts/15_widget_json.py
```

수 초. 결과는 커밋된 파일과 byte까지 같다.

앱은 진입 파일 옆의 `app/requirements.txt`로 따로 깐다. Community Cloud는 이 파일을 먼저 찾는다.

```bash
python3.14 -m venv app/.venv
app/.venv/bin/pip install -r app/requirements.txt
app/.venv/bin/streamlit run app/app.py
```

Community Cloud 진입 파일은 저장소 루트 기준 `job-role-shift-commoncrawl/app/app.py`. 경로가 `__file__` 기준이라 작업 폴더가 저장소 루트여도 된다.

전체 pipeline은 공고 원문이 든 중간 파일을 `work/`(환경변수 `JOB_ROLE_WORK`로 변경)에 쓴다. 커밋하지 않는다. Spark는 Java 17이 필요해 `JAVA_HOME`을 먼저 잡는다. 코어 수는 `SPARK_CORES`.

```bash
.venv/bin/python scripts/01_wdc_parse.py --crawl 2022-12   # 2023-12, 2024-12도
.venv/bin/python scripts/02_build_wdc_dedup.py
.venv/bin/python scripts/03_role_share.py
.venv/bin/python scripts/04_skill_rates.py
.venv/bin/python scripts/05_label_sample.py
.venv/bin/python scripts/06_merge_labels.py train
.venv/bin/python scripts/06_merge_labels.py test
.venv/bin/python scripts/07_prepare_train.py
.venv/bin/python scripts/08_train.py cv
.venv/bin/python scripts/08_train.py final
.venv/bin/python scripts/09_apply_spark.py
.venv/bin/python scripts/10_eval_test.py
.venv/bin/python scripts/11_share_final.py
.venv/bin/python scripts/12_recheck.py
.venv/bin/python scripts/13_article_numbers.py
```

`05`와 `06` 사이에 표본 라벨을 `work/task1/labels_*/out/`에 jsonl로 둔다 (Claude Sonnet, 지시문은 저장소에 없음)

`05`의 입력 `label_pool.parquet`, `labels_agent.parquet`는 국내 공고 원문을 포함해 저장소에 없어 `05`, `07`, `13`의 국내 표는 저장소만으로 다시 만들 수 없다

dbt는 Snowsight에서 `snowflake/01`, `02`, `03`을 실행한 뒤 Workspace에서 build:

```sql
EXECUTE DBT PROJECT FROM WORKSPACE user$.public."analysis-archive-code" PROJECT_ROOT = 'job-role-shift-commoncrawl/dbt' ARGS = 'build --target dev'
```

pipeline: Python 3.12.14, PySpark 4.2.0, DuckDB 1.5.6, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.4, pyarrow 26.0.0.

앱: Python 3.12.14, 3.14.7, streamlit 1.66.1, altair 6.3.0, pandas 3.1.1, numpy 2.5.4.

## 데이터

저장소에 있는 것:

| 파일 | 내용 |
| --- | --- |
| `data/SKILL_DEMAND_VS_ACTIVITY.csv`, `INT_SKILL_DEMAND.csv`, `STG_PYPI_WEEKLY.csv`, `STG_GITHUB_WEEKLY.csv`, `TASK_MIX.csv`, `TASK_GROUP_MIX.csv` | Snowflake 안의 dbt로 만든 표를 내려받은 것. 앱이 읽음 |
| `data/task_taxonomy.csv` | 업무 유형 14개와 묶음 5개 |
| `data/eval_labels.csv` | 평가 표본 510건의 라벨과 직무군, 회차. 제목과 본문은 없다. `doc_id`는 페이지 URL + 노드의 해시, `label_row`는 라벨 파일 순서 |
| `data/task_pred.parquet` | 적용 대상 143,762건의 모델 예측. 페이지 URL과 확률은 뺐다 |
| `data/base_skills.csv`, `new_skills.json` | 기존 기술 37개의 패턴, 새 후보 59개의 별칭 |
| `data/external/` | Indeed Hiring Lab `job_postings_tracker`, BLS JOLTS 구인과 채용. 2026-11-05 조회 |
| `outputs/results/task1_*.csv` | 스크립트가 쓰는 집계 |
| `outputs/results/*.json` | 위젯 json |

저장소에 없는 것:

| 파일 | 크기 | 만드는 곳 |
| --- | --- | --- |
| `work/wdc/<회차>/all/part_<i>.gz` | 20.6GB | [Web Data Commons](https://webdatacommons.org/structureddata/)의 schema.org JobPosting. `https://data.dws.informatik.uni-mannheim.de/structureddata/<회차>/quads/classspecific/JobPosting/part_<i>.gz`, 2022-12는 `part_0`\~`1`, 2023-12는 `part_0`\~`15`, 2024-12는 `part_0`\~`13` |
| `work/wdc_jobs/crawl=<회차>/` | 21GB | `01` |
| `work/task1/wdc_jobs_dedup.parquet` | 226MB | `02` |
| `work/task1/c_skill_hits.parquet` | 7MB | `04`. 공고별 페이지 URL 포함 |
| `work/task1/labels_*/` | 9MB | `05`, `06`, `07`. 공고 제목과 본문 포함 |
| `work/task1/model_b/`, `task_pred_b/` | 28MB | `08`, `09`. 모델 어휘는 공고 본문의 단어, 예측은 페이지 URL 포함 |
| 국내 공고 parquet, `work/task1/label_pool.parquet`, `labels_agent.parquet` | 3MB | 직접 운영하는 채용 공고 수집기. 공고 원문 |

글로벌 공고는 Web Data Commons의 JobPosting 2022-12, 2023-12, 2024-12 회차다.

## 알려진 문제

- 업무 유형은 회차 추세를 그리지 않는다. 회차 차이의 구간 폭이 평균 27%p
- 앱의 `(참고용)`과 회색 칸은 분류 F1이 낮은 묶음(데이터 엔지니어링, AI 엔지니어링, DS 밖의 모델링)
- 앱은 pandas `round`, 위젯은 `toFixed`로 표시해서 끝자리가 5인 값은 0.1 다를 수 있다
- `data/TASK_MIX.csv`, `TASK_GROUP_MIX.csv`와 국내 비교 표에는 DA, BA/BI, DS만 있다. 국내 DE와 그 밖은 라벨이 1건, 4건이라 공고 한 건의 라벨이 드러나 뺐고, dbt mart를 다시 만들면 이 행이 나온다
- `03`은 `any_value`를 써서 실행마다 직무군 건수가 몇 건 달라진다 (BA/BI 2,831, 2,836). 커밋된 `task1_role_share.csv`가 글의 값이고 `13`의 직함 변경 상한도 이 파일을 읽는다
- `01`\~`03`은 21GB를 읽어 이 폴더로 옮긴 뒤에는 다시 돌리지 않았다. `08 cv`도 다시 돌리지 않았다
- `work/`를 채운 상태에서 `04`, `05`, `07`, `08 final`, `10`\~`13`을 다시 돌려 결과 csv와 모델 파일이 커밋된 값과 byte까지 같은 것을 확인했다. `09`는 2022-12 회차 57,682건이 같다
- 소요 시간: `04` 86초, `05` 40초, `07` 14초, `08 final` 12초, `09` 2022-12 11초, `10` `11` 수 초, `12` 68초, `13` 80초 (`SPARK_CORES=4`)
- 글에 적은 SageMaker, Databricks 확인에 쓴 묶음 스크립트는 저장소에 없다

## 확인할 숫자

- 사전 기술 60, 활동량 측정 32, 공고 그룹 537
- 국내 공고 248 (DA 145, BA/BI 23, DS 80), 글로벌 공고 68,720 (DA 28,668, BA/BI 24,319, DS 15,733)
- 기준 데이터 159,431건, 적용 대상 143,762건, 평가 표본 510건
- `TASK_MIX` 210행, `TASK_GROUP_MIX` 75행, seed `task_share_wdc` 456행
