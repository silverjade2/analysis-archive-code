# loyal-user-prediction-jobplatform

글: [핵심 유저 전환 예측: 저니맵이 가리킨 병목을 모델로 검증하기](https://analysis-archive.vercel.app/analyses/loyal-user-prediction-jobplatform)

구직 플랫폼 유저 45만 명(저니맵)과 모델링 대상 38,355명을 생성해, snapshot feature(v1)와 타깃 결정 직전에 자른 feature(v2)로 핵심 유저 전환 예측을 나란히 채점한다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py scripts/10_site_figures_2.py; do .venv/bin/python "$s"; done
```

- Apple Silicon 기준 01\~06 약 2분, 07(절단 시점 sweep) 약 2분 30초, 09 약 15초, 08·10 수 초.
- 환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4, matplotlib 3.11. macOS에서는 `brew install libomp`. 08·10의 한글 폰트는 Pretendard이고, 없으면 AppleGothic으로 대체된다.
- seed 고정(`SEED=42`). `data/`를 지우고 다시 돌려도 데이터와 `outputs/results/*.csv`가 byte 단위로 같다(`model_comparison.csv`의 학습 시간 열만 예외).
- 제대로 돌았는지 확인할 숫자: 01의 동의율 0.8056(30,898 / 38,355). 05의 v1_asis LightGBM 10-fold AUC 0.944, v2_pref 0.687. 06의 oracle AUC 0.716, v1 OOF 0.944, v2 OOF 0.687, 두 리스트 겹침 28.1%.

## Pipeline

| 스크립트 | 역할 | 출력 |
| --- | --- | --- |
| `01_generate_users.py` | 전체 가입 유저 45만 명(저니맵용 집계표) + 모델링 대상 38,355명과 이벤트 생성. 구조 3종 주입 | `data/population.csv`, `users.csv`, `logins.npz`, `events_apply.csv`, `events_test.csv`, `events_notice.csv` |
| `02_journey_map.py` | 상태 퍼널, 선호 정보 미작성률, 휴면·시즌 로그인 | `results/journey_*.csv`, `figures/fig1~3` |
| `03_features_v1_snapshot.py` | 조회 시점 snapshot feature(원본 노트북 구조) | `data/features_v1_snapshot.csv` |
| `04_features_v2_timecut.py` | 타깃 결정 직전 시점으로 자른 feature | `data/features_v2_timecut.csv` |
| `05_compare_models.py` | 95/5 분할 → 10-fold CV, LR·RF·XGBoost·LightGBM, v1_asis / v1_pref / v2_pref | `results/model_comparison.csv`, `holdout_lightgbm.csv`, `importance_*.csv`, `figures/fig4~5` |
| `06_nudge_list_compare.py` | 두 버전의 넛지 리스트(비동의 유저 상위 10%)를 심어둔 동의 성향으로 채점 | `results/oof_auc_vs_oracle.csv`, `nudge_list_comparison.csv`, `nudge_list_overlap.csv`, `figures/fig6` |
| `07_cutoff_sweep.py` | 절단 시점을 동의일 기준 −30일\~+365일·snapshot으로 옮기며 AUC와 상위 feature 기록. 글의 인터랙티브 위젯 데이터 | `results/cutoff_sweep.csv`, `cutoff_sweep.json`(사이트 `src/data/cutoff-sweep.json`으로 복사, 위젯 CutoffExplorer) |
| `08_site_figures.py` | 02·05·06의 그림 1\~6을 사이트 톤으로 작도. fig4(AUC 덤벨 + oracle 기준선)·fig5(중요도 순위 범프)는 여기서 재설계 | `outputs/figures/site/fig1~6.webp` |
| `09_leakage_anatomy.py` | 그림 7\~11의 데이터: 동의 전후 일별 로그인 확률, 유저 12명 타임라인, v1/v2 OOF 점수와 ROC, 비동의 유저의 점수 분위·사분면. 06의 수치와 일치하는지 assert | `results/login_around_consent.csv`, `timeline_sample.csv`, `oof_scores.csv`, `roc_curves.csv`, `score_deciles.csv`, `score_quadrants.csv`, `leakage_anatomy_summary.csv` |
| `10_site_figures_2.py` | 09의 CSV만 읽어 그림 7\~11 작도 | `outputs/figures/site/fig7~11.webp` |

`common.py`: 경로, 상수(스냅샷일 2023-06-23, 공채 시즌 4개, 원본 노트북의 feature 목록 32개), `features.py`: v1·v2가 공유하는 feature 계산(`build_features(cutoff)`는 유저별 절단일 이하의 이벤트만 집계한다), `sitestyle.py`: 08·10이 공유하는 사이트 톤.

## 데이터 스키마

| 파일 | 행 | 내용 |
| --- | --- | --- |
| `population.csv` | 76,384 | 전체 가입 유저 451,314명의 집계표. `join_date`, `season_joiner`, `status`(join_only / test_only / profile_only / test_and_profile), `last_login_date` 조합별 인원 `n`. 저니맵(02)에서만 쓴다 |
| `users.csv` | 38,355 | 모델링 대상. 정적 속성(`gender`, `age`, `career_year`, `career_type`, `extra`, `final_edu_level`, `acca_grade`, `acca_t_score`, `mental_health_grade`, `marketing_consent_yn`), 선호 정보(`pref_salary_default_yn`, `pref_welfare_cnt`), 타임라인(`join_day`, `test1_day`, `profile_day`, `consent_day`. 미동의는 −1), 타깃 `matching_use_yn`, **`truth_*`** 4열 |
| `logins.npz` | 38,355 × 758 | 유저 × 일 로그인 행렬(uint8). 2021-05-27부터 스냅샷일까지 |
| `events_apply.csv` | 77,179 | 지원 이벤트. `user`, `day`, `midas`(특정 채널 여부), `midas_kind`, 퍼널 플래그 `try`·`apply`·`test`·`complete`, `company_id` |
| `events_test.csv`, `events_notice.csv` | 68,058 / 25,180 | 역량 진단 검사 응시, 알림 응답. `user`, `day`. 같은 유저·같은 날 행이 여러 개면 그날 여러 건이다 |
| `features_v1_snapshot.csv`, `features_v2_timecut.csv` | 38,355 × 37 | 원본 노트북의 feature 32개 + 선호 정보 2개 + 타깃 + `cutoff_day`. v1은 모두 스냅샷일, v2는 유저별 절단일 |

`truth_p_consent`(동의 확률), `truth_commit`(잠재 성실도), `truth_season_joiner`, `truth_pref_complete`는 생성기의 잠재 변수다. 채점에만 쓰고 feature로는 쓰지 않는다. `days_since_last_login`은 0 이상으로 clip한다(동의일 = 가입일인 유저는 절단일이 가입일 전날이라 음수가 나올 수 있다).

## 심어둔 구조

| 구조 | 생성기 | 확인 |
| --- | --- | --- |
| 1. 동의 → 로그인. 동의한 유저는 검사 결과 확인·지원 관리로 다시 들어오므로 로그인이 타깃 결정 이후에 오른다 | 01의 `after` 마스크(강도 2.2배), 결과표 확인 burst | `leakage_anatomy_summary.csv`(동의 다음 날 로그인 확률 0.775 vs 비동의 0.103), `cutoff_sweep.csv`(절단일이 동의일을 넘는 순간 AUC 0.69 → 0.88) |
| 2. 시즌 가입자. 공채 시즌 직전 가입자는 한 번 지원하고 떠난다. 가입 월이 그 동기의 proxy | `motive_pop`, `leave_day` | `journey_dormancy.csv`, fig3의 시즌 로그인 |
| 3. 선호 정보 완성이 진짜 원인. 연봉을 기본값으로 두지 않고 복지 항목을 5개 이상 고른 유저의 동의 확률이 높다 | `pref_complete`, 동의 logit의 계수 1.30 | `journey_preference_by_consent.csv`, `importance_v2_pref.csv`(v2에서 선호 정보가 상위로 올라옴) |

동의율은 절편을 풀어 81%로 맞춘다(원본 기록). oracle은 `truth_p_consent`로 순위를 매긴 AUC이고, v1(0.944)이 그것(0.716)을 넘는 것이 이 폴더가 보여주려는 leakage다.

## 결과 파일

| 파일 | 내용 | 글에서 |
| --- | --- | --- |
| `journey_funnel.csv`, `journey_preference_fields.csv`, `journey_preference_by_consent.csv`, `journey_dormancy.csv` | 상태 퍼널, 선호 정보 미작성률, 동의 여부별 선호 정보, 휴면 비율 | 저니맵 절, fig1\~3 |
| `model_comparison.csv`, `holdout_lightgbm.csv` | 10-fold CV 지표(모델 4 × feature 구성 3), 5% holdout | 모델 비교 표, fig4 |
| `importance_v1_asis.csv`, `importance_v1_pref.csv`, `importance_v2_pref.csv` | LightGBM gain 중요도(one-hot을 원 feature로 합산) | fig5 |
| `oof_auc_vs_oracle.csv` | oracle, v1, v2의 OOF AUC | "v1 0.944 > oracle 0.716" |
| `nudge_list_comparison.csv`, `nudge_list_overlap.csv` | 두 리스트의 구성과 심어둔 동의 성향, 겹침, 무작위 baseline | 넛지 리스트 표, fig6 |
| `cutoff_sweep.csv`, `cutoff_sweep.json` | 절단 시점별 AUC·상위 feature | 위젯 CutoffExplorer |
| `login_around_consent.csv`, `timeline_sample.csv`, `oof_scores.csv`, `roc_curves.csv`, `score_deciles.csv`, `score_quadrants.csv`, `leakage_anatomy_summary.csv` | 그림 7\~11의 데이터. 캡션의 숫자는 `leakage_anatomy_summary.csv`에 모여 있다 | fig7\~11 |

## 원본과 다른 점

원본 노트북의 PyCaret 호출은 scikit-learn의 StratifiedKFold·cross_validate로 옮겼다. 분할(95/5, `random_state=786`)과 fold 수(10), 비교 모델 4종은 원본과 같다. 원본 40개 컬럼 중 모델에 쓰인 32개 feature + 타깃을 그대로 쓰고, 저니맵에서 확인했지만 당시 모델에는 넣지 않았던 선호 정보 2개(`pref_salary_default_yn`, `pref_welfare_cnt`)를 더했다.
