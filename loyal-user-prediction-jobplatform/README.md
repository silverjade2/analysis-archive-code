# loyal-user-prediction-jobplatform

[핵심 유저 전환 예측: 저니맵이 가리킨 병목을 모델로 검증하기](https://analysis-archive.vercel.app/analyses/loyal-user-prediction-jobplatform) 재현 코드.

채용 플랫폼에서 추천 동의(핵심 유저 전환)를 예측하던 원본 노트북의 재현. 같은 구조의 가상 유저 38,355명과 이벤트를 만들어 그 절차를 그대로 돌리면 snapshot feature가 oracle AUC 0.716보다 높은 0.944를 내는데, 왜 그런지 cutoff 버전(0.687)과 대조한다. 실제 유저 데이터는 없음.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py scripts/10_site_figures_2.py; do .venv/bin/python "$s"; done
```

01~06이 2분 정도, 07의 cutoff sweep이 2분 30초, 09가 15초. 08과 10은 몇 초.

Python 3.14, numpy 2.5, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4, matplotlib 3.11, Pillow 12. `model_comparison.csv`의 학습 시간 열만 실행마다 다름. 08과 10의 폰트는 Pretendard, 없으면 AppleGothic.

## 스크립트

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 전체 가입 45만 명 (집계표) + 모델링 대상 38355명과 이벤트 생성. 구조 3종 주입 | `data/population.csv`, `users.csv`, `logins.npz`, `events_*.csv` |
| 02 | 저니맵. 상태 퍼널, 선호 정보 미작성률, 휴면과 시즌 로그인 | `journey_*.csv`, `fig1~3` |
| 03 | v1 snapshot feature (원본 노트북 구조) | `data/features_v1_snapshot.csv` |
| 04 | v2 cutoff feature (타깃 결정 직전) | `data/features_v2_timecut.csv` |
| 05 | 95/5 split 후 10-fold CV. LR, RF, XGBoost, LightGBM x v1_asis / v1_pref / v2_pref | `model_comparison.csv`, `importance_*.csv`, `fig4~5` |
| 06 | 넛지 리스트 (비동의 상위 10%) 두 버전을 넣어 둔 동의 성향으로 평가 | `oof_auc_vs_oracle.csv`, `nudge_list_*.csv`, `fig6` |
| 07 | cutoff를 동의일 기준 -30일~+365일, snapshot으로 옮기며 AUC와 상위 feature 기록 | `cutoff_sweep.csv`, `cutoff_sweep.json` (사이트 `src/data/cutoff-sweep.json`) |
| 08 | 그림 1~6 사이트 톤. fig4, fig5는 재설계 | `outputs/figures/site/fig1~6.webp` |
| 09 | 그림 7~11 데이터. 동의 전후 로그인, 타임라인 12명, OOF 점수와 ROC, 분위와 사분면. 06과 assert 대조 | `login_around_consent.csv`, `oof_scores.csv`, `leakage_anatomy_summary.csv` 등 |
| 10 | 09의 CSV만 읽어 그림 7~11 | `outputs/figures/site/fig7~11.webp` |

`common.py`에 snapshot일 2023-06-23, 공채 시즌 4개, 원본 노트북 feature 32개. `features.py`의 `build_features(cutoff)`는 유저별 cutoff일 이하 이벤트만 집계하고 v1, v2가 같이 쓴다. `sitestyle.py`는 voice, wellness 폴더에 같은 파일이 있고 depression 폴더 것은 SEQUENTIAL이 하나 더 있음. 한쪽 고치면 나머지도.

원본의 PyCaret 호출은 scikit-learn StratifiedKFold, cross_validate로 옮김. 95/5 split, random_state 786, 10-fold, 모델 4종과 각 모델의 파라미터(RF 100, XGBoost 200/깊이 6/lr 0.1, LightGBM 200/잎 31)는 원본 노트북 값 그대로. 여기에 선호 정보 2개(`pref_salary_default_yn`, `pref_welfare_cnt`)를 더했다.

## 확인할 숫자

- 01: 동의율 0.806 (30,898 / 38,355). 생성기 절편을 이 값에 맞춰 풀었으므로 여기서 벗어나면 생성기가 달라진 것
- 05: v1_asis LightGBM 10-fold AUC 0.944, v2_pref 0.687. 앞의 것이 ground truth 확률의 AUC 0.716보다 높은 게 leakage 증거, 뒤의 것이 그 아래 붙는 게 정상
- 06: oracle 0.716, v1 OOF 0.944, v2 OOF 0.687. 두 리스트 겹침 28.1%, 상위 10%를 뽑아도 열에 일곱은 다른 사람
- 06: 선호 정보 빈 유저(연봉 기본값 또는 복지 5개 미만) 7,165명 안에서 같은 745명을 뽑으면 v2 0.859, v1 0.821, 무작위 0.725 (`nudge_list_pref_empty.csv`). 비동의 전체에서 뽑는 위의 비교와 모집단만 다름

## 데이터

`population.csv` 74,822행은 전체 가입 45만 명을 `join_date`, `season_joiner`, `status`, `last_login_date` 조합별 인원 `n`으로 집계한 것. 저니맵(02)에서만 쓴다. `users.csv` 38,355행은 모델링 대상으로 정적 속성, 선호 정보 2열, 타임라인(`join_day`, `test1_day`, `profile_day`, `consent_day`. 미동의는 -1), 타깃 `matching_use_yn`, `truth_*` 4열. `logins.npz`는 유저 x 일(38,355 x 758) 로그인 행렬. 이벤트는 `events_apply.csv` 77,179행 (퍼널 플래그, 채널, `company_id`), `events_test.csv` 68,058행, `events_notice.csv` 25,180행.

`features_v1_snapshot.csv`, `features_v2_timecut.csv`는 38,355 x 37. 원본 feature 32 + 선호 정보 2 + 타깃 + `cutoff_day`. v1은 전원 snapshot일, v2는 유저별 cutoff일.

`truth_*` 4열은 `truth_p_consent`(동의 확률), `truth_commit`(잠재 성실도), `truth_season_joiner`, `truth_pref_complete`. `days_since_last_login`은 0으로 clip. 동의일이 가입일과 같으면 cutoff일이 가입일 전날이라 음수가 나와서.

## 생성기에 넣은 것

- 동의 후 로그인 증가 (01의 `after` 마스크 2.2배 + 결과표 확인 burst). 동의 다음 날 로그인 확률 0.775 vs 비동의 0.103, cutoff일이 동의일을 넘는 순간 AUC 0.69 -> 0.88 (`cutoff_sweep.csv`)
- 시즌 가입자. 공채 시즌 직전 가입자는 한 번 지원하고 떠남, 가입 월이 그 proxy (`journey_dormancy.csv`, fig3)
- 선호 정보 완성이 진짜 원인. 연봉 기본값이 아니고 복지 5개 이상이면 동의 logit +1.30. v2에서 상위로 올라옴 (`importance_v2_pref.csv`)

oracle은 `truth_p_consent`로 순위를 매긴 AUC.

## 알려진 문제

- 07의 음의 offset은 거의 의미 없음. cutoff 하한이 프로필 완성 전날이고 동의 지연 중앙값이 3일이라 -30일이든 -1일이든 대부분 같은 날로 잘림. AUC가 0.68 근처에서 안 움직이는 이유. 위젯에는 그대로
- 01의 상태 비율(가입만 80 / 검사만 9 / 프로필만 3)은 글의 분포에 맞춘 값. 모델링 대상 38,355명을 뽑는 주 rng를 건드리지 않으려고 나머지 41만 명만 별도 스트림(`SEED + 1`)으로 재배정했고, 그래서 `population.csv`만 바뀌고 `users.csv`와 결과 CSV는 그대로
