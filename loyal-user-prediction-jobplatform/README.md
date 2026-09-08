# loyal-user-prediction-jobplatform

가상데이터로 재현한 "핵심 유저 전환 예측" 프로젝트. 글은 [analysis-archive](https://analysis-archive.vercel.app/analyses/loyal-user-prediction-jobplatform)에 있다.

원본 프로젝트(2023.07)는 구직 플랫폼 유저 45만 명을 저니맵으로 나누고, 프로필·검사를 마친 유저 중 누가 기업 추천에 동의하는지를 분류 모델로 예측했다. 이 저장소는 그 파이프라인을 같은 스키마의 가상데이터로 다시 돌리되, 당시 모델의 구조적 문제(타깃 결정 이후의 행동이 피처에 포함됨)를 데이터에 그대로 심어 놓고, 시간 기준을 자른 두 번째 버전과 비교한다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in notebooks/0*.py; do .venv/bin/python "$s"; done
```

Apple Silicon 기준 01~06 약 2분, 07(절단 시점 스윕) 약 2분 30초, 09 약 15초, 08·10 수 초. 시드 고정(`SEED=42`) — `data/`를 지우고 다시 돌려도 데이터와 `outputs/results/*.csv`가 바이트 단위로 같다(`model_comparison.csv`의 학습 시간 열만 예외).

환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4, matplotlib 3.11. macOS에서는 LightGBM·XGBoost용 `libomp`가 필요하다(`brew install libomp`). 08의 한글 폰트는 Pretendard이고, 없으면 AppleGothic으로 대체된다.

## 파이프라인

| 스크립트 | 역할 |
| --- | --- |
| `01_generate_users.py` | 전체 45만 명(저니맵용) + 모델링 대상 38,355명 생성. 전환→로그인, 시즌 가입 코호트, 선호 정보 완성이라는 구조 3종 주입 |
| `02_journey_map.py` | 상태 퍼널, 선호 정보 미작성률, 휴면·시즌 로그인 재현 |
| `03_features_v1_snapshot.py` | 조회 시점 스냅샷 피처 (원본 노트북 구조) |
| `04_features_v2_timecut.py` | 타깃 결정 직전 시점으로 자른 피처 |
| `05_compare_models.py` | 95/5 분할 → 10-fold CV, LR·RF·XGBoost·LightGBM 비교, v1/v2 |
| `06_nudge_list_compare.py` | 두 버전의 넛지 리스트(비동의 유저 상위 10%)를 심어둔 정답으로 채점 |
| `07_cutoff_sweep.py` | 절단 시점을 동의일 기준 −30일~+365일·스냅샷으로 옮기며 AUC와 상위 피처 기록 — 글의 인터랙티브 위젯 데이터(`cutoff_sweep.json`) |
| `08_site_figures.py` | 02·05·06의 그림 1~6을 사이트 톤(Pretendard, webp)으로 작도. 데이터·축은 동일. fig4(AUC 덤벨 + 정답 AUC 기준선)와 fig5(중요도 순위 범프 차트)는 여기서 재설계 |
| `09_leakage_anatomy.py` | 그림 7~11의 데이터: 동의(또는 v2 절단 다음 날) 전후 일별 로그인 확률, 유저 12명 타임라인 표본, v1/v2 OOF 점수와 ROC, 비동의 유저의 점수 분위·v1×v2 사분면. 06의 AUC·넛지 리스트·겹침과 일치하는지 assert |
| `10_site_figures_2.py` | 09의 CSV만 읽어 그림 7~11 작도 (동의 전후 로그인, 점수 평면 hexbin, 타임라인, ROC, 점수 분위) |

결과는 `outputs/results/*.csv`, 그림은 `outputs/figures/*.png`(원본)와 `outputs/figures/site/*.webp`(사이트용). `notebooks/sitestyle.py`는 08·10이 공유하는 사이트 톤(폰트·팔레트·피처 한글명·저장)이다. 글의 캡션에 쓰인 숫자는 `leakage_anatomy_summary.csv`에 모여 있다.

## 스키마

원본 노트북의 40개 컬럼 중 모델에 쓰인 32개 피처 + 타깃(`matching_use_yn`)을 그대로 쓰고, 저니맵 단계에서 확인했지만 당시 모델에는 넣지 않았던 선호 정보 2개(`pref_salary_default_yn`, `pref_welfare_cnt`)를 더했다. `truth_*` 컬럼은 생성기의 잠재 변수로, 채점에만 쓰고 피처로는 쓰지 않는다.

`data/`는 커밋한다. 코드를 돌리지 않아도 생성 데이터를 열어볼 수 있게 하기 위해서다. 생성기(`01_generate_users.py`)를 고치면 파이프라인 전체를 다시 돌려 데이터와 결과를 같은 커밋에 넣는다.
