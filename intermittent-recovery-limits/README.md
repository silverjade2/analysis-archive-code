# intermittent-recovery-limits

가상데이터로 재현한 "피처 표현의 한계: 2호에서 못 찾은 28대의 행방". 글은 [analysis-archive](https://analysis-archive.vercel.app/analyses/intermittent-recovery-limits)에 있다.

2호 글([usage-pattern-clustering](../usage-pattern-clustering))에서 어떤 k로도 회수되지 않던 intermittent 28대를 두 가설(알고리즘 교체 vs 피처 재설계)로 추적하고, 둘 다 실패한 원인을 해부한다. 데이터 생성 스크립트는 2호 폴더의 `01_generate_usage_profiles.py`와 같은 파일이고, 생성되는 `data/usage_profiles.csv`도 바이트 단위로 같다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in notebooks/0*.py; do .venv/bin/python "$s"; done
```

Apple Silicon 기준 전체 약 7초. 시드 고정(`SEED=42`).

환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, scipy 1.18, matplotlib 3.11.

## 파이프라인

| 스크립트 | 역할 |
| --- | --- |
| `00_generate_usage_profiles.py` | 가상데이터 생성 (2호와 동일한 파일, 검증 플롯은 `fig0`) |
| `01_recover_intermittent.py` | 기준선·GMM·DBSCAN·버스트 피처 4방법 비교 |
| `02_diagnose_missed.py` | 해부: blur 복원(생성기의 난수 시퀀스를 같은 시드로 재밟음), 피크 시각, 모양 지표, GMM 사후확률, 강화 피처 |

그림은 `outputs/figures/fig1~3.png`. `02_`는 `00_`을 모듈로 import해서 생성 과정의 잠재 플래그를 복원하므로, 2호 생성기를 고치면 이 폴더의 `00_`도 같은 내용으로 바꾼다.

`data/`는 커밋한다.
