# auc-illusion-imbalanced

가상데이터로 재현한 "불균형 분류에서 AUC 0.94의 착시: 지표가 만드는 안심". 글은 [analysis-archive](https://analysis-archive.vercel.app/analyses/auc-illusion-imbalanced)에 있다.

클래스 조건부 분포가 완전히 같은 데이터를 양성 비율만 3% / 10% / 30%로 바꿔 동일 LightGBM으로 학습하고, ROC-AUC·PR-AUC·calibration·precision@k가 유병률에 따라 어떻게 갈라지는지 본다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in notebooks/0*.py; do .venv/bin/python "$s"; done
```

Apple Silicon 기준 전체 약 10초. 시드 고정(`SEED=42`) — `data/`를 지우고 다시 돌려도 CSV가 바이트 단위로 같다.

환경: Python 3.14.6, pandas 3.0, scikit-learn 1.9, scipy 1.18, LightGBM 4.7, matplotlib 3.11. macOS에서는 LightGBM용 `libomp`가 필요하다(`brew install libomp`).

## 파이프라인

| 스크립트 | 역할 |
| --- | --- |
| `01_generate_imbalance_variants.py` | 조건부 분포 동일·양성 비율 3종 데이터 생성 (`data/imbalance_pos{3,10,30}.csv`) + 검증 플롯 |
| `02_metric_illusion.py` | 동일 LightGBM 학습, 지표 4종 비교표 + 플롯 5장 |

그림은 `outputs/figures/fig1~6.png`.

`data/`는 커밋한다. 생성기(`01_`)를 고치면 파이프라인 전체를 다시 돌려 데이터·결과·글의 숫자를 같은 커밋에 넣는다.
