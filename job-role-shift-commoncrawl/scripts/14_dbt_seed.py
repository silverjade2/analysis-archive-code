"""최종 표에서 dbt seed task_share_wdc.csv를 만든다

전체 층, 주 대상 DA, BA/BI, DS만. DA 인접은 근거로 쓰지 않고, DE, ML/AI 엔지니어는 평가 라벨이 없어 보정값이 비어 뺀다
evidence: 참고용 묶음과 그 소속 유형은 참고용. 모델링 묶음과 그 소속 유형은 DS에서만 근거
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "outputs" / "results" / "task1_b_share_final.csv")
df = df[(df["slice"] == "전체") & df.role_family.isin(["DA", "BA/BI", "DS"])].copy()
evidence = (df.group_status == "통과") & ((df.task_group != "모델링") | (df.role_family == "DS"))
df["evidence"] = evidence.map({True: "근거", False: "참고용"})
out = df.rename(columns={"share_raw": "model_share", "share_ppi": "share"})[
    [
        "role_family",
        "crawl",
        "kind",
        "unit_type",
        "unit",
        "task_group",
        "group_status",
        "evidence",
        "n_pop",
        "n_test",
        "model_share",
        "share",
        "ci_lo",
        "ci_hi",
    ]
].sort_values(["role_family", "crawl", "kind", "unit_type", "unit"])
for c in ["model_share", "share", "ci_lo", "ci_hi"]:
    out[c] = out[c].round(4)
out.to_csv(ROOT / "dbt/seeds/task_share_wdc.csv", index=False)
print(len(out), "행", out.evidence.value_counts().to_dict())
