"""EDA. 그룹 vs PHQ-9, 과제별 텍스트 길이, 치료 언급률, 음향 feature 분산 분해"""

import _path  # noqa: F401
import pandas as pd
from audiogen import AUDIO_COLS
from config import DATA, NARRATIVE_TASKS, RESULTS, TASK_KO, TASKS
from evalutil import load_recordings

df = load_recordings()
coh = pd.read_csv(DATA / "participants.csv")

ct = pd.crosstab(coh.group, coh.label_depressed).rename(columns={0: "not_depressed", 1: "depressed"}).reset_index()
ct.to_csv(RESULTS / "eda_group_by_label.csv", index=False)
hist = coh.groupby(["group", "phq9"]).size().rename("n").reset_index()
hist.to_csv(RESULTS / "eda_phq9_hist.csv", index=False)
task_stats = (
    df.groupby("task")
    .agg(
        n=("recording_id", "size"),
        mean_words=("n_words", "mean"),
        n_unique_texts=("stt_text", "nunique"),
        treatment_mention_rate=("truth_treatment_mention", "mean"),
    )
    .round(3)
    .reindex(TASKS)
    .reset_index()
)
task_stats["task_ko"] = task_stats.task.map(TASK_KO)
task_stats.to_csv(RESULTS / "eda_task_stats.csv", index=False)
nar = df[df.task.isin(NARRATIVE_TASKS)]
tm = (
    nar.groupby(["group", "label_depressed"])
    .truth_treatment_mention.mean()
    .round(4)
    .rename("treatment_mention_rate")
    .reset_index()
)
tm.to_csv(RESULTS / "eda_treatment_mention.csv", index=False)


# eta^2: 화자 간 / 라벨 간 / 과제 간 분산 비율
def eta2(x, g):
    grand = x.mean()
    ss_tot = ((x - grand) ** 2).sum()
    ss_b = sum(len(v) * (v.mean() - grand) ** 2 for _, v in x.groupby(g))
    return ss_b / ss_tot


rows = []
for c in AUDIO_COLS:
    rows.append(
        dict(
            feature=c,
            eta2_speaker=eta2(df[c], df.participant_id),
            eta2_label=eta2(df[c], df.label_depressed),
            eta2_task=eta2(df[c], df.task),
        )
    )
var = pd.DataFrame(rows).round(4)
var.to_csv(RESULTS / "eda_audio_variance_decomposition.csv", index=False)
demo = pd.DataFrame(
    [
        dict(stat="age_median", value=coh.age.median()),
        dict(stat="age_20_59_share", value=((coh.age >= 20) & (coh.age <= 59)).mean().round(4)),
        dict(stat="employed_n", value=int(coh.employed.sum())),
        dict(stat="employed_depressed_share", value=coh[coh.employed == 1].label_depressed.mean().round(4)),
        dict(stat="unemployed_depressed_share", value=coh[coh.employed == 0].label_depressed.mean().round(4)),
    ]
)
demo.to_csv(RESULTS / "eda_demographics.csv", index=False)
print(ct.to_string(index=False))
print(task_stats.to_string(index=False))
print(tm.to_string(index=False))
print(var.sort_values("eta2_speaker", ascending=False).head(8).to_string(index=False))
print(demo.to_string(index=False))
