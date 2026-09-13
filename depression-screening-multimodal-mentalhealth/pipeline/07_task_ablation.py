"""07. 과제별 신호. 참가자 단위 5-fold에서 과제 하나씩만 써서 텍스트/음성/융합 AUC.
그리고 참가자 단위 집계를 (전체 8과제 / 서술 5과제 / 고정 텍스트 3과제)로 나눠 본다.
숫자 세기·낭독은 텍스트가 모두 같으므로 텍스트 모델은 우연 수준이어야 한다."""

import _path  # noqa: F401
import numpy as np
import pandas as pd
from config import FIXED_TEXT_TASKS, NARRATIVE_TASKS, RESULTS, SEED, TASK_KO, TASKS
from evalutil import cross_val_predict, fp_audio, fp_fusion, fp_text, load_recordings, metrics, participant_level

df = load_recordings()
rows = []
for task in TASKS:
    sub = df[df.task == task].reset_index(drop=True)
    for model, fp in [("text", fp_text), ("audio", fp_audio), ("fusion", fp_fusion)]:
        if model == "text" and task in FIXED_TEXT_TASKS:
            rows.append(
                dict(task=task, task_ko=TASK_KO[task], model=model, auc=0.5, f1=np.nan, note="identical text; skipped")
            )
            continue
        oof = cross_val_predict(sub, fp, split="participant", seed=SEED)
        m = metrics(sub.label_depressed, oof)
        rows.append(dict(task=task, task_ko=TASK_KO[task], model=model, auc=m["auc"], f1=m["f1"], note=""))
        print(task, model, round(m["auc"], 4))
res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS / "task_ablation.csv", index=False)

# 참가자 단위 집계 범위 비교 (융합 모델, 참가자 분할)
oof_all = cross_val_predict(df, fp_fusion, split="participant", seed=SEED)
agg_rows = []
for name, tasks in [
    ("all_8_tasks", TASKS),
    ("narrative_5", NARRATIVE_TASKS),
    ("fixed_text_3", FIXED_TEXT_TASKS),
    ("memory_2", ["happy_memory", "hard_memory"]),
]:
    m = df.task.isin(tasks).values
    par = participant_level(df[m], oof_all[m])
    agg_rows.append(dict(aggregation=name, n_tasks=len(tasks), **{f"par_{k}": v for k, v in par.items()}))
agg = pd.DataFrame(agg_rows).round(4)
agg.to_csv(RESULTS / "task_aggregation.csv", index=False)
print(agg.to_string(index=False))
