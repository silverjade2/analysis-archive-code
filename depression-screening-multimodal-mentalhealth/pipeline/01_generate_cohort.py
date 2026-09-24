"""참가자 코호트 2000명 (D1 schema). 그룹, 성별, 나이, 직장인 여부, PHQ-9, 라벨(PHQ-9 >= 10)

환자군에도 PHQ 낮은 사람(치료 중 remission), 대조군에도 높은 사람 -> 그룹 != 라벨
"""

import _path  # noqa: F401
import numpy as np
import pandas as pd
from config import (
    DATA,
    N_PARTICIPANTS,
    P_EMPLOYED_WORKING_AGE,
    P_FEMALE,
    P_PATIENT,
    PHQ_CONTROL,
    PHQ_CUTOFF,
    PHQ_PATIENT,
    RESULTS,
    SEED,
)

rng = np.random.default_rng(SEED)
n = N_PARTICIPANTS
group = np.where(rng.random(n) < P_PATIENT, "patient", "control")
sex = np.where(rng.random(n) < P_FEMALE, "F", "M")
# 20~30대 많고 60대 이상 적은 외래 코호트 모양 (17~80)
age = np.clip(rng.gamma(shape=3.5, scale=8.0, size=n) + 17, 17, 80).round().astype(int)
employed = ((age >= 20) & (age <= 59) & (rng.random(n) < P_EMPLOYED_WORKING_AGE)).astype(int)
phq = np.where(group == "patient", rng.normal(*PHQ_PATIENT, size=n), rng.normal(*PHQ_CONTROL, size=n))
phq = np.clip(phq, 0, 27).round().astype(int)
label = (phq >= PHQ_CUTOFF).astype(int)

cohort = pd.DataFrame(
    dict(
        participant_id=[f"P{i:04d}" for i in range(n)],
        group=group,
        sex=sex,
        age=age,
        employed=employed,
        phq9=phq,
        label_depressed=label,
    )
)
cohort["phq9_z"] = ((cohort.phq9 - cohort.phq9.mean()) / cohort.phq9.std()).round(4)  # 생성기 내부용, feature 아님
cohort.to_csv(DATA / "participants.csv", index=False)

summary = pd.DataFrame(
    [
        dict(stat="n_participants", value=n),
        dict(stat="patient_share", value=(group == "patient").mean().round(4)),
        dict(stat="female_share", value=(sex == "F").mean().round(4)),
        dict(stat="depressed_share", value=label.mean().round(4)),
        dict(stat="employed_share", value=employed.mean().round(4)),
        dict(stat="patient_but_not_depressed", value=int(((group == "patient") & (label == 0)).sum())),
        dict(stat="control_but_depressed", value=int(((group == "control") & (label == 1)).sum())),
        dict(stat="phq9_mean_patient", value=phq[group == "patient"].mean().round(2)),
        dict(stat="phq9_mean_control", value=phq[group == "control"].mean().round(2)),
    ]
)
summary.to_csv(RESULTS / "cohort_summary.csv", index=False)
print(summary.to_string(index=False))
print(pd.crosstab(cohort.group, cohort.label_depressed))
