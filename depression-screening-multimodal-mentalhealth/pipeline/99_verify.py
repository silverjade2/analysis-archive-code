"""99. 자동 검증. (1) 누수 점검: 어떤 피처 행렬에도 truth_*·라벨·participant_id가 없는지.
(2) 오라클/상한: 화자 사전확률이 녹음 랜덤에서 1.0인지(누수 상한 존재), 참가자 분할에서 모델이 오라클을 넘지 않는지.
(3) 숫자 manifest: 글에서 인용할 값을 한 파일로 모은다."""

import _path  # noqa: F401
import pandas as pd
from audiogen import AUDIO_COLS
from config import RESULTS
from evalutil import load_recordings

problems = []
df = load_recordings()
# (1) 음성 피처에 잠재변수/식별자가 섞이지 않았는지
leaked = set(AUDIO_COLS) & {
    c
    for c in df.columns
    if c.startswith("truth_") or c in ("label_depressed", "phq9", "phq9_z", "participant_id", "group")
}
if leaked:
    problems.append(f"audio feature leak: {leaked}")
# 토큰에 라벨 흔적이 없는지
if df.tokens.str.contains("label|phq", case=False).any():
    problems.append("token contains label string")
# truth_ 컬럼이 존재하고 채점 전용인지 (recordings.csv에는 있어야 하고, 어떤 스크립트도 피처로 쓰지 않음)
truth_cols = [c for c in df.columns if c.startswith("truth_")]

# (2) 화자 사전확률 상한
sc = pd.read_csv(RESULTS / "split_comparison.csv")
sp = sc[(sc.split == "recording_random") & (sc.model == "speaker_prior")]
if len(sp) and sp.rec_auc_mean.iloc[0] < 0.999:
    problems.append("speaker_prior not ~1.0 under recording_random (leakage path missing)")
# 참가자 분할 융합이 완벽(1.0)이면 의심
fus = sc[(sc.split == "participant") & (sc.model == "fusion")].par_auc_mean.iloc[0]
if fus > 0.99:
    problems.append(f"participant fusion AUC too high ({fus}) — possible leak")


# (3) manifest
def g(csv, q):
    return pd.read_csv(RESULTS / csv).query(q)


M = {}
coh = pd.read_csv(RESULTS / "cohort_summary.csv").set_index("stat").value
M["n_participants"] = int(coh["n_participants"])
M["n_recordings"] = len(df)
M["patient_share"] = float(coh["patient_share"])
M["depressed_share"] = float(coh["depressed_share"])
M["patient_but_not_depressed"] = int(coh["patient_but_not_depressed"])
M["control_but_depressed"] = int(coh["control_but_depressed"])
for _, r in sc.iterrows():
    M[f"auc__{r.split}__{r.model}__recording"] = r.rec_auc_mean
    M[f"auc__{r.split}__{r.model}__participant"] = r.par_auc_mean
ld = pd.read_csv(RESULTS / "label_definition_comparison.csv")
for _, r in ld.iterrows():
    M[f"label__{r.trained_on}__{r.tokens}__par_auc_vs_phq"] = r.par_auc_vs_phq
    M[f"label__{r.trained_on}__{r.tokens}__prob_patient_not_depressed"] = r.mean_prob_patient_not_depressed
ta = pd.read_csv(RESULTS / "task_aggregation.csv").set_index("aggregation").par_auc
for k in ta.index:
    M[f"task_agg__{k}"] = float(ta[k])
tr = pd.read_csv(RESULTS / "emotion_transfer.csv")
M["emotion_transfer_par_auc"] = tr[(tr.signal.str.contains("transfer")) & (tr.level == "participant")].auc.iloc[0]
M["emotion_transfer_spearman"] = tr[
    (tr.signal.str.contains("transfer")) & (tr.level == "participant")
].spearman_phq.iloc[0]
M["dedicated_text_par_auc"] = tr[(tr.signal.str.contains("dedicated")) & (tr.level == "participant")].auc.iloc[0]
M["emotion_accuracy"] = float(pd.read_csv(RESULTS / "emotion_accuracy.csv").query("scope=='all'").accuracy.iloc[0])
sg = pd.read_csv(RESULTS / "subgroup_auc.csv")
for _, r in sg[(sg.model == "fusion") & (sg.dimension.isin(["group", "subgroup_employed"]))].iterrows():
    M[f"subgroup__{r.subgroup}__fusion_auc"] = r.auc
va = pd.read_csv(RESULTS / "eda_audio_variance_decomposition.csv")
M["audio_eta2_speaker_max"] = float(va.eta2_speaker.max())
M["audio_eta2_label_max"] = float(va.eta2_label.max())
manifest = pd.DataFrame(sorted(M.items()), columns=["key", "value"])
manifest.to_csv(RESULTS / "number_manifest.csv", index=False)

print("truth columns (scoring-only):", truth_cols)
print("PROBLEMS:", problems if problems else "none")
print(manifest.to_string(index=False))
