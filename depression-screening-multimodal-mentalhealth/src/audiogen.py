"""음향 피처 20차원 생성 (eGeMAPS 요약 통계 흉내)

피처 = 성별 기준선 + 화자 지문 + 과제 효과 + 우울 효과(AUDIO_SIGNAL * z) + 잡음. 표준화 단위로 만든 뒤 물리 단위로
"""

import numpy as np
from config import AUDIO_SIGNAL, MFCC_NOISE_SD, PROSODY_NOISE_SD, SPEAKER_FP_SD

AUDIO_COLS = [
    "f0_mean",
    "f0_sd",
    "f0_range",
    "loudness_mean",
    "loudness_sd",
    "jitter",
    "shimmer",
    "hnr",
    "speech_rate",
    "pause_ratio",
    "mean_pause",
    "voiced_ratio",
    "mfcc1",
    "mfcc2",
    "mfcc3",
    "mfcc4",
    "mfcc5",
    "mfcc6",
    "spectral_flux",
    "duration_sec",
]
# 우울 효과 방향 (표준화 단위). 문헌 방향: 단조로운 억양, 낮은 에너지, 느린 말, 긴 쉼
DEP_EFFECT = np.array(
    [-0.2, -1.0, -1.0, -0.8, -0.6, 0.4, 0.4, -0.5, -0.9, 1.0, 0.8, -0.5, -0.2, 0.1, -0.1, 0.0, 0.1, 0.0, -0.4, 0.3]
)
# 과제 효과
TASK_EFFECT = {
    "happy_memory": np.array([0.2, 0.3, 0.3, 0.2, 0.1, 0, 0, 0.1, 0, -0.1, -0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0.4]),
    "hard_memory": np.array(
        [-0.3, -0.2, -0.2, -0.3, 0, 0.1, 0.1, -0.1, -0.2, 0.3, 0.3, -0.1, 0, 0, 0, 0, 0, 0, -0.1, 0.5]
    ),
    "pic_positive": np.array([0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "pic_negative": np.array([-0.1, -0.1, -0.1, -0.1, 0, 0, 0, 0, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "pic_neutral": np.zeros(20),
    "count_up": np.array([0, -0.4, -0.5, 0, -0.3, 0, 0, 0.2, 1.0, -0.5, -0.4, 0.3, 0, 0, 0, 0, 0, 0, 0.3, -1.0]),
    "count_down": np.array([0, -0.4, -0.5, 0, -0.3, 0, 0, 0.2, 0.9, -0.4, -0.3, 0.3, 0, 0, 0, 0, 0, 0, 0.3, -0.9]),
    "reading": np.array([0, -0.2, -0.2, 0.1, -0.2, 0, 0, 0.3, 0.5, -0.3, -0.3, 0.4, 0, 0, 0, 0, 0, 0, 0.2, 0.2]),
}
# 화자 지문 크기. mfcc, f0, 음량에 크게, 쉼과 속도에 작게
FP_SCALE = np.array(
    [1.2, 0.5, 0.5, 0.9, 0.5, 0.6, 0.6, 0.8, 0.5, 0.3, 0.3, 0.5, 1.3, 1.3, 1.3, 1.2, 1.2, 1.2, 0.6, 0.2]
)
# 물리 단위 변환 (mean, sd)
UNITS = {
    "f0_mean": (170, 30),
    "f0_sd": (28, 8),
    "f0_range": (110, 30),
    "loudness_mean": (62, 5),
    "loudness_sd": (7, 2),
    "jitter": (0.012, 0.004),
    "shimmer": (0.08, 0.02),
    "hnr": (14, 3),
    "speech_rate": (4.2, 0.7),
    "pause_ratio": (0.22, 0.07),
    "mean_pause": (0.55, 0.18),
    "voiced_ratio": (0.68, 0.08),
    "mfcc1": (0, 1),
    "mfcc2": (0, 1),
    "mfcc3": (0, 1),
    "mfcc4": (0, 1),
    "mfcc5": (0, 1),
    "mfcc6": (0, 1),
    "spectral_flux": (0.35, 0.1),
    "duration_sec": (48, 14),
}


def speaker_fingerprint(rng, sex):
    fp = rng.normal(0, SPEAKER_FP_SD, size=20) * FP_SCALE
    fp[0] += 1.2 if sex == "F" else -1.2  # 성별 기본 주파수
    return fp


def recording_features(fp, task, z, n_words, rng):
    noise_sd = np.where(np.isin(np.arange(20), [12, 13, 14, 15, 16, 17]), MFCC_NOISE_SD, PROSODY_NOISE_SD)
    x = fp + TASK_EFFECT[task] + AUDIO_SIGNAL * z * DEP_EFFECT + rng.normal(0, 1.0, size=20) * noise_sd
    x[19] += 0.015 * (n_words - 30)  # 말한 양에 비례
    out = {}
    for i, c in enumerate(AUDIO_COLS):
        m, s = UNITS[c]
        out[c] = float(m + s * x[i])
    out["jitter"] = max(out["jitter"], 0.002)
    out["shimmer"] = max(out["shimmer"], 0.01)
    out["pause_ratio"] = float(np.clip(out["pause_ratio"], 0.02, 0.7))
    out["voiced_ratio"] = float(np.clip(out["voiced_ratio"], 0.3, 0.95))
    out["duration_sec"] = max(out["duration_sec"], 6.0)
    return out
