"""공통 경로·상수. 모든 스크립트가 이 파일 기준 상대경로를 쓴다."""
from pathlib import Path
import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
for d in (DATA, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["happiness", "angry", "neutral", "sadness", "disgust", "surprise", "fear"]
# 실제 기록(슬라이드 07)의 라벨 비중을 그대로 쓴다.
EMOTION_P = np.array([0.235, 0.168, 0.168, 0.147, 0.120, 0.091, 0.072])
EMOTION_P = EMOTION_P / EMOTION_P.sum()

N_DAILY = 19374     # 포스터: 감정 대화 음성 데이터셋 활용 건수
N_COUNSEL_SYMPTOM = 16000  # 포스터: 상담 스크립트 활용 건수 (증상 의도)
N_COUNSEL_NORMAL = 4000    # 슬라이드 04의 추출 20,000건 − 활용 16,000건. 일상 의도로 보고 학습에서 뺀 것으로 가정
TEST_SIZE = 0.3            # 원 프로젝트 홀드아웃 7:3
N_SPEAKERS = 150    # 가정. 실제 기록에 화자 수는 없다.
N_MULTIMODAL = 2000 # 슬라이드 15: "전체 데이터의 10%인 2,000개"

AUDIO_COLS = (["zcr_mean", "zcr_std", "rms_mean", "rms_std"]
              + [f"mfcc{i}_mean" for i in range(1, 14)]
              + ["chroma_mean", "chroma_std", "duration"])

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for f in fm.findSystemFonts():
        if "NotoSansCJK" in f:
            fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK KR", "Apple SD Gothic Neo", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    return plt


def depression_split(df, seed=SEED):
    """우울 진단용 공통 분할. 출처×실제라벨 그룹으로 층화한 7:3 홀드아웃.
    v1(03)과 v2(04)가 같은 테스트셋을 쓰게 한다."""
    from sklearn.model_selection import train_test_split
    strat = df.source + "_" + df.intent.fillna("").str.startswith("정신증상").astype(int).astype(str)
    return train_test_split(df, test_size=TEST_SIZE, stratify=strat, random_state=seed)

# 시각화 팔레트 (분석 그림의 구분용. 사이트 톤은 07_site_figures.py가 sitestyle.py로 맞춘다)
C_TEXT, C_MM, C_AUDIO, C_NEG, C_GRAY, C_DARK = "#b8b8e0", "#6f6fd1", "#2e2e8a", "#d17f6f", "#c9c9c9", "#333333"
EMO_COLORS = {"happiness": "#f2b134", "angry": "#d1495b", "neutral": "#9aa5b1", "sadness": "#4f6d9a",
              "disgust": "#6a994e", "surprise": "#e07a5f", "fear": "#7b5ea7"}
