"""경로, 시드, 생성 파라미터, 과제 정의"""

from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "outputs" / "results", ROOT / "outputs" / "figures"
for d in (DATA, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# ---- D1 (정신건강진단 및 예측을 위한 멀티모달 데이터) 스키마를 따르는 코호트 ----
N_PARTICIPANTS = 2000
P_PATIENT = 0.565  # 환자군 비율 (실제 데이터 56.5%)
P_FEMALE = 0.647  # 여성 비율 (실제 64.7%)
PHQ_CUTOFF = 10  # 라벨 "우울한 기분" = PHQ-9 >= 10. 실제 클래스 비율 41.8%에 맞춰 분포를 조정
PHQ_PATIENT = (12.5, 5.5)  # 환자군 PHQ-9 (mean, sd). 치료 중 관해가 섞임
PHQ_CONTROL = (4.0, 3.2)  # 대조군
P_EMPLOYED_WORKING_AGE = 0.62  # 20~59세 중 직장인 비율 (가정; 서비스 타깃 하위군 평가용)

TASKS = [
    "happy_memory",
    "hard_memory",
    "pic_positive",
    "pic_negative",
    "pic_neutral",
    "count_up",
    "count_down",
    "reading",
]
TASK_KO = {
    "happy_memory": "행복했던 기억",
    "hard_memory": "힘들었던 기억",
    "pic_positive": "긍정 그림 설명",
    "pic_negative": "부정 그림 설명",
    "pic_neutral": "중립 그림 설명",
    "count_up": "숫자 세기(오름)",
    "count_down": "숫자 세기(내림)",
    "reading": "문단 낭독",
}
NARRATIVE_TASKS = TASKS[:5]  # 텍스트에 내용이 있는 과제
FIXED_TEXT_TASKS = TASKS[5:]  # 텍스트가 모두 같은 과제

# ---- 심어둔 구조. 신호 강도는 원본 프로젝트의 텍스트/음성 AUC 수준에 맞춰 조정한 값 ----
TEXT_SIGNAL = 1.0  # PHQ z가 서술 과제 문장 선택에 미치는 강도
AUDIO_SIGNAL = 0.28  # PHQ z점수가 음향 피처에 미치는 강도 (표준편차 단위)
MFCC_NOISE_SD = 0.35  # 음색 피처 녹음 간 잡음. 작아서 화자 안에서 안정적 = 참가자 누수 경로
PROSODY_NOISE_SD = 1.0  # 운율, 에너지 피처 녹음 간 잡음
SPEAKER_FP_SD = 1.0  # 화자 지문 크기 (sd 단위). 참가자 누수의 원천
P_TREATMENT_MENTION = {"patient": 0.60, "control": 0.03}  # 치료 맥락 언급 확률. 환자군 shortcut의 원천
P_STT_ERROR = 0.04  # STT 오인식으로 어절이 바뀌는 확률
P_IDIOLECT = 0.55  # 서술 과제에서 참가자 고유 말버릇이 나올 확률 (텍스트 쪽 참가자 누수의 경로)

# ---- D2 (감성 대화 말뭉치) 스키마를 따르는 감정 코퍼스 ----
N_EMOTION_SENT = 24000
EMOTIONS = ["기쁨", "슬픔", "분노", "불안", "당황", "상처"]
SITUATIONS = ["직장", "학업", "연애", "가족", "건강", "대인관계", "금전"]
SITUATION_P = [0.26, 0.12, 0.13, 0.15, 0.11, 0.14, 0.09]
AGE_GROUPS = ["청년", "중년", "노년"]

TEST_SIZE = 0.3
N_FOLDS = 5
