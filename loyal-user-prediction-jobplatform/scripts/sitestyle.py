"""그림의 사이트 시각 톤 (Pretendard, 사이트 팔레트, webp)."""
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from common import FIG

OUT = FIG / "site"
BLUE, ORANGE, GRAY, LIGHT, DARK = "#2563eb", "#ea580c", "#a1a1aa", "#e4e4e7", "#18181b"
LIGHT_BLUE, LIGHT_ORANGE, MUTED = "#bfdbfe", "#fed7aa", "#71717a"

# feature id → 한국어 라벨 (위젯과 같은 매핑)
NAMES = {
    "login_counts": "로그인 횟수 (6개월)", "days_since_last_login": "마지막 로그인 경과일",
    "user_cnt": "알림 응답 횟수", "total_apply_cnt": "전체 지원 횟수", "apply_try_cnt": "지원 시도 횟수",
    "apply_cnt": "지원 완료 횟수", "company_cnt": "지원 기업 수", "acc_apply_counts": "검사 응시 횟수",
    "test_cnt": "지원 후 검사 횟수", "complete_cnt": "전형 완료 횟수",
    "acca_t_score": "검사 점수", "acca_grade": "검사 등급", "mental_health_grade": "정서 등급",
    "pref_welfare_cnt": "복지 선호 개수", "pref_salary_default_yn": "연봉 기본값 여부",
    "marketing_consent_yn": "마케팅 수신 동의", "join_year": "가입연도", "join_month": "가입월",
    "age": "나이", "career_year": "경력 연수", "extra": "학교 등급", "final_edu_level": "학력",
    "gender": "성별", "career_type": "신입/경력",
}
# (대부분) 타깃의 결과인 feature: 동의 이후의 활동
AFTER_TARGET = {"login_counts", "days_since_last_login", "user_cnt", "total_apply_cnt", "apply_try_cnt",
                "apply_cnt", "company_cnt", "acc_apply_counts", "test_cnt", "complete_cnt"}
# 심어둔 동의의 동인
PLANTED = {"acca_t_score", "pref_welfare_cnt", "pref_salary_default_yn", "marketing_consent_yn", "join_month"}


def name(f):
    return NAMES.get(f, f)


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in list(Path.home().glob("Library/Fonts/Pretendard-*.otf")) + list(Path("/Library/Fonts").glob("Pretendard-*.otf")):
        font_manager.fontManager.addfont(str(f))
    have = any("Pretendard" in f.name for f in font_manager.fontManager.ttflist)
    plt.rcParams.update({
        "font.family": "Pretendard" if have else "AppleGothic",
        "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#a1a1aa", "axes.labelcolor": "#3f3f46",
        "xtick.color": "#52525b", "ytick.color": "#52525b",
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlecolor": "#18181b", "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False,
        "grid.color": "#e4e4e7", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return have


def save(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.save(OUT / f"{name}.webp", "WEBP", quality=88, method=6)
    img.save(OUT / f"{name}.png")
    plt.close(fig)
    print(f"saved {name}.webp  {img.size[0]}x{img.size[1]}")
