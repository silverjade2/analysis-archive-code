"""그림에 쓸 한글 폰트. 설치된 것 중 첫 번째를 고른다."""

from matplotlib import font_manager, rcParams

FONT_CANDIDATES = ["AppleGothic", "Pretendard", "NanumGothic"]


def setup_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in FONT_CANDIDATES if n in installed), "DejaVu Sans")
    rcParams["font.family"] = chosen
    rcParams["axes.unicode_minus"] = False
    return chosen
