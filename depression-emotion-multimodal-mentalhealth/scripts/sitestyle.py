"""사이트 톤 (Pretendard, 사이트 팔레트, webp). 07이 쓴다. 숫자는 만들지 않는다."""
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from common import FIGURES

OUT = FIGURES / "site"
BLUE, ORANGE, GRAY, LIGHT, DARK = "#2563eb", "#ea580c", "#a1a1aa", "#e4e4e7", "#18181b"
LIGHT_BLUE, LIGHT_ORANGE, MUTED, RED = "#bfdbfe", "#fed7aa", "#71717a", "#dc2626"
# 화자 8명 / 감정 7종처럼 범주가 많을 때만 쓰는 팔레트 (tailwind 600 계열)
CAT8 = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#db2777", "#ca8a04", "#0891b2", "#52525b"]
# 발산 컬러맵: 음(파랑) - 0(흰색) - 양(붉은색). 혼동행렬 차이와 토큰 기여에 쓴다
DIVERGING = LinearSegmentedColormap.from_list("site_diverging", [BLUE, "#ffffff", RED])


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
