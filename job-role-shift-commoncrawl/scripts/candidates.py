"""기술 후보 96개. 기존 37개는 data/base_skills.csv의 패턴, 새 후보는 data/new_skills.json의 별칭

별칭은 소문자, 영문 별칭은 앞뒤가 영숫자가 아닐 때만 맞춘다
정규식은 Python re, Java (Spark), Snowflake에서 같은 뜻이 되게 lookaround 없이 쓴다
"""

import csv
import json
import re

from common import DATA

BASE_CSV = DATA / "base_skills.csv"

NEW = json.loads((DATA / "new_skills.json").read_text(encoding="utf-8"))  # (이름, 분류, 별칭)


def alias_regex(alias: str) -> str:
    # 영숫자로 시작, 끝나는 별칭은 앞뒤 경계를 둔다. 한글 별칭은 조사가 붙어 경계 없이
    # re.escape는 공백, 하이픈까지 막아 Snowflake에서 뜻이 애매해져 메타문자만 [.] 꼴로 감싼다
    a = alias if "\\" in alias else "".join(f"[{ch}]" if ch in ".^$*+?(){}|" else ch for ch in alias)
    # Snowflake 정규식은 lookaround가 없어 기존 37개와 같은 (^|[^a-z0-9]) 꼴로
    pre = r"(^|[^a-z0-9])" if re.match(r"[a-z0-9]", alias) else ""
    post = r"([^a-z0-9]|$)" if re.search(r"[a-z0-9]$", alias) else ""
    return pre + a + post


B = r"(^|[^a-z0-9])"
E = r"([^a-z0-9]|$)"

# 영어 공고 오탐 수정. 영어 일반 단어 대안만 좁히고 한국어 대안은 그대로. 기술: [(바꿀 부분, 새 부분)]
PATTERN_FIX = {
    "LLM": [("|prompt", f"|prompt engineer|prompt design|{B}prompting{E}")],
    "추천": [("recommend|", f"recommender|{B}recommendation (systems?|engines?|models?|algorithms?|platforms?){E}|")],
    "코호트 · 리텐션": [
        ("cohort|", "cohort analys|cohort retention|"),
        (
            "|retention",
            "|(user|customer|client|subscriber|member|donor)s? retention|retention (rate|analys|curve|metric)",
        ),
    ],
    "수요 예측": [(f"{B}forecasting{E}", f"{B}(demand|sales) forecasting{E}")],
    "Docker": [(f"{B}container{E}", "containeriz|container orchestration|container technolog")],
}


def load(fixed: bool = True) -> list[dict]:
    """fixed=False면 PATTERN_FIX 적용 전 패턴"""
    rows = [
        {"skill": r["skill"], "category": r["category"], "source": "base", "pattern": r["pattern"]}
        for r in csv.DictReader(open(BASE_CSV, encoding="utf-8"))
    ]
    base = {r["skill"] for r in rows}
    for name, cat, aliases in NEW:
        assert name not in base, name
        pattern = "|".join(alias_regex(a) for a in aliases)
        rows.append({"skill": name, "category": cat, "source": "new", "pattern": pattern})
    if fixed:
        for r in rows:
            for old, new in PATTERN_FIX.get(r["skill"], []):
                assert old in r["pattern"], (r["skill"], old)
                r["pattern"] = r["pattern"].replace(old, new)
    return rows
