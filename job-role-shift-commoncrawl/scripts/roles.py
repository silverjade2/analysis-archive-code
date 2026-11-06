"""공고 제목으로 직무군 라벨. 위에서부터 먼저 맞는 것 하나

주 대상 DA, BA/BI, DS. 비교군 DE, ML/AI 엔지니어. DA 인접(재무, 영업, 리스크 등의 analyst)은 민감도 확인용
정규식은 소문자 제목에 쓰고 Snowflake, Spark에서도 같은 뜻이 되게 lookaround 없이 쓴다
"""

import re

B = r"(^|[^a-z])"  # 단어 앞 경계
E = r"([^a-z]|$)"  # 단어 뒤 경계

DS_CORE = (
    r"data[ -]?scien|데이터\s?(사이언|과학)|사이언티스트"
    rf"|(machine learning|{B}ml|{B}ai|applied|decision|nlp|computer vision|research data) scientist"
)
BA_BI = r"business analy|비즈니스\s?분석|사업\s?분석|business intelligence|(^|[^a-z0-9])bi($|[^a-zà-ỹ0-9-])"
DE = r"데이터\s?엔지니어|data engineer|analytics engineer|애널리틱스\s?엔지니어|데이터\s?아키텍"
ML_AI = (
    r"ml\s?engineer|machine learning engineer|머신\s?러닝\s?엔지니어|딥\s?러닝\s?엔지니어|ai\s?engineer"
    r"|ai\s?엔지니어|mlops|llm\s?엔지니어|ai\s?개발|ai\s?developer"
)
ANALYST_ANY = r"분석가|analyst|애널리스트|데이터\s?분석|analytics|data analy"  # 이 안에서만 아래 층을 나눈다
DA_EXCLUDE = r"behaviou?r analy|bcba"
DA_CORE = (
    r"data[ -]?analy|data ?(&|and) ?analy|데이터\s?분석|데이터\s?애널|애널리스트|분석가|datenanaly|analyste de donn"
    rf"|{B}analytics"
    r"|(reporting|insights?|product|marketing|growth|web|digital|crm|seo|ppc|paid media|social media|campaign"
    r"|statistical"
    r"|customer data|customer insights?)[a-z ,&/-]{0,25} analy(st|ste)"
)
DA_UNRELATED = (
    rf"behaviou?r|bcba|security|cyber|{B}soc{E}|threat|incident|disaster|{B}iam{E}|service desk|help ?desk|support"
    rf"|{B}qa{E}|{B}qc{E}|quality|test|systems?{E}|programmer|application|functional|{B}sap{E}|workday|oracle"
    rf"|{B}epic{E}|kronos|hris|configuration|implementation|network|infrastructure|compliance|{B}kyc{E}|{B}aml{E}"
    rf"|sanction|legal|contract|grant|policy|legislat|payroll"
    rf"|benefits|payable|receivable|billing|collections|laboratory|{B}lab{E}|chemi|cell culture|asbestos|nurse"
    rf"|relations"
)
DA_ADJACENT = (
    r"financ|fp ?(&|＆|and) ?a|treasury|account|audit|budget|pricing|revenue|credit|risk|investment|portfolio|equity"
    r"|venture|actuar|quant|underwrit|trading|market|research|econom|supply chain|logistic|demand|inventory|replenish"
    r"|procure|sourcing|operations"
    rf"|sales|workforce|commercial|strategy|cost|energy|sustainab|planning|performance|{B}gis{E}|compensation|{B}analyst{E}"
)
ROLE_RULES = [("DS", DS_CORE), ("BA/BI", BA_BI), ("DE", DE), ("ML/AI 엔지니어", ML_AI)]
MAIN = {"DA", "BA/BI", "DS"}
MIN_BODY_CHARS = 200  # 이보다 짧은 본문은 판단 불가. 중복 키도 제목 기준


def role_family(title: str | None) -> str:
    t = (title or "").lower()
    for name, pat in ROLE_RULES:
        if re.search(pat, t):
            return name
    if not re.search(ANALYST_ANY, t):
        return "그 밖"
    if re.search(DA_CORE, t) and not re.search(DA_EXCLUDE, t):
        return "DA"
    if re.search(DA_UNRELATED, t):
        return "그 밖"
    return "DA 인접" if re.search(DA_ADJACENT, t) else "그 밖"


def role_sql(col: str) -> str:
    """role_family와 같은 판정의 SQL CASE (DuckDB). col은 소문자 제목 열"""

    def m(pat: str) -> str:
        return "regexp_matches({}, '{}')".format(col, pat.replace("'", "''"))

    rules = " ".join(f"WHEN {m(p)} THEN '{n}'" for n, p in ROLE_RULES)
    return (
        f"CASE {rules} WHEN NOT {m(ANALYST_ANY)} THEN '그 밖'"
        f" WHEN {m(DA_CORE)} AND NOT {m(DA_EXCLUDE)} THEN 'DA'"
        f" WHEN {m(DA_UNRELATED)} THEN '그 밖'"
        f" WHEN {m(DA_ADJACENT)} THEN 'DA 인접' ELSE '그 밖' END"
    )
