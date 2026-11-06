"""데이터 직무 공고 앱. 탭 둘: 기술 수요 vs 오픈소스 활동량, 직무군별 업무 유형 비중

data/ 의 csv만 읽는다. 업무 유형은 회차 추세를 그리지 않음 (회차 차이의 구간이 넓어서)
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"

MART = "SKILL_DEMAND_VS_ACTIVITY"
DEMAND = "INT_SKILL_DEMAND"
PYPI = "STG_PYPI_WEEKLY"
GITHUB = "STG_GITHUB_WEEKLY"
TASK_MIX = "TASK_MIX"
TASK_GROUP_MIX = "TASK_GROUP_MIX"
TAXONOMY = "task_taxonomy"

# csv 값 그대로
QUADRANTS = [
    "수요 높음 · 활동 높음",
    "수요 높음 · 활동 낮음",
    "수요 낮음 · 활동 높음",
    "수요 낮음 · 활동 낮음",
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK_GRAY = "#7f8590"  # 점 이름과 점선. 밝은, 어두운 배경 양쪽에서 읽히는 회색 (기본 검정은 어두운 테마에서 안 보임)
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ROLES = ["DA", "BA/BI", "DS"]  # DE, ML/AI 엔지니어는 평가 라벨이 없어 보정값 없음
TASK_GROUPS = ["분석", "모델링", "데이터 엔지니어링", "AI 엔지니어링", "연구"]

st.set_page_config(page_title="데이터 직무 공고 분석", layout="wide")


@st.cache_data
def load(name: str, **kwargs):
    df = pd.read_csv(DATA / f"{name}.csv", **kwargs)
    return df.rename(columns=str.upper)


def task_grid(mix, roles):
    """직무군 x 출처 x 업무 유형 전체 칸. 글로벌은 세 회차 합 (period ALL) 만"""
    g = mix[mix.ROLE_FAMILY.isin(roles) & ((mix.SOURCE == "jobkko") | (mix.PERIOD == "ALL"))].copy()
    g["COLUMN"] = np.where(g.SOURCE == "jobkko", "국내", "글로벌")
    g["COL_ORDER"] = np.where(g.SOURCE == "jobkko", "0", "1")
    g["N_POSTINGS"] = g.N_POSTINGS.astype(int)
    for c in ["POSTING_SHARE", "POSTING_LO", "POSTING_HI", "PRIMARY_SHARE", "PRIMARY_LO", "PRIMARY_HI"]:
        g[c] = g[c].astype(float)
    return g


def pct_text(share, lo, hi):
    """표 글자: 국내는 '12.3', 글로벌은 '12.3 [8.0 ~ 17.0]'"""
    base = (share * 100).round(1).astype(str)
    band = " [" + (lo * 100).round(1).astype(str) + " ~ " + (hi * 100).round(1).astype(str) + "]"
    return np.where(lo.isna(), base, base + band)


st.title("데이터 직무 공고 분석")
st.caption("값은 Snowflake 안의 dbt로 만든 표를 내려받은 것이다")
tab2, tab1 = st.tabs(["기술 수요 vs 활동량", "업무 유형 비중"])

# ---------------------------------------------------------------- 기술 수요
with tab2:
    mart = load(MART)
    measured = mart[mart.DOWNLOADS_13W.notna()].copy()
    measured["SHARE_PCT"] = measured.POSTING_SHARE * 100
    measured["DL_M"] = measured.DOWNLOADS_13W / 1e6

    st.header("기술 수요 vs 오픈소스 활동량")
    st.caption(
        "수요: 직접 운영하는 채용 공고 수집기의 국내 데이터 직무 공고 그룹 중 그 기술이 나온 비율. "
        "활동량: 최근 13주 PyPI 다운로드 (기술에 연결한 패키지 합). 사분면 기준은 측정 기술의 중앙값"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("사전 기술", f"{len(mart)}개")
    c2.metric("활동량 측정", f"{len(measured)}개")
    c3.metric("공고 그룹", f"{int(load(DEMAND).N_GROUPS.max()):,}개")

    # 두 값 모두 기술마다 수십~수백 배 차이라 양쪽 로그 축
    # 이름은 전부 달면 겹쳐서 사분면마다 중앙값에서 가장 먼 3개만 기본으로
    med_s, med_dl = measured.SHARE_PCT.median(), measured.DL_M.median()
    measured["DIST"] = (np.log(measured.SHARE_PCT / med_s) ** 2 + np.log(measured.DL_M / med_dl) ** 2) ** 0.5
    default_labels = measured.sort_values("DIST", ascending=False).groupby("QUADRANT").head(3).SKILL.tolist()
    show_all = st.toggle("모든 기술 이름 보기", value=False)
    picked = st.multiselect("이름을 표시할 기술", measured.SKILL.sort_values().tolist(), default=default_labels)
    measured["LABEL"] = np.where(show_all | measured.SKILL.isin(picked), measured.SKILL, "")

    color = alt.Color(
        "QUADRANT:N", title="사분면", scale=alt.Scale(domain=QUADRANTS, range=COLORS), legend=alt.Legend(orient="top")
    )
    base = alt.Chart(measured).encode(
        x=alt.X(
            "SHARE_PCT:Q",
            title="국내 공고 등장률 (%, 로그)",
            scale=alt.Scale(type="log"),
            axis=alt.Axis(values=[1, 2, 5, 10, 20, 40], grid=False),
        ),
        y=alt.Y(
            "DL_M:Q",
            title="최근 13주 PyPI 다운로드 (백만, 로그)",
            scale=alt.Scale(type="log"),
            axis=alt.Axis(values=[1, 10, 100, 1000, 10000], gridOpacity=0.3),
        ),
    )
    rule = alt.Chart(measured).mark_rule(strokeDash=[4, 4], opacity=0.6, color=INK_GRAY)
    rules = rule.encode(x="median(SHARE_PCT):Q") + rule.encode(y="median(DL_M):Q")
    points = base.mark_circle(size=110, opacity=0.95, stroke="white", strokeWidth=1.5).encode(
        color=color,
        tooltip=[
            alt.Tooltip("SKILL:N", title="기술"),
            alt.Tooltip("SHARE_PCT:Q", title="공고 등장률 %", format=".1f"),
            alt.Tooltip("DL_M:Q", title="PyPI 13주 (백만)", format=",.1f"),
            alt.Tooltip("GROWTH_3Y:Q", title="3년 전 대비 배수", format=".1f"),
            alt.Tooltip("STRENGTH:N", title="연결 강도"),
        ],
    )
    labels = base.mark_text(align="left", dx=8, dy=-6, fontSize=12, color=INK_GRAY).encode(text="LABEL:N")
    st.altair_chart((rules + points + labels).properties(height=560), width="stretch")
    st.caption("점선 = 측정 기술의 중앙값 (사분면 경계). 점에 마우스를 올리면 값이 보인다")

    st.subheader("기술별 추이")
    skill = st.selectbox("기술", measured.sort_values("POSTING_SHARE", ascending=False).SKILL.tolist())
    pypi_all = load(PYPI, parse_dates=["WEEK"])
    pypi = pypi_all[pypi_all.SKILL == skill].groupby("WEEK", as_index=False).DOWNLOADS.sum()
    pypi["DL_M"] = pypi.DOWNLOADS / 1e6
    st.altair_chart(
        alt.Chart(pypi)
        .mark_line(strokeWidth=2, color=COLORS[0])
        .encode(
            x=alt.X("WEEK:T", title=None),
            y=alt.Y("DL_M:Q", title="주간 PyPI 다운로드 (백만)"),
            tooltip=[alt.Tooltip("WEEK:T", title="주"), alt.Tooltip("DL_M:Q", title="백만", format=",.2f")],
        )
        .properties(height=260, title=f"{skill}: PyPI 주간 다운로드"),
        width="stretch",
    )
    gh_all = load(GITHUB, parse_dates=["WEEK"])
    gh_rows = gh_all[(gh_all.SKILL == skill) & (gh_all.EVENT_TYPE == "WatchEvent")]
    gh = gh_rows.groupby("WEEK", as_index=False).EVENTS.sum().rename(columns={"EVENTS": "STARS"})
    if len(gh):
        st.altair_chart(
            alt.Chart(gh)
            .mark_line(strokeWidth=2, color=COLORS[1])
            .encode(
                x=alt.X("WEEK:T", title=None),
                y=alt.Y("STARS:Q", title="주간 GitHub 스타"),
                tooltip=[alt.Tooltip("WEEK:T", title="주"), alt.Tooltip("STARS:Q", title="스타", format=",")],
            )
            .properties(height=200, title=f"{skill}: GitHub 주간 스타 (2025-03 까지, 이후 수집 단절)"),
            width="stretch",
        )

    st.subheader("전체 표")
    table = mart.assign(
        공고_등장률_pct=(mart.POSTING_SHARE * 100).round(1),
        글로벌_표본_pct=(mart.WDC_SAMPLE_SHARE * 100).round(1),
        PyPI_13주_백만=(mart.DOWNLOADS_13W / 1e6).round(1),
        증가_3년=mart.GROWTH_3Y.round(1),
    )[["SKILL", "CATEGORY", "공고_등장률_pct", "글로벌_표본_pct", "PyPI_13주_백만", "증가_3년", "QUADRANT", "STRENGTH"]]
    st.dataframe(table.sort_values("공고_등장률_pct", ascending=False), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 업무 유형
with tab1:
    st.header("데이터 직무의 업무 유형 비중")
    st.caption(
        "국내 = 직접 운영하는 채용 공고 수집기의 공고에 Sonnet 라벨 (대표 업무 1개 + 업무 목록, "
        "경계 사례는 사람이 덮어씀) 을 그대로 센 값. "
        "글로벌 = WDC JobPosting 2022-12, 2023-12, 2024-12 전체 공고 (중복 제거) 에 분류 모델을 적용하고 "
        "평가 표본 510건으로 보정한 추정값, [ ] 는 95% 구간. 분모는 데이터 직무가 아닌 공고 (none) 를 뺀 공고 수"
    )
    st.info(
        "이 탭은 직무군 사이 구성 비교와 국내 vs 글로벌을 본다. 글로벌 회차 사이 변화 (추세) 는 그리지 않는다: "
        "2022-12와 2024-12의 보정 차이는 구간 폭이 평균 27%p 라 이 표본으로는 확인되지 않는다. "
        "회색 칸은 분류 정확도가 낮은 업무 (데이터 엔지니어링 묶음, AI 엔지니어링 묶음, DS 밖의 모델링 묶음) 라 "
        "근거로 읽지 않는다."
    )
    taxonomy = load(TAXONOMY)
    taxonomy = taxonomy.assign(G=taxonomy.TASK_GROUP.map(TASK_GROUPS.index)).sort_values(["G", "TASK_ID"])
    task_order = taxonomy.TASK.tolist()
    grid = task_grid(load(TASK_MIX), ROLES)

    n_cells = grid.drop_duplicates(["SOURCE", "ROLE_FAMILY"])
    k1, k2, k3 = st.columns(3)
    k1.metric("국내 공고", f"{n_cells[n_cells.SOURCE == 'jobkko'].N_POSTINGS.sum():,}건")
    k2.metric("글로벌 공고 (3회차, 주 대상 3직무군)", f"{n_cells[n_cells.SOURCE != 'jobkko'].N_POSTINGS.sum():,}건")
    k3.metric("업무 유형", f"{len(taxonomy)}개")

    metric = st.radio(
        "지표",
        ["업무 목록에 있는 비율", "대표 업무 비율"],
        horizontal=True,
        help="목록: 공고 하나에 업무가 여러 개라 열 합이 100%를 넘는다. 대표: 열 합 100%",
    )
    key = "POSTING" if metric.startswith("업무 목록") else "PRIMARY"
    grid["PCT"] = grid[f"{key}_SHARE"] * 100
    grid["LO"] = grid[f"{key}_LO"] * 100
    grid["HI"] = grid[f"{key}_HI"] * 100
    grid["CELL_TEXT"] = grid.PCT.round().astype(int).astype(str)
    grid["BAND"] = np.where(grid.LO.isna(), "", grid.LO.round(1).astype(str) + " ~ " + grid.HI.round(1).astype(str))
    grid["IS_REF"] = grid.EVIDENCE == "참고용"
    # 축 글자 두 줄 (열 이름 | 공고 수). 직무군마다 공고 수가 달라 x 축을 따로 둔다
    grid["XLAB"] = grid.COLUMN + "|n=" + grid.N_POSTINGS.map("{:,}".format)

    st.subheader("직무군 x 출처별 업무 비중 (%)")
    max_pct = float(grid.PCT.max())
    heat = alt.Chart().encode(
        x=alt.X(
            "XLAB:N",
            title=None,
            sort=alt.EncodingSortField("COL_ORDER"),
            axis=alt.Axis(orient="top", labelAngle=0, labelExpr="split(datum.label, '|')", ticks=False, domain=False),
        ),
        y=alt.Y("TASK:N", title=None, sort=task_order, axis=alt.Axis(ticks=False, domain=False, labelLimit=220)),
    )
    tip = [
        alt.Tooltip("ROLE_FAMILY:N", title="직무군"),
        alt.Tooltip("COLUMN:N", title="출처"),
        alt.Tooltip("TASK:N", title="업무"),
        alt.Tooltip("PCT:Q", title=f"{metric} %", format=".1f"),
        alt.Tooltip("BAND:N", title="95% 구간 (글로벌)"),
        alt.Tooltip("EVIDENCE:N", title="근거 여부"),
        alt.Tooltip("N_POSTINGS:Q", title="열 공고 수"),
    ]
    rects = (
        heat.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2)
        .encode(
            color=alt.Color(
                "PCT:Q",
                title="%",
                scale=alt.Scale(domain=[0, max_pct], range=SEQ_BLUE),
                legend=alt.Legend(orient="bottom", gradientLength=240),
            ),
            tooltip=tip,
        )
        .transform_filter("!datum.IS_REF")
    )
    ref_rects = (
        heat.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2, color="#e4e4e0")
        .encode(tooltip=tip)
        .transform_filter("datum.IS_REF")
    )
    # 칸 바탕색은 테마를 따르지 않아 글자색 고정
    # scale=None 색 인코딩을 rects 와 겹치면 vega-lite 컴파일 오류라 글자색별로 층을 나눔
    grid["TEXT_COLOR"] = np.where(grid.IS_REF, "#8a8a86", np.where(grid.PCT > max_pct * 0.55, "#ffffff", "#1f1f1f"))
    texts = [
        heat.mark_text(fontSize=11, color=c).encode(text="CELL_TEXT:N").transform_filter(f"datum.TEXT_COLOR == '{c}'")
        for c in grid.TEXT_COLOR.unique()
    ]
    st.altair_chart(
        alt.layer(rects, ref_rects, *texts, data=grid)
        .properties(width=150, height=26 * len(task_order))
        .facet(column=alt.Column("ROLE_FAMILY:N", title=None, sort=ROLES, header=alt.Header(labelFontSize=14)))
        .resolve_scale(x="independent"),
        width="content",
    )
    st.caption(
        "칸에 마우스를 올리면 95% 구간과 공고 수가 보인다. "
        "국내 열은 구간이 없는 라벨 그대로의 비율이다 (공고 수가 적어 작은 차이는 읽지 않는다)."
    )

    st.subheader("국내 vs 글로벌")
    c1, c2 = st.columns([2, 3])
    role = c1.radio("직무군", ROLES, horizontal=True)
    show_ref = c2.checkbox("회색 (근거 아님) 업무도 보기", value=False)
    pool = grid[grid.ROLE_FAMILY == role].copy()
    if not show_ref:
        # 국내 행은 근거 값이 라벨이라 글로벌 행 기준으로 업무를 고른다
        pool = pool[~pool.TASK.isin(pool[pool.IS_REF].TASK)]
    sides = ["국내", "글로벌"]
    wide = pool.pivot(index="TASK", columns="COLUMN", values="PCT").reindex(columns=sides)
    gap_order = (wide[sides[0]] - wide[sides[1]]).sort_values(ascending=False).index.tolist()
    dumb = alt.Chart(pool).encode(
        y=alt.Y("TASK:N", title=None, sort=gap_order, axis=alt.Axis(ticks=False, domain=False, labelLimit=220)),
        x=alt.X("PCT:Q", title=f"{metric} (%)", axis=alt.Axis(gridOpacity=0.3)),
    )
    link = dumb.mark_line(color="#9a9a96", strokeWidth=2).encode(detail="TASK:N")
    band = (
        alt.Chart(pool[pool.LO.notna()])
        .mark_rule(color=COLORS[1], strokeWidth=4, opacity=0.35)
        .encode(y=alt.Y("TASK:N", sort=gap_order), x="LO:Q", x2="HI:Q")
    )
    dots = dumb.mark_circle(size=120, opacity=1, stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "COLUMN:N", title=None, scale=alt.Scale(domain=sides, range=COLORS[:2]), legend=alt.Legend(orient="top")
        ),
        tooltip=[
            alt.Tooltip("COLUMN:N", title="출처"),
            alt.Tooltip("TASK:N", title="업무"),
            alt.Tooltip("PCT:Q", title=f"{metric} %", format=".1f"),
            alt.Tooltip("BAND:N", title="95% 구간"),
            alt.Tooltip("N_POSTINGS:Q", title="공고 수"),
        ],
    )
    # height 가 범례와 축까지 포함한 크기라 그만큼 더해야 행 라벨이 전부 남는다
    st.altair_chart((link + band + dots).properties(height=28 * max(len(gap_order), 1) + 110), width="stretch")
    n_side = pool.drop_duplicates("COLUMN").set_index("COLUMN").N_POSTINGS
    st.caption(
        f"{role}: 국내 {n_side.get(sides[0], 0):,}건, 글로벌 {n_side.get(sides[1], 0):,}건. "
        "위쪽일수록 국내 비중이 글로벌보다 큰 업무. 주황 막대 = 글로벌 95% 구간"
    )

    # 직무군 단위로 읽는 근거는 유형이 아니라 이 표
    st.subheader("업무 묶음별 비중 (%)")
    gmix = load(TASK_GROUP_MIX)
    gmix = gmix[gmix.ROLE_FAMILY.isin(ROLES) & ((gmix.SOURCE == "jobkko") | (gmix.PERIOD == "ALL"))].copy()
    gmix["COLUMN"] = np.where(gmix.SOURCE == "jobkko", "국내", "글로벌")
    for c in ["POSTING_SHARE", "POSTING_LO", "POSTING_HI", "PRIMARY_SHARE", "PRIMARY_LO", "PRIMARY_HI"]:
        gmix[c] = gmix[c].astype(float)
    gk = "POSTING" if key == "POSTING" else "PRIMARY"
    gmix["VALUE"] = pct_text(gmix[f"{gk}_SHARE"], gmix[f"{gk}_LO"], gmix[f"{gk}_HI"])
    gmix["VALUE"] = np.where(gmix.EVIDENCE == "참고용", gmix.VALUE + " (참고용)", gmix.VALUE)
    gmix["열"] = gmix.ROLE_FAMILY + " " + gmix.COLUMN
    gcols = [f"{r} {c}" for r in ROLES for c in sides]
    gtable = gmix.pivot(index="TASK_GROUP", columns="열", values="VALUE").reindex(index=TASK_GROUPS, columns=gcols)
    st.dataframe(gtable, width="stretch")

    st.subheader("전체 표")
    grid["VALUE"] = pct_text(grid[f"{key}_SHARE"], grid[f"{key}_LO"], grid[f"{key}_HI"])
    grid["VALUE"] = np.where(grid.IS_REF, grid.VALUE + " (참고용)", grid.VALUE)
    grid["열"] = grid.ROLE_FAMILY + " " + grid.COLUMN
    col_order = (
        grid.drop_duplicates("열")
        .assign(R=lambda d: d.ROLE_FAMILY.map(ROLES.index))
        .sort_values(["R", "COL_ORDER"])["열"]
        .tolist()
    )
    table = grid.pivot(index="TASK", columns="열", values="VALUE").reindex(index=task_order, columns=col_order)
    st.dataframe(table, width="stretch")
