"""기술 수요 vs 오픈소스 활동량 대시보드

Streamlit in Snowflake 에서 돈다. dbt 가 만든 BIGDATA.ANALYTICS_* 를 읽기만 한다.
사분면 4개 색은 범주 팔레트 1 ~ 4번. 노랑과 초록은 대비가 낮아 점 이름 라벨과 표를 같이 둔다
"""

import altair as alt
import numpy as np
import streamlit as st
from snowflake.snowpark.context import get_active_session

MART = "BIGDATA.ANALYTICS_MARTS.SKILL_DEMAND_VS_ACTIVITY"
PYPI = "BIGDATA.ANALYTICS_STAGING.STG_PYPI_WEEKLY"
GITHUB = "BIGDATA.ANALYTICS_STAGING.STG_GITHUB_WEEKLY"

QUADRANTS = ["수요 높음 · 활동 높음", "수요 높음 · 활동 낮음", "수요 낮음 · 활동 높음", "수요 낮음 · 활동 낮음"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

st.set_page_config(page_title="기술 수요 vs 오픈소스 활동량", layout="wide")
session = get_active_session()


@st.cache_data(ttl=3600)
def load(sql: str):
    return session.sql(sql).to_pandas()


mart = load(f"select * from {MART}")
measured = mart[mart.DOWNLOADS_13W.notna()].copy()
measured["SHARE_PCT"] = measured.POSTING_SHARE * 100
measured["DL_M"] = measured.DOWNLOADS_13W / 1e6

st.header("기술 수요 vs 오픈소스 활동량")
st.caption(
    "수요: 국내 데이터 직무 공고 그룹 중 그 기술이 나온 비율, "
    "활동량: 최근 13주 PyPI 다운로드 (기술에 연결한 패키지 합). 사분면 기준은 측정 기술의 중앙값"
)

c1, c2, c3 = st.columns(3)
c1.metric("사전 기술", f"{len(mart)}개")
c2.metric("활동량 측정", f"{len(measured)}개")
c3.metric(
    "공고 그룹", f"{int(load('select max(n_groups) n from BIGDATA.ANALYTICS_INTERMEDIATE.INT_SKILL_DEMAND').N[0]):,}개"
)

# 사분면 산점도: 두 값 모두 기술마다 수십 ~ 수백 배 차이라 양쪽 로그 축 (왼쪽 아래에 몰리던 점이 펴진다)
# 이름은 전부 달면 겹치므로 사분면마다 중앙값에서 가장 먼 3개만 기본으로 달고, 나머지는 고르거나 마우스로 본다
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
# 사분면 경계 (중앙값) 를 옅은 점선으로
rules = alt.Chart(measured).mark_rule(strokeDash=[4, 4], opacity=0.4).encode(x="median(SHARE_PCT):Q") + alt.Chart(
    measured
).mark_rule(strokeDash=[4, 4], opacity=0.4).encode(y="median(DL_M):Q")
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
# 글자 색은 지정하지 않아 Streamlit 테마 (밝은 화면, 어두운 화면) 의 글자색을 따른다
labels = base.mark_text(align="left", dx=8, dy=-6, fontSize=12).encode(text="LABEL:N")
st.altair_chart((rules + points + labels).properties(height=560), use_container_width=True)
st.caption("점선 = 측정 기술의 중앙값 (사분면 경계). 점에 마우스를 올리면 값이 보인다")

# 기술 하나의 추이
st.subheader("기술별 추이")
skill = st.selectbox("기술", measured.sort_values("POSTING_SHARE", ascending=False).SKILL.tolist())
pypi = load(
    f"select week, sum(downloads) / 1e6 as dl_m from {PYPI} where skill = '{skill.replace(chr(39), '')}' "
    "group by week order by week"
)
st.altair_chart(
    alt.Chart(pypi)
    .mark_line(strokeWidth=2, color=COLORS[0])
    .encode(
        x=alt.X("WEEK:T", title=None),
        y=alt.Y("DL_M:Q", title="주간 PyPI 다운로드 (백만)"),
        tooltip=[alt.Tooltip("WEEK:T", title="주"), alt.Tooltip("DL_M:Q", title="백만", format=",.2f")],
    )
    .properties(height=260, title=f"{skill}: PyPI 주간 다운로드"),
    use_container_width=True,
)
gh = load(
    f"select week, sum(events) as stars from {GITHUB} where skill = '{skill.replace(chr(39), '')}' "
    "and event_type = 'WatchEvent' group by week order by week"
)
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
        use_container_width=True,
    )

# 표: 색에 기대지 않고 읽을 수 있게 전체 값
st.subheader("전체 표")
table = mart.assign(
    공고_등장률_pct=(mart.POSTING_SHARE * 100).round(1),
    글로벌_표본_pct=(mart.WDC_SAMPLE_SHARE * 100).round(1),
    PyPI_13주_백만=(mart.DOWNLOADS_13W / 1e6).round(1),
    증가_3년=mart.GROWTH_3Y.round(1),
)[["SKILL", "CATEGORY", "공고_등장률_pct", "글로벌_표본_pct", "PyPI_13주_백만", "증가_3년", "QUADRANT", "STRENGTH"]]
st.dataframe(table.sort_values("공고_등장률_pct", ascending=False), use_container_width=True, hide_index=True)
