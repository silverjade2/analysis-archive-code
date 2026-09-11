"""저니맵. 유저는 어디서 멈추는가.

모델링 이전에 나온 발견 셋을 재현한다. 전체 가입 유저의 상태 퍼널은 대부분 가입 직후에서 멈춘다. 검사를 치르고
필수 프로필까지 채운 유저 중에서도 비어 있는 것은 선호 정보다. 연봉은 기본값 그대로, 복지 항목은 몇 개만 고른다.
휴면은 6개월에서 12개월간 로그인하지 않은 비중이 크고, 재방문은 공채 시즌 주변에 몰린다.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import DATA, FIG, RES, SEASONS, SERVICE_START, SNAPSHOT

matplotlib.use("Agg")

pop = pd.read_csv(DATA / "population.csv", parse_dates=["join_date", "last_login_date"])
users = pd.read_csv(DATA / "users.csv")
logins = np.load(DATA / "logins.npz")["logins"]

order = ["join_only", "test_only", "profile_only", "test_and_profile"]
labels = ["Joined only", "Test only", "Profile only", "Test + profile"]
cnt = pop.groupby("status")["n"].sum().reindex(order)
share = cnt / cnt.sum()
consent_rate = users["matching_use_yn"].mean()
funnel = pd.DataFrame({"status": labels, "users": cnt.values, "share": share.values.round(4)})
funnel.loc[len(funnel)] = [
    "  └ consented (core users)",
    int(round(cnt.iloc[-1] * consent_rate)),
    round(share.iloc[-1] * consent_rate, 4),
]
funnel.to_csv(RES / "journey_funnel.csv", index=False)
print(funnel.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.8))
vals = list(cnt.values) + [cnt.iloc[-1] * consent_rate]
names = labels + ["  └ consented"]
colors = ["#a0aec0", "#a0aec0", "#a0aec0", "#2b6cb0", "#c05621"]
ax.barh(names[::-1], vals[::-1], color=colors[::-1])
for i, v in enumerate(vals[::-1]):
    ax.text(v + 3000, i, f"{v / 1000:,.1f}k  ({v / cnt.sum():.1%})", va="center", fontsize=9)
ax.set_xlim(0, cnt.sum() * 0.95)
ax.set_xlabel("users")
ax.set_title("Where 451k registered users stopped (snapshot)", loc="left", fontsize=11)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig1_journey_funnel.png", dpi=150)

pref = pd.DataFrame(
    {
        "metric": [
            "salary left at default",
            "welfare: 1 item or fewer",
            "welfare: fewer than 5 items",
            "preference complete (salary set & welfare ≥ 5)",
        ],
        "share": [
            (users["pref_salary_default_yn"] == "Y").mean(),
            (users["pref_welfare_cnt"] <= 1).mean(),
            (users["pref_welfare_cnt"] < 5).mean(),
            users["truth_pref_complete"].mean(),
        ],
    }
).round(4)
by_consent = (
    users.groupby("matching_use_yn")
    .agg(
        salary_default=("pref_salary_default_yn", lambda s: (s == "Y").mean()),
        welfare_lt5=("pref_welfare_cnt", lambda s: (s < 5).mean()),
        pref_complete=("truth_pref_complete", "mean"),
    )
    .round(4)
)
pref.to_csv(RES / "journey_preference_fields.csv", index=False)
by_consent.to_csv(RES / "journey_preference_by_consent.csv")
print("\n", pref.to_string(index=False), "\n\nby consent:\n", by_consent.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
h = users["pref_welfare_cnt"].value_counts().sort_index()
ax.bar(h.index, h.values / h.sum(), color="#2b6cb0")
ax.axvline(4.5, color="#c05621", ls="--", lw=1)
ax.text(4.7, ax.get_ylim()[1] * 0.9, f"< 5 items: {(users['pref_welfare_cnt'] < 5).mean():.0%}", color="#c05621")
ax.set_xlabel("welfare preferences selected")
ax.set_ylabel("share of users")
ax.set_title("Preference fields are what users skip", loc="left", fontsize=11)
ax = axes[1]
x = np.arange(2)
w = 0.35
ax.bar(x - w / 2, by_consent["salary_default"], w, label="salary left at default", color="#a0aec0")
ax.bar(x + w / 2, by_consent["pref_complete"], w, label="preference complete", color="#2b6cb0")
ax.set_xticks(x, ["not consented", "consented"])
ax.set_ylim(0, 1)
ax.set_ylabel("share")
ax.legend(frameon=False, fontsize=9)
ax.set_title("…and it separates the two groups", loc="left", fontsize=11)
for a in axes:
    a.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig2_preference_fields.png", dpi=150)

gap = (SNAPSHOT - pop["last_login_date"]).dt.days
dorm = pd.DataFrame(
    {
        "metric": ["no login for 6+ months", "no login for 1+ year", "logged in within 30 days"],
        "share": [
            np.average(gap >= 182, weights=pop["n"]),
            np.average(gap >= 365, weights=pop["n"]),
            np.average(gap <= 30, weights=pop["n"]),
        ],
    }
).round(4)
dorm.to_csv(RES / "journey_dormancy.csv", index=False)
print("\n", dorm.to_string(index=False))

days = pd.date_range(SERVICE_START, SNAPSHOT, freq="D")
weekly = pd.Series(logins.sum(0), index=days).resample("W").sum()[:-1]  # 마지막 불완전한 주는 제외
pos = users["matching_use_yn"].values == 1
weekly_pos = pd.Series(logins[pos].sum(0), index=days).resample("W").sum()[:-1]
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(weekly.index, weekly.values, color="#2b6cb0", lw=1.5, label="all test+profile users")
ax.plot(weekly_pos.index, weekly_pos.values, color="#c05621", lw=1.2, label="consented users")
for a, b in SEASONS:
    ax.axvspan(a, b, color="#f6ad55", alpha=0.25, lw=0)
ax.text(SEASONS[1][0], ax.get_ylim()[1] * 0.95, "recruiting seasons", color="#c05621", fontsize=9, va="top")
ax.set_ylabel("weekly logins")
ax.set_title(
    "Returns cluster around recruiting seasons and, for consented users, after consent", loc="left", fontsize=11
)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig3_weekly_logins.png", dpi=150)
print("figures saved")
