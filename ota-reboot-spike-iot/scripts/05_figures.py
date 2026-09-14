import matplotlib
import pandas as pd
from common import DATA, FIG, G1, G2, MAIN, RES, SUB

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

d = pd.read_csv(RES / "daily_startdate.csv", parse_dates=["dt"])
m = pd.read_csv(RES / "daily_midnight.csv", parse_dates=["dt"])
ota = d[d.is_ota == 1]

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(d.dt, d.off_n, color=SUB, lw=1.2, label="off_n (fixed-only devices)")
ax[0].scatter(ota.dt, ota.off_n, color=MAIN, zorder=3, label="OTA day")
ax[0].axhline(d[d.is_ota == 0].off_n.median(), color=G2, ls="--", lw=0.8, label="baseline median")
ax[0].set_ylabel("off_n")
ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(d.dt, d.off_ratio, color=SUB, lw=1.2)
ax[1].scatter(ota.dt, ota.off_ratio, color=MAIN, zorder=3)
ax[1].set_ylabel("off_ratio")
ax[1].set_ylim(0.2, 0.55)
fig.suptitle("Daily off_n and off_ratio, start-date attribution", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig1_daily_offn_ota.png")
plt.close(fig)

hp = pd.read_csv(RES / "start_hour_profile.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(hp.hour - 0.2, hp.normal, width=0.4, color=G1, label="normal days")
ax.bar(hp.hour + 0.2, hp.spike, width=0.4, color=MAIN, label="spike days")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("first fixed-session start hour")
ax.set_ylabel("share of devices")
ax.legend(frameon=False)
ax.set_title("First fixed-session start hour, normal vs spike days", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig2_start_hour_profile.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(d.dt, d.mode_share, color=SUB, lw=1.2)
ax.scatter(ota.dt, ota.mode_share, color=MAIN, zorder=3, label="OTA day")
for thr, c in [(0.3, G2), (0.5, MAIN)]:
    ax.axhline(thr, color=c, ls="--", lw=0.8)
    ax.text(d.dt.iloc[1], thr + 0.01, f"threshold {thr}", color=c, fontsize=8)
ax.set_ylabel("mode-hour share")
ax.set_ylim(0, 0.6)
ax.legend(frameon=False)
ax.set_title("Concentration of first start hour, per day", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig3_mode_share_threshold.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(m.dt, m.off_n, color=G2, lw=1.2, label="midnight split")
ax.plot(d.dt, d.off_n, color=SUB, lw=1.2, label="start-date attribution")
for x in ota.dt:
    ax.axvline(x, color=MAIN, lw=0.8, alpha=0.6)
ax.set_ylabel("off_n")
ax.legend(frameon=False)
ax.set_title("off_n by attribution rule (midnight split vs start date)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig4_midnight_vs_startdate.png")
plt.close(fig)

sess = pd.read_csv(DATA / "sessions.csv", parse_dates=["start"])
otad = pd.read_csv(DATA / "ota_days.csv", parse_dates=["ota_dt"])
w = []
for r in otad.itertuples():
    T = r.ota_dt + pd.Timedelta(hours=int(r.ota_hour))
    w.append(sess[(sess.kind == "fixed") & (sess.start >= T) & (sess.start < T + pd.Timedelta(minutes=15))])
w = pd.concat(w)
fig, ax = plt.subplots(figsize=(8, 3.4))
bins = [0, 30, 60, 120, 240, 480, 1440, 2880, 5760, 10000]
for src, c, lab in [
    ("ota_resume", SUB, "path 1: resumed after reboot"),
    ("ota_lmk", MAIN, "path 2: started by reboot error"),
]:
    ax.hist(w[w.truth_source == src].duration_min, bins=bins, color=c, alpha=0.75, label=lab)
ax.set_xscale("log")
ax.set_xlabel("session length (min, log)")
ax.set_ylabel("sessions")
ax.legend(frameon=False)
ax.set_title("Fixed sessions started within 15 min after OTA, by truth path", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig5_ota_session_paths.png")
plt.close(fig)
print("figures written")
