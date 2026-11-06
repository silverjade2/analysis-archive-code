"""결과 csv를 사이트 위젯 json 3개로: role-share, task-mix, skill-trend"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "outputs" / "results"

ROLES = ["BA/BI", "DE", "DA", "DS", "ML/AI 엔지니어", "데이터 직무 합계", "전체 공고"]
GROUPS = ["분석", "모델링", "데이터 엔지니어링", "AI 엔지니어링", "연구"]


def rd(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dump(name, obj):
    with open(RESULTS / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def role_share():
    share_rows = rd("task1_role_share.csv")
    out = {"scopes": {}}
    for scope in ["미국", "미국 사이트 고정"]:
        rows = []
        for g in ROLES:
            r = next(x for x in share_rows if x["scope"] == scope and x["g"] == g)
            rows.append(
                {
                    "g": g,
                    "n": [round(float(r["2022-12"])), round(float(r["2023-12"])), round(float(r["2024-12"]))],
                    "nChange": round(float(r["n_change_pct"]), 2),
                    "shareChange": round(float(r["share_change_pct"]), 2),
                }
            )
        out["scopes"][scope] = rows
    return out


def task_mix():
    with open(DATA / "task_taxonomy.csv", encoding="utf-8") as f:
        tax = {r["task_id"]: r for r in csv.DictReader(f)}
    mix_rows = rd("task1_article2_task_mix.csv")
    by_type = {r["type"]: int(r["n_pos"]) for r in rd("task1_b_test_by_type.csv")}
    units = []
    for g in GROUPS:
        units.append({"id": f"[그룹] {g}", "label": g, "group": g, "isGroup": True, "nPos": by_type[f"[그룹] {g}"]})
        for tid, t in tax.items():
            if t["task_group"] == g:
                units.append({"id": tid, "label": t["task"], "group": g, "isGroup": False, "nPos": by_type[tid]})
    status = {r["task_group"]: r["group_status"] for r in mix_rows}
    cells = {}
    sizes = {}
    for r in mix_rows:
        cells.setdefault(r["slice"], {}).setdefault(r["kind"], {}).setdefault(r["unit"], {})[r["role_family"]] = [
            float(r["share_ppi"]),
            float(r["ci_lo"]),
            float(r["ci_hi"]),
            float(r["share_raw"]),
            float(r["share_label_test"]),
        ]
        sizes.setdefault(r["slice"], {})[r["role_family"]] = [int(r["n_pop"]), int(r["n_test"])]
    none = {
        r["role"]: [float(r["value"]), float(r["lo"]), float(r["hi"])]
        for r in rd("task1_article2_label_mix.csv")
        if r["measure"] == "none"
    }
    return {
        "units": units,
        "groupStatus": status,
        "cells": cells,
        "sizes": sizes,
        "none": {k: none[k] for k in ["DA", "BA/BI", "DS"]},
    }


def skill_trend():
    skills = []
    values = {}
    for r in rd("task1_article2_skill_trend.csv"):
        if (r["skill"], r["source"]) not in skills:
            skills.append((r["skill"], r["source"]))
        values.setdefault(r["scope"], {}).setdefault(r["role"], {})[r["skill"]] = [
            float(r["rate_2022"]),
            float(r["rate_2023"]),
            float(r["rate_2024"]),
            round(float(r["diff_pp"]), 2),
            round(float(r["diff_lo"]), 2),
            round(float(r["diff_hi"]), 2),
            int(r["n_2022"]),
            int(r["n_2024"]),
        ]
    return {"skills": [{"name": s, "source": src} for s, src in skills], "values": values}


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    dump("role-share.json", role_share())
    dump("task-mix.json", task_mix())
    dump("skill-trend.json", skill_trend())
