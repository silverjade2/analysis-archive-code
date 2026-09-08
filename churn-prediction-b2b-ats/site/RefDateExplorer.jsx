// 기준일 탐색기 — 라벨의 '오늘'을 움직이면 같은 회사의 이탈 라벨이 얼마나 바뀌는가.
// 데이터: site/ref_sweep.json (06_reference_date_sweep.py 출력). 사이트의 기존 위젯(절단 시점 탐색기)과 같은 스타일로 붙인다.
import { useMemo, useState } from "react";
import data from "./ref_sweep.json";

const W = 640, H = 220, PAD = { l: 44, r: 12, t: 12, b: 28 };

export default function RefDateExplorer() {
  const live = useMemo(() => data.rows.filter((r) => r.mode === "live"), []);
  const frozen = useMemo(() => data.rows.filter((r) => r.mode === "frozen"), []);
  const baseIdx = live.findIndex((r) => r.ref_date === data.ref_date_base);
  const [idx, setIdx] = useState(baseIdx);
  const x = (i) => PAD.l + (i / (live.length - 1)) * (W - PAD.l - PAD.r);
  const y = (v) => PAD.t + (1 - v) * (H - PAD.t - PAD.b);
  const path = (rows, key) => rows.map((r, i) => `${i ? "L" : "M"}${x(i)},${y(r[key])}`).join(" ");
  const cur = live[idx], curF = frozen[idx];
  const fmt = (v) => `${(v * 100).toFixed(1)}%`;
  return (
    <div className="widget">
      <div className="widget-title">기준일 탐색기 — 이탈 라벨의 '오늘'은 언제인가</div>
      <p className="widget-desc">
        원본 노트북은 <code>datetime.today()</code>를 기준으로 "최종 종료일 + 90일 초과"를 이탈로 잡았다.
        슬라이더로 실행일을 옮기면 같은 {data.n.toLocaleString()}개 회사의 라벨이 어떻게 바뀌는지 보인다.
        파랑은 그 시점까지의 새 계약을 반영해 다시 돌린 경우, 빨강은 {data.ref_date_base} 스냅샷을 그대로 두고 날짜만 지난 경우.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {[0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}><line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="var(--border)" strokeDasharray="2 3" />
            <text x={PAD.l - 6} y={y(v) + 4} fontSize="10" textAnchor="end" fill="var(--muted)">{v * 100}%</text></g>
        ))}
        <path d={path(frozen, "churn_rate")} fill="none" stroke="#c94a4a" strokeWidth="2" />
        <path d={path(live, "churn_rate")} fill="none" stroke="#2f6fd6" strokeWidth="2" />
        <line x1={x(baseIdx)} x2={x(baseIdx)} y1={PAD.t} y2={H - PAD.b} stroke="var(--muted)" strokeDasharray="3 3" />
        <line x1={x(idx)} x2={x(idx)} y1={PAD.t} y2={H - PAD.b} stroke="var(--fg)" />
        {live.filter((_, i) => i % 6 === 0).map((r, k) => (
          <text key={k} x={x(k * 6)} y={H - 8} fontSize="10" textAnchor="middle" fill="var(--muted)">{r.ref_date.slice(2, 7)}</text>
        ))}
      </svg>
      <input type="range" min={0} max={live.length - 1} value={idx} onChange={(e) => setIdx(+e.target.value)} style={{ width: "100%" }} />
      <div className="widget-stats">
        <div><span className="label">실행일</span><strong>{cur.ref_date}</strong></div>
        <div><span className="label">이탈률 (데이터 갱신)</span><strong style={{ color: "#2f6fd6" }}>{fmt(cur.churn_rate)}</strong></div>
        <div><span className="label">이탈률 (스냅샷 고정)</span><strong style={{ color: "#c94a4a" }}>{fmt(curF.churn_rate)}</strong></div>
        <div><span className="label">{data.ref_date_base} 라벨과 달라진 회사</span><strong>{fmt(cur.flip_rate)} / {fmt(curF.flip_rate)}</strong></div>
      </div>
      <p className="widget-note">원본 실행일 이후로 달라진 라벨은 전부 고객 → 이탈 방향이다. 데이터를 갱신해도 이탈 판정은 되돌아오지 않는다 — 원본 규칙에서 '이탈했다가 돌아온' 회사는 최종 종료일이 새로 잡혀 다시 고객이 되어야 하지만, 이 가상데이터에서는 이탈 후 재계약이 발생하지 않기 때문이다. 실행일을 앞으로 돌리면 반대 방향(이탈 → 고객)이 생긴다.</p>
    </div>
  );
}
