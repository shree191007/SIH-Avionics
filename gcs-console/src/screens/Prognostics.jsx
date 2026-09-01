import { useState, useEffect, useRef, useMemo } from 'react';
import { AreaChart, Area, Line, XAxis, YAxis, ReferenceLine, ResponsiveContainer } from 'recharts';
import VerdictBadge from '../components/VerdictBadge';
import Field from '../components/Field';
import { useScenario } from '../context/ScenarioContext';
import { buildRulFunnel, failLimit } from '../mock/data';

export default function Prognostics() {
  const { liveFrame } = useScenario();
  const { rul } = liveFrame;
  const funnel = useMemo(() => buildRulFunnel(rul.p05, rul.median, rul.p95), [rul]);

  const [frame, setFrame] = useState(funnel.length - 1);
  const [playing, setPlaying] = useState(false);
  const ref = useRef(null);

  useEffect(() => { setFrame(funnel.length - 1); }, [funnel]);

  useEffect(() => {
    if (!playing) { clearInterval(ref.current); return; }
    setFrame(0);
    ref.current = setInterval(() => {
      setFrame((f) => {
        if (f >= funnel.length - 1) { clearInterval(ref.current); setPlaying(false); return f; }
        return f + 1;
      });
    }, 220);
    return () => clearInterval(ref.current);
  }, [playing, funnel]);

  const data = funnel.slice(0, frame + 1);

  return (
    <div className="screen">
      <div className="panel">
        <div className="panel-header">
          <span className="mono">RUL</span>
          <Field label="p05" field="ParticleFilter.rul_p05" equation="5th percentile of remaining-useful-life particles" value={rul.p05}>
            <span className="mono" style={{ color: 'var(--critical)' }}>p05 {rul.p05}h</span>
          </Field>
          <span className="mono">● {rul.median}h</span>
          <span className="mono" style={{ color: 'var(--nominal)' }}>p95 {rul.p95}h</span>
        </div>
        <div className="mono" style={{ fontSize: 11, color: 'var(--ink-400)', marginBottom: 8 }}>{rul.note}</div>

        <div className="panel-title">RUL fan</div>
        <div className="rul-bar">
          <div className="rul-bar-track">
            <div className="rul-bar-band" style={{ left: `${(rul.p05 / 320) * 100}%`, width: `${((rul.p95 - rul.p05) / 320) * 100}%` }} />
            <div className="rul-bar-marker" style={{ left: `${(rul.median / 320) * 100}%` }} />
          </div>
        </div>

        <hr />
        <div className="panel-header">
          <div className="panel-title" style={{ margin: 0 }}>funnel replay (last 6h of flight)</div>
          <button onClick={() => setPlaying(true)} disabled={playing}>▶ replay</button>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={data}>
            <XAxis dataKey="t" tick={{ fontSize: 10, fill: 'var(--ink-400)' }} tickFormatter={(t) => `t-${6 - Math.round((t / 24) * 6)}h`} stroke="var(--line-500)" />
            <YAxis tick={{ fontSize: 10, fill: 'var(--ink-400)' }} stroke="var(--line-500)" />
            <ReferenceLine y={failLimit} stroke="var(--critical)" strokeDasharray="4 4" label={{ value: 'fail limit', fill: 'var(--critical)', fontSize: 10, position: 'insideTopRight' }} />
            <Area type="monotone" dataKey="p95" stroke="none" fill="var(--ink-700)" isAnimationActive={false} />
            <Area type="monotone" dataKey="p05" stroke="none" fill="var(--ink-900)" isAnimationActive={false} />
            <Line type="monotone" dataKey="median" stroke="var(--borrowed)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
        <VerdictBadge verdict={rul.status} />
      </div>
    </div>
  );
}
