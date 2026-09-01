import { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ReferenceLine, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import Field from '../components/Field';
import { useScenario } from '../context/ScenarioContext';
import { missionAlternatives, lifeBudget, probeCandidates, probeRanked } from '../mock/data';

const RISK_COLOR = { LOW: 'var(--nominal)', MEDIUM: 'var(--caution)', HIGH: 'var(--critical)' };

export default function Mission() {
  const [alt, setAlt] = useState('nominal');
  const a = missionAlternatives[alt];
  const { scenario, liveFrame } = useScenario();
  const { mission } = liveFrame;

  return (
    <div className="screen">
      <div className="panel">
        <div className="panel-header">
          <span className="mono">risk: <span style={{ color: RISK_COLOR[mission.risk.sortie] }}>{mission.risk.sortie}</span> (sortie) · <span style={{ color: RISK_COLOR[mission.risk.endurance] }}>{mission.risk.endurance}</span> (endurance)</span>
        </div>
        <div className="mono" style={{ fontSize: 11, color: 'var(--ink-400)', marginBottom: 8 }}>{mission.note}</div>

        <div className="alt-toggle">
          {Object.keys(missionAlternatives).map((k) => (
            <button key={k} className={alt === k ? 'active' : ''} onClick={() => setAlt(k)}>{k}</button>
          ))}
        </div>
        <div className="alt-readout mono">
          <span style={{ color: RISK_COLOR[a.risk] }}>{a.risk}</span> · endurance {a.endurance} — {a.note}
        </div>

        <hr />
        <div className="panel-title">life budget this mission</div>
        <div className="mono">
          <Field label="ΔLife spent" field="CostModel.lifeBudget.spentHours" equation="Σ ΔLife over probes flown this session" value={lifeBudget.spentHours}>
            {lifeBudget.spentHours}h
          </Field> spent across {lifeBudget.probesUsed} probes (cap {lifeBudget.sessionCapHours}h)
        </div>
        <div className="encounter-bar-track" style={{ marginTop: 6 }}>
          <div className="encounter-bar-fill" style={{ width: `${(lifeBudget.spentHours / lifeBudget.sessionCapHours) * 100}%`, background: 'var(--probe-action)' }} />
        </div>
      </div>

      <div className="panel">
        <div className="panel-title" style={{ margin: 0 }}>Probe Selection</div>
        {mission.probeNeeded ? (
          <>
            <div className="probe-layout">
              <table className="probe-table mono">
                <thead>
                  <tr><th>id</th><th>maneuver</th><th>G(m)</th><th>C(m)</th><th>value</th></tr>
                </thead>
                <tbody>
                  {probeRanked.map((p) => (
                    <tr key={p.id} style={{ color: p.value < 0 ? 'var(--ink-400)' : 'var(--ink-000)' }}>
                      <td>{p.id}</td><td>{p.name}</td><td>{p.G}</td><td>{p.C}</td><td>{p.value >= 0 ? '+' : ''}{p.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
                  <XAxis type="number" dataKey="C" name="C(m)" domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--ink-400)' }} stroke="var(--line-500)" label={{ value: 'C(m) cost', fill: 'var(--ink-400)', fontSize: 10, position: 'insideBottom', offset: -4 }} />
                  <YAxis type="number" dataKey="G" name="G(m)" domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--ink-400)' }} stroke="var(--line-500)" label={{ value: 'G(m) benefit', fill: 'var(--ink-400)', fontSize: 10, angle: -90, position: 'insideLeft' }} />
                  <ReferenceLine segment={[{ x: 0, y: 0.15 }, { x: 1, y: 1.15 }]} stroke="var(--borrowed)" strokeDasharray="4 4" />
                  <Tooltip contentStyle={{ background: 'var(--ink-800)', border: '1px solid var(--line-500)', fontSize: 11 }} />
                  <Scatter data={probeCandidates} isAnimationActive={false}>
                    {probeCandidates.map((p, i) => (
                      <Cell key={i} fill={p.best ? 'var(--nominal)' : p.feasible ? 'var(--borrowed)' : 'var(--ink-400)'} r={p.best ? 6 : 3} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <div className="mono" style={{ color: 'var(--nominal)' }}>
              value-of-information: E[benefit]={probeRanked[0].G} &gt; cost={probeRanked[0].C} → FLY IT
            </div>
          </>
        ) : (
          <div className="mono" style={{ fontSize: 12, color: 'var(--ink-400)', padding: '8px 0' }}>
            {liveFrame.resolved
              ? `no probe needed — ${scenario.verdict === 'DEGRADED_INPUT' ? 'this is an input-availability issue, not an attribution ambiguity a probe could resolve' : 'the current fault is already locally identifiable'}.`
              : 'assessing whether a probe will be needed…'}
          </div>
        )}
      </div>
    </div>
  );
}
