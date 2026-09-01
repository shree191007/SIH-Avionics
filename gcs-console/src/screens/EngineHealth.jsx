import { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import Field from '../components/Field';
import VerdictBadge from '../components/VerdictBadge';
import { useScenario } from '../context/ScenarioContext';
import { regimeTimeline, buildEgtSeries, spatialDecomposition, regimeEncounters } from '../mock/data';

const REGIME_COLOR = (r) => (r.startsWith('CLIMB') ? 'var(--caution)' : r.startsWith('CRUISE') ? 'var(--nominal)' : 'var(--borrowed)');

export default function EngineHealth() {
  const [scrubIdx, setScrubIdx] = useState(regimeTimeline.length - 1);
  const current = regimeTimeline[scrubIdx];
  const { scenario, liveFrame } = useScenario();
  const { health } = liveFrame;

  // Generated once per scenario (final z, not the live one) so the RNG
  // stays stable across ticks — the live reveal just slices how much of
  // the trace has "streamed in" so far.
  const finalSeries = useMemo(() => (
    scenario.health.cylZ ? scenario.health.cylZ.map((z) => buildEgtSeries(z)) : null
  ), [scenario]);

  const revealCount = Math.max(2, Math.round(liveFrame.progress * 40));

  const cylinders = health.cylZ ? health.cylZ.map((z, i) => ({
    cyl: i + 1, z, warn: health.warnCyl === i + 1, series: finalSeries[i].slice(0, revealCount),
  })) : null;

  return (
    <div className="screen">
      <div className="panel">
        <div className="panel-header">
          <VerdictBadge verdict={health.status} suffix={` — ${health.note}`} />
          <span className="mono">regime: {current.regime}</span>
        </div>

        <div className="panel-title">regime timeline (scrubbable)</div>
        <div className="regime-timeline">
          {regimeTimeline.map((r, i) => (
            <div
              key={i}
              className="regime-timeline-cell"
              style={{ background: REGIME_COLOR(r.regime), opacity: i === scrubIdx ? 1 : 0.35 }}
              onClick={() => setScrubIdx(i)}
              title={`${r.regime} @ t=${r.t}s`}
            />
          ))}
        </div>
        <input
          type="range" min={0} max={regimeTimeline.length - 1} value={scrubIdx}
          onChange={(e) => setScrubIdx(+e.target.value)}
          className="regime-slider"
        />

        <hr />
        <div className="panel-title">EGT[1..4] — predicted ŷ vs actual y</div>
        {cylinders ? (
          <div className="egt-grid">
            {cylinders.map((c) => (
              <div key={c.cyl} className="egt-row">
                <div className="mono egt-label" style={{ color: c.warn ? 'var(--critical)' : 'var(--ink-000)' }}>
                  cyl {c.cyl}  z=<Field label={`EGT[${c.cyl}] z-score`} field={`ResidualFrame.z[EGT${c.cyl}]`} equation="z = (y − ŷ) / σ" value={c.z.toFixed(2)}>{c.z >= 0 ? '+' : ''}{c.z.toFixed(2)}</Field>
                  {c.warn && liveFrame.resolved && <span style={{ marginLeft: 6 }}>⚠ above G1.4 band</span>}
                </div>
                <ResponsiveContainer width="100%" height={44}>
                  <LineChart data={c.series} margin={{ top: 2, right: 4, bottom: 2, left: 4 }}>
                    <XAxis dataKey="t" hide domain={[0, 39]} type="number" />
                    <YAxis hide domain={['dataMin - 5', 'dataMax + 5']} />
                    <Tooltip contentStyle={{ background: 'var(--ink-800)', border: '1px solid var(--line-500)', fontSize: 11 }} />
                    <Line type="monotone" dataKey="predicted" stroke="var(--ink-400)" strokeDasharray="3 3" dot={false} strokeWidth={1.2} isAnimationActive={false} />
                    <Line type="monotone" dataKey="actual" stroke={c.warn ? 'var(--critical)' : 'var(--nominal)'} dot={false} strokeWidth={1.5} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        ) : (
          <div className="mono" style={{ fontSize: 12, color: 'var(--caution)', padding: '8px 0' }}>
            EGT[1..4] suppressed this frame — input dropout, per the residual-suppression rule (no imputation).
          </div>
        )}

        <hr />
        <div className="panel-title">spatial decomposition</div>
        <div className="mono spatial-readout">
          <Field label="alpha (mean EGT shift)" field="ResidualFrame.spatial.alpha" equation="dT = α·1 + β·g + s" value={spatialDecomposition.alpha}>α={spatialDecomposition.alpha}</Field>{'  '}
          <Field label="beta (gradient loading)" field="ResidualFrame.spatial.beta" equation="dT = α·1 + β·g + s" value={spatialDecomposition.beta}>β={spatialDecomposition.beta}</Field>{'  '}
          <Field label="residual infinity-norm" field="ResidualFrame.spatial.sInf" equation="‖s‖∞ = max_i |s_i|" value={spatialDecomposition.sInf}>‖s‖∞={spatialDecomposition.sInf}</Field>{'  '}
          @cyl{spatialDecomposition.argmax}
        </div>

        <hr />
        <div className="panel-title">regime encounter</div>
        <div className="encounter-bars">
          {regimeEncounters.map((r) => (
            <div key={r.regime} className="encounter-bar-row">
              <span className="mono encounter-label">{r.regime}</span>
              <div className="encounter-bar-track">
                <div className="encounter-bar-fill" style={{ width: `${Math.min(100, r.seconds / 4)}%` }} />
              </div>
              <span className="mono">{r.seconds}s</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
