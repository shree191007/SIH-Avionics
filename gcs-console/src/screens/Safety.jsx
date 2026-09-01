import SafetyHeatCalendar from '../components/SafetyHeatCalendar';
import { useScenario } from '../context/ScenarioContext';
import { safetyHeatCalendar } from '../mock/data';

const STATUS = [
  { name: 'sensor dropout gate', ok: true },
  { name: 'ambiguous refusal never over-confident', ok: true },
  { name: 'fleet borrow admissibility guardrail', ok: true },
  { name: 'probe cost never exceeds session cap', ok: true },
];

export default function Safety() {
  const { liveFrame } = useScenario();

  return (
    <div className="screen">
      <div className="panel">
        <div className="panel-title" style={{ margin: 0 }}>Safety Validation</div>
        <div className="status-list mono">
          {STATUS.map((s) => (
            <div key={s.name} className="status-row">
              <span style={{ color: s.ok ? 'var(--nominal)' : 'var(--critical)' }}>{s.ok ? '✓' : '✗'}</span>
              {s.name}
            </div>
          ))}
        </div>

        <hr />
        <div className="panel-title">heat-calendar (10⁴ checks, by scenario)</div>
        {liveFrame.safetyCategory ? (
          <div className="mono" style={{ fontSize: 11, color: 'var(--borrowed)', marginBottom: 6 }}>
            current fault is a tested edge case — highlighted below: “{liveFrame.safetyCategory}”
          </div>
        ) : (
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-400)', marginBottom: 6 }}>
            {liveFrame.resolved
              ? 'current fault is a nominal identifiable case, not one of the randomized-check edge categories below.'
              : 'evaluating whether this frame lands in one of the edge-case categories below…'}
          </div>
        )}
        <SafetyHeatCalendar rows={safetyHeatCalendar} highlight={liveFrame.safetyCategory} />
      </div>
    </div>
  );
}
