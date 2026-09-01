import { useScenario } from '../context/ScenarioContext';

const VERDICT_COLOR = {
  NAMED: 'var(--critical)',
  AMBIGUOUS: 'var(--ambiguous)',
  DEGRADED_INPUT: 'var(--caution)',
};

function fmtT(ms) {
  const s = (ms / 1000).toFixed(1).padStart(4, '0');
  return `T+${s}s`;
}

// Replaces Judge Mode: press 1-6 (or click) to start simulating a fault
// developing live — Diagnosis, the reasoning chain, Engine Health, RUL,
// Mission risk, and the 3D Blueprint all reveal evidence together as the
// simulated frame count advances, instead of snapping to a final state.
export default function ScenarioBar({ onOpenLive }) {
  const { scenarios, scenario, liveFrame, progress, playing, durationMs, setActiveId, togglePlay, restart } = useScenario();
  return (
    <div className="scenario-bar">
      <span className="scenario-bar-label mono">fault scenario</span>
      <div className="scenario-bar-buttons">
        {scenarios.map((s) => (
          <button
            key={s.id}
            className={`scenario-bar-btn ${scenario.id === s.id ? 'active' : ''}`}
            onClick={() => setActiveId(s.id)}
            title={`hotkey ${s.hotkey}`}
          >
            <span className="mono scenario-bar-key">{s.hotkey}</span>
            <span className="scenario-bar-dot" style={{ background: VERDICT_COLOR[s.verdict] }} />
            {s.label}
          </button>
        ))}
        <button className="scenario-bar-btn scenario-bar-live" onClick={onOpenLive} title="call the real backend">
          <span className="scenario-bar-dot" style={{ background: 'var(--nominal)' }} />
          LIVE
        </button>
      </div>

      <div className="scenario-sim-controls">
        <button onClick={togglePlay} title="spacebar">{playing ? '‖' : '▶'}</button>
        <button onClick={restart} title="replay from t=0">↻</button>
        <div className="scenario-sim-track">
          <div className="scenario-sim-fill" style={{ width: `${progress * 100}%`, background: liveFrame.resolved ? 'var(--nominal)' : 'var(--caution)' }} />
        </div>
        <span className="mono scenario-sim-time">{fmtT(progress * durationMs)}</span>
      </div>
    </div>
  );
}
