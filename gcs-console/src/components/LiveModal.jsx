import { useState } from 'react';
import Modal from './Modal';
import { postTelemetry, buildSampleFrame } from '../api/client';

// Genuinely live: calls the real FastAPI backend (src/replan_to_learn/api/app.py),
// which runs the actual Stage3Orchestrator (physics twin + L4 gate + L6 RUL +
// L7 planner). Deliberately separate from the scripted Demo scenarios in
// mock/ -- this shows real, one-shot computed output, not a pre-authored
// 12s animation, and it can take a real 10-60+ seconds per call (see
// api/client.js's latency note).
export default function LiveModal({ onClose }) {
  const [status, setStatus] = useState('idle'); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedS, setElapsedS] = useState(0);

  async function run() {
    setStatus('loading');
    setError(null);
    setResult(null);
    const t0 = performance.now();
    const tick = setInterval(() => setElapsedS((performance.now() - t0) / 1000), 200);
    try {
      const frame = buildSampleFrame('LIVE-DEMO');
      const data = await postTelemetry(frame);
      setResult(data);
      setStatus('done');
    } catch (e) {
      setError(String(e.message || e));
      setStatus('error');
    } finally {
      clearInterval(tick);
    }
  }

  return (
    <Modal title="LIVE — real backend call" onClose={onClose}>
      <div className="live-modal">
        <p className="live-modal-intro">
          Sends one real telemetry frame to <code>POST /telemetry</code> and shows the
          actual <code>Stage3Orchestrator</code> output — real physics-twin residuals, a
          real gate verdict, real particle-filter RUL, a real mission-risk assessment.
          Not scripted. Requires the <code>api</code> dev server running (
          <code>scripts/run_api.sh</code>, port 8000).
        </p>

        {status === 'idle' && (
          <button className="live-modal-run" onClick={run}>Send real telemetry frame →</button>
        )}

        {status === 'loading' && (
          <div className="live-modal-loading">
            <div className="live-modal-spinner" />
            <div>computing… {elapsedS.toFixed(1)}s</div>
            <div className="live-modal-loading-note">
              real gate/probe-selector physics simulation — this genuinely takes tens of
              seconds at this calibration, not a network delay.
            </div>
          </div>
        )}

        {status === 'error' && (
          <div className="live-modal-error">
            <div>Request failed: {error}</div>
            <button onClick={run}>retry</button>
          </div>
        )}

        {status === 'done' && result && (
          <div className="live-modal-result">
            <button className="live-modal-run" onClick={run}>run again</button>
            <div className="live-modal-grid">
              <section>
                <h4>engine_health</h4>
                <pre>{JSON.stringify(result.engine_health, null, 2)}</pre>
              </section>
              <section>
                <h4>diagnosis</h4>
                <pre>{JSON.stringify(result.diagnosis, null, 2)}</pre>
              </section>
              <section>
                <h4>rul</h4>
                <pre>{JSON.stringify(result.rul, null, 2)}</pre>
              </section>
              <section>
                <h4>mission</h4>
                <pre>{JSON.stringify(result.mission, null, 2)}</pre>
              </section>
              <section>
                <h4>fleet</h4>
                <pre>{JSON.stringify(result.fleet, null, 2)}</pre>
              </section>
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
}
