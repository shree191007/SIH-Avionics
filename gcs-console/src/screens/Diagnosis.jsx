import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import VerdictBadge from '../components/VerdictBadge';
import FingerprintCompass from '../components/FingerprintCompass';
import CosineHeatmap from '../components/CosineHeatmap';
import RegimeLedger from '../components/RegimeLedger';
import ReasoningChain from '../components/ReasoningChain';
import Field from '../components/Field';
import { useScenario } from '../context/ScenarioContext';
import { cosMatrix, THETA_PARAMS, crlbTrail, resolvedByLog, regimeLedgerCool } from '../mock/data';

export default function Diagnosis() {
  const [chainOpen, setChainOpen] = useState(true);
  const { scenario, liveFrame } = useScenario();
  const revealed = liveFrame.chain.filter((s) => s.revealed);
  const hasFleetStep = scenario.chain.some((s) => s.id === 'fleet');

  return (
    <div className="screen">
      <div className="panel">
        <div className="panel-header">
          <VerdictBadge verdict={liveFrame.verdict} suffix={` — ${liveFrame.verdictSuffix}`} />
          {liveFrame.cos !== null && (
            <Field label="|cos|" field="AttributionFrame.reason.cos" equation="|cos(θ_a, θ_b)|" value={liveFrame.cos}>
              <span className="mono">|cos| = {liveFrame.cos}</span>
            </Field>
          )}
          <button onClick={() => setChainOpen((v) => !v)}>
            {chainOpen ? 'hide' : 'show'} reasoning chain
          </button>
        </div>

        {chainOpen && (
          <>
            <div className="panel-title">how it got here — physics → estimator → gate{hasFleetStep ? ' → fleet' : ''} → verdict</div>
            <ReasoningChain steps={revealed} />
            {!liveFrame.resolved && (
              <div className="chain-pending">
                <span className="chain-pending-dot" />
                watching for the next signal…
              </div>
            )}
            <hr />
          </>
        )}

        <div className="diagnosis-plots">
          <div>
            <div className="panel-title">Fingerprint Compass</div>
            <FingerprintCompass />
          </div>
          <div>
            <div className="panel-title">Cosine Matrix Heatmap — all pairs</div>
            <CosineHeatmap params={THETA_PARAMS} matrix={cosMatrix} />
          </div>
        </div>

        <hr />
        <div className="mono">fleet transfer: rejected — R² = 0.41</div>

        <div className="panel-title">CRLB trail (last 40 frames)</div>
        <ResponsiveContainer width="100%" height={50}>
          <LineChart data={crlbTrail}>
            <XAxis dataKey="t" hide />
            <YAxis hide />
            <Line type="monotone" dataKey="width" stroke="var(--critical)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="mono" style={{ color: 'var(--ink-400)', fontSize: 11 }}>widening</div>

        <hr />
        <div className="panel-title">resolved-by log</div>
        <div className="ledger-list mono">
          {resolvedByLog.map((r, i) => (
            <div key={i} className="ledger-row">
              <span>{r.time}</span>
              <span>{r.text}</span>
              <span style={{ color: 'var(--ink-400)' }}>via {r.basis}</span>
            </div>
          ))}
        </div>

        <hr />
        <div className="panel-title">Regime Ledger — θ_cool</div>
        <RegimeLedger entries={regimeLedgerCool} />
      </div>
    </div>
  );
}
