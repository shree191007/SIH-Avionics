import { useState, useMemo, useEffect } from 'react';
import RegimeLedger from '../components/RegimeLedger';
import Field from '../components/Field';
import { useScenario } from '../context/ScenarioContext';
import { fleetShape, regimeLedgerCool, guardrailCandidates } from '../mock/data';

export default function Fleet() {
  const { scenario, liveFrame } = useScenario();
  const [selected, setSelected] = useState(guardrailCandidates[0].fault);

  // Follow the active scenario's own fault into the guardrail preview,
  // when it's one of the tracked candidates — otherwise leave the
  // picker where it is rather than forcing a mismatched selection.
  useEffect(() => {
    const target = scenario.named ?? scenario.ambiguous_set?.[0];
    if (target && guardrailCandidates.some((c) => c.fault === target)) setSelected(target);
  }, [scenario]);

  const candidate = useMemo(() => guardrailCandidates.find((c) => c.fault === selected), [selected]);
  const maxWeight = Math.max(...fleetShape.contributors.map((c) => c.weight));

  return (
    <div className="screen">
      <div className="panel">
        <div className="panel-header mono">
          <span>epoch: {fleetShape.epoch}</span>
          <span>contributors: {fleetShape.contributors.length}</span>
        </div>

        <div className="panel-title">constellation</div>
        <svg width="100%" height="70" viewBox="0 0 400 70">
          {fleetShape.contributors.map((c, i) => {
            const x = 20 + i * (360 / fleetShape.contributors.length);
            const r = 3 + (c.weight / maxWeight) * 10;
            return (
              <circle key={c.id} cx={x} cy={35} r={r} fill="var(--borrowed)" opacity={0.4 + c.weight * 0.5}>
                <title>{`${c.id}: weight ${c.weight}`}</title>
              </circle>
            );
          })}
        </svg>

        <hr />
        <div className="panel-title">Regime Ledger — this aircraft</div>
        <RegimeLedger entries={regimeLedgerCool} />
        <div className="mono" style={{ marginTop: 8 }}>
          current scenario fleet check:{' '}
          {liveFrame.fleet.attempted ? (
            <>
              <Field label="R² (try_borrow)" field="IdentifiabilityGate.try_borrow(fault).r2" equation="R² of fleet-shape prediction vs local samples" value={liveFrame.fleet.r2}>
                R² = {liveFrame.fleet.r2}
              </Field>{' '}
              <span style={{ color: liveFrame.fleet.admissible ? 'var(--nominal)' : 'var(--critical)' }}>
                {liveFrame.fleet.admissible ? 'admissible ✓' : 'rejected ✗'}
              </span>
            </>
          ) : (
            <span style={{ color: 'var(--ink-400)' }}>{liveFrame.fleet.note}</span>
          )}
        </div>

        <hr />
        <div className="panel-title">guardrail stress preview — try a different fault</div>
        <div className="alt-toggle">
          {guardrailCandidates.map((c) => (
            <button key={c.fault} className={selected === c.fault ? 'active' : ''} onClick={() => setSelected(c.fault)}>{c.fault}</button>
          ))}
        </div>
        <div className="mono">
          <Field label="R² (try_borrow)" field="IdentifiabilityGate.try_borrow(fault).r2" equation="recomputed live from IdentifiabilityGate.try_borrow()" value={candidate.r2}>
            R² = {candidate.r2}
          </Field>{' '}
          <span style={{ color: candidate.admissible ? 'var(--nominal)' : 'var(--critical)' }}>
            {candidate.admissible ? 'admissible ✓' : 'rejected ✗'}
          </span>
        </div>
      </div>
    </div>
  );
}
