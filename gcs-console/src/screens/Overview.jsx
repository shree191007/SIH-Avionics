import { useState } from 'react';
import Tile from '../components/Tile';
import Modal from '../components/Modal';
import VerdictBadge from '../components/VerdictBadge';
import ReasoningChain from '../components/ReasoningChain';
import { useScenario } from '../context/ScenarioContext';

import EngineHealth from './EngineHealth';
import Diagnosis from './Diagnosis';
import Prognostics from './Prognostics';
import Mission from './Mission';
import Fleet from './Fleet';
import Safety from './Safety';
import Blueprint from './Blueprint';

import { lifeBudget, fleetShape, safetyHeatCalendar } from '../mock/data';

const RISK_COLOR = { LOW: 'var(--nominal)', MEDIUM: 'var(--caution)', HIGH: 'var(--critical)' };

// Everything except the 3D Blueprint still expands into a full-detail
// modal — the Blueprint is the one exception: it's the centerpiece now,
// always live in the middle of the page, not tucked behind a click.
const TILES = [
  { id: 'health', num: '1', title: 'Engine Health', Screen: EngineHealth },
  { id: 'diagnosis', num: '2', title: 'Diagnosis & Evidence', Screen: Diagnosis },
  { id: 'prognostics', num: '3', title: 'Prognostics (RUL)', Screen: Prognostics },
  { id: 'mission', num: '4', title: 'Mission & Probe', Screen: Mission },
  { id: 'fleet', num: '5', title: 'Fleet', Screen: Fleet },
  { id: 'safety', num: '7', title: 'Safety Validation', Screen: Safety },
];

export default function Overview() {
  const [open, setOpen] = useState(null); // tile id or null
  const { liveFrame } = useScenario();

  const activeTile = TILES.find((t) => t.id === open);
  const failCount = safetyHeatCalendar.reduce((sum, r) => sum + r.cells.filter((c) => c === 'fail').length, 0);
  const totalChecks = safetyHeatCalendar.reduce((sum, r) => sum + r.cells.length, 0);

  return (
    <div className="overview-grid">
      {/* left — sensor values + RUL + mission risk + safety, one tall panel */}
      <Tile area="health" num="1 · 3 · 4 · 7" title="Engine Health, Prognostics, Mission & Safety">
        <div className="tile-subsection">
          <div className="tile-subheader">
            <span className="mono">1 · sensors</span>
            <button className="tile-expand" onClick={() => setOpen('health')}>expand ⤢</button>
          </div>
          <VerdictBadge verdict={liveFrame.health.status} />
          {liveFrame.health.cylZ ? (
            <div className="overview-mini-rows mono" style={{ marginTop: 10 }}>
              {liveFrame.health.cylZ.map((z, i) => (
                <div key={i} className="overview-mini-row" style={{ color: liveFrame.health.warnCyl === i + 1 ? 'var(--critical)' : 'var(--ink-400)' }}>
                  cyl {i + 1}  z={z >= 0 ? '+' : ''}{z.toFixed(2)} {liveFrame.health.warnCyl === i + 1 && '⚠'}
                </div>
              ))}
            </div>
          ) : (
            <div className="mono" style={{ marginTop: 10, fontSize: 11, color: 'var(--caution)' }}>EGT[1..4]: suppressed</div>
          )}
          <div className="mono" style={{ marginTop: 10, fontSize: 10.5, color: 'var(--ink-400)' }}>{liveFrame.health.note}</div>
        </div>

        <hr />

        <div className="tile-subsection">
          <div className="tile-subheader">
            <span className="mono">3 · prognostics (RUL)</span>
            <button className="tile-expand" onClick={() => setOpen('prognostics')}>expand ⤢</button>
          </div>
          <div className="mono" style={{ fontSize: 12, marginBottom: 8 }}>
            <span style={{ color: 'var(--critical)' }}>p05 {liveFrame.rul.p05}h</span> · {liveFrame.rul.median}h · <span style={{ color: 'var(--nominal)' }}>p95 {liveFrame.rul.p95}h</span>
          </div>
          <div className="rul-bar-track">
            <div className="rul-bar-band" style={{ left: `${(liveFrame.rul.p05 / 320) * 100}%`, width: `${((liveFrame.rul.p95 - liveFrame.rul.p05) / 320) * 100}%` }} />
            <div className="rul-bar-marker" style={{ left: `${(liveFrame.rul.median / 320) * 100}%` }} />
          </div>
          <VerdictBadge verdict={liveFrame.rul.status} />
        </div>

        <hr />

        <div className="tile-subsection">
          <div className="tile-subheader">
            <span className="mono">4 · mission & probe</span>
            <button className="tile-expand" onClick={() => setOpen('mission')}>expand ⤢</button>
          </div>
          <div className="mono" style={{ fontSize: 12 }}>
            risk: <span style={{ color: RISK_COLOR[liveFrame.mission.risk.sortie] }}>{liveFrame.mission.risk.sortie}</span> (sortie) · <span style={{ color: RISK_COLOR[liveFrame.mission.risk.endurance] }}>{liveFrame.mission.risk.endurance}</span> (endurance)
          </div>
          <div className="mono" style={{ fontSize: 10.5, color: 'var(--ink-400)', marginTop: 4 }}>{liveFrame.mission.note}</div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-400)', marginTop: 8 }}>
            life budget: {lifeBudget.spentHours}h / {lifeBudget.sessionCapHours}h
          </div>
          <div className="encounter-bar-track" style={{ marginTop: 6 }}>
            <div className="encounter-bar-fill" style={{ width: `${(lifeBudget.spentHours / lifeBudget.sessionCapHours) * 100}%`, background: 'var(--probe-action)' }} />
          </div>
        </div>

        <hr />

        <div className="tile-subsection">
          <div className="tile-subheader">
            <span className="mono">7 · safety validation</span>
            <button className="tile-expand" onClick={() => setOpen('safety')}>expand ⤢</button>
          </div>
          <div className="mono" style={{ fontSize: 18, color: failCount ? 'var(--caution)' : 'var(--nominal)' }}>
            {(((totalChecks - failCount) / totalChecks) * 100).toFixed(2)}%
          </div>
          <div className="mono" style={{ fontSize: 10, color: liveFrame.safetyCategory ? 'var(--borrowed)' : 'var(--ink-400)' }}>
            {liveFrame.safetyCategory ? `edge case: "${liveFrame.safetyCategory}"` : liveFrame.resolved ? 'nominal case' : 'evaluating…'}
          </div>
        </div>
      </Tile>

      {/* center — the 3D model, always live */}
      <Tile area="blueprint" className="tile-hero" num="9" title="Engine Fault Blueprint">
        <Blueprint />
      </Tile>

      {/* right — reasoning chain */}
      <Tile area="diagnosis" num="2" title="Diagnosis & Evidence" onExpand={() => setOpen('diagnosis')}>
        <VerdictBadge verdict={liveFrame.verdict} suffix={` — ${liveFrame.verdictSuffix}`} />
        {liveFrame.cos !== null && (
          <div className="mono" style={{ marginTop: 6, fontSize: 11, color: 'var(--ink-400)' }}>|cos| = {liveFrame.cos}</div>
        )}
        <div style={{ marginTop: 10 }}>
          <ReasoningChain steps={liveFrame.chain.filter((s) => s.revealed)} />
          {!liveFrame.resolved && (
            <div className="chain-pending">
              <span className="chain-pending-dot" />
              watching for the next signal…
            </div>
          )}
        </div>
      </Tile>

      <Tile area="fleet" num="5" title="Fleet" onExpand={() => setOpen('fleet')}>
        <div className="mono" style={{ fontSize: 12 }}>epoch: {fleetShape.epoch} · contributors: {fleetShape.contributors.length}</div>
        {liveFrame.fleet.attempted ? (
          <div className="mono" style={{ fontSize: 12, marginTop: 8 }}>
            R² = {liveFrame.fleet.r2} <span style={{ color: liveFrame.fleet.admissible ? 'var(--nominal)' : 'var(--critical)' }}>{liveFrame.fleet.admissible ? 'admissible ✓' : 'rejected ✗'}</span>
          </div>
        ) : (
          <div className="mono" style={{ fontSize: 11, marginTop: 8, color: 'var(--ink-400)' }}>{liveFrame.fleet.note}</div>
        )}
      </Tile>

      {activeTile && (
        <Modal title={`${activeTile.num} · ${activeTile.title}`} onClose={() => setOpen(null)}>
          <activeTile.Screen />
        </Modal>
      )}
    </div>
  );
}
