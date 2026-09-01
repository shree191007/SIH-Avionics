// Turns a static scenario definition (mock/scenarios.js) into a live,
// time-varying frame — the console now watches a fault develop instead
// of snapping to its final state on a button press. No RNG here: this
// only ever numerically interpolates already-fixed endpoints, so the
// same scenario always develops identically (no jitter).

export function easeInQuad(t) { return t * t; }
export function lerp(a, b, t) { return a + (b - a) * t; }

// A comfortably healthy baseline the RUL band eases away from as the
// fault accumulates evidence — not any real aircraft's number, just a
// wide, low-urgency starting point for the animation.
export const NOMINAL_RUL = { p05: 150, median: 220, p95: 300 };

const RISK_LEVEL = { LOW: 1, MEDIUM: 2, HIGH: 3 };
const RISK_LABEL = { 1: 'LOW', 2: 'MEDIUM', 3: 'HIGH' };

export function buildLiveFrame(scenario, progress) {
  const p = Math.min(1, Math.max(0, progress));
  const ease = easeInQuad(p);
  const resolved = p >= 1;
  const n = scenario.chain.length;

  const chain = scenario.chain.map((step, i) => ({ ...step, revealed: p >= (i + 1) / n }));
  const gateRevealed = chain.find((s) => s.id === 'gate')?.revealed ?? false;
  const fleetStep = chain.find((s) => s.id === 'fleet');
  const fleetRevealed = fleetStep ? fleetStep.revealed : resolved;

  const health = {
    status: p === 0 ? 'HEALTHY' : resolved ? scenario.health.status : 'DEGRADING',
    cylZ: scenario.health.cylZ ? scenario.health.cylZ.map((z) => +(z * ease).toFixed(2)) : (p > 0 ? null : [0, 0, 0, 0]),
    warnCyl: scenario.health.warnCyl,
    note: resolved ? scenario.health.note : p > 0 ? 'evaluating residuals…' : 'nominal',
  };

  const rul = {
    p05: +lerp(NOMINAL_RUL.p05, scenario.rul.p05, ease).toFixed(1),
    median: +lerp(NOMINAL_RUL.median, scenario.rul.median, ease).toFixed(1),
    p95: +lerp(NOMINAL_RUL.p95, scenario.rul.p95, ease).toFixed(1),
    status: resolved ? scenario.rul.status : 'TRACKING',
    note: resolved ? scenario.rul.note : 'updating with each new frame…',
  };

  const finalSortie = RISK_LEVEL[scenario.mission.risk.sortie];
  const finalEndurance = RISK_LEVEL[scenario.mission.risk.endurance];
  const mission = {
    risk: {
      sortie: RISK_LABEL[Math.round(lerp(1, finalSortie, ease))],
      endurance: RISK_LABEL[Math.round(lerp(1, finalEndurance, ease))],
    },
    note: resolved ? scenario.mission.note : 'assessing risk as evidence accumulates…',
    probeNeeded: resolved ? scenario.mission.probeNeeded : false,
  };

  const fleet = fleetRevealed
    ? scenario.fleet
    : {
        attempted: false, r2: null, admissible: null,
        note: scenario.fleet.attempted ? 'attempting fleet borrow…' : 'not required — locally identifiable',
      };

  return {
    id: scenario.id,
    label: scenario.label,
    blueprintKey: scenario.blueprintKey,
    progress: p,
    resolved,
    verdict: resolved ? scenario.verdict : p > 0 ? 'MONITORING' : 'HEALTHY',
    verdictSuffix: resolved ? scenario.verdictSuffix : 'accumulating evidence',
    cos: gateRevealed ? scenario.cos : null,
    named: resolved ? scenario.named : undefined,
    ambiguous_set: resolved ? scenario.ambiguous_set : undefined,
    health,
    rul,
    mission,
    fleet,
    chain,
    safetyCategory: resolved ? scenario.safetyCategory : null,
  };
}
