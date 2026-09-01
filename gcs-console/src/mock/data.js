// Deterministic mock data — stands in for the live telemetry / estimator
// stream (ResidualFrame, AttributionFrame, particle-filter, planner, fleet).
// No Math.random() jitter on render: values are generated once per module
// load from a seeded PRNG, matching "tabular numerals that don't jitter."

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.cross ? seed : seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(915);

export const REGIMES = [
  'CLIMB·LOW', 'CLIMB·MID', 'CLIMB·HIGH',
  'CRUISE·LOW', 'CRUISE·MID', 'CRUISE·HIGH',
  'DESCENT·LOW', 'DESCENT·MID', 'DESCENT·HIGH',
  'IDLE', 'TRANSIENT', 'INVALID',
];

export const THETA_PARAMS = [
  'theta_vol', 'theta_comb', 'theta_cool',
  'theta_inj[1]', 'theta_inj[2]', 'theta_inj[3]', 'theta_inj[4]',
  'theta_oilp', 'theta_fric',
];

// ---------- Engine Health / ResidualFrame ----------
export const regimeTimeline = Array.from({ length: 60 }, (_, i) => {
  let regime = 'CRUISE·MID';
  if (i < 8) regime = 'CLIMB·LOW';
  else if (i < 14) regime = 'CLIMB·MID';
  else if (i < 46) regime = 'CRUISE·MID';
  else if (i < 52) regime = 'CRUISE·HIGH';
  else regime = 'DESCENT·MID';
  return { t: i * 10, regime };
});

// Builds a synthetic predicted-vs-actual EGT trace for one cylinder,
// shaped so its final offset matches the given z-score — used by
// EngineHealth to render a chart consistent with whichever fault
// scenario is active, not just the one hardcoded example.
export function buildEgtSeries(z) {
  return Array.from({ length: 40 }, (_, i) => {
    const predicted = 620 + Math.sin(i / 6) * 12;
    const ramp = 0.3 + 0.7 * (i / 39); // a developing fault grows in, doesn't jump
    const actual = predicted + z * 15 * ramp + (rand() - 0.5) * 4;
    return { t: i, predicted: +predicted.toFixed(1), actual: +actual.toFixed(1) };
  });
}

export const spatialDecomposition = { alpha: 0.12, beta: 0.71, sInf: 0.31, argmax: 3 };

export const regimeEncounters = REGIMES.slice(0, 9).map((r) => ({
  regime: r,
  seconds: r === 'CRUISE·MID' ? 340 : r === 'CLIMB·MID' ? 12 : Math.round(rand() * 60),
}));

// ---------- Diagnosis / AttributionFrame ----------
// The active verdict + reasoning chain now live in mock/scenarios.js,
// driven by ScenarioContext (hotkeys 1-6) instead of being static here.

// upper-triangle cosine matrix across THETA_PARAMS, quantized
export const cosMatrix = (() => {
  const n = THETA_PARAMS.length;
  const m = Array.from({ length: n }, () => Array(n).fill(null));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let v = rand() * 0.5;
      if ((THETA_PARAMS[i] === 'theta_cool' && THETA_PARAMS[j] === 'theta_comb')) v = 0.97;
      m[i][j] = +v.toFixed(2);
    }
  }
  return m;
})();

export const crlbTrail = Array.from({ length: 40 }, (_, i) => ({
  t: i,
  width: +(0.04 + i * 0.0035 + (i > 30 ? (i - 30) * 0.01 : 0)).toFixed(3),
}));

export const resolvedByLog = [
  { time: '09:14', text: 'NAMED θ_inj[2]', basis: 'PROBE 4min climb' },
  { time: '09:41', text: 'BORROWED θ_cool', basis: 'FLEET R²=0.94' },
  { time: '10:02', text: 'NAMED θ_fric', basis: 'LOCAL' },
];

export const regimeLedgerCool = REGIMES.slice(0, 9).map((r, i) => ({
  regime: r,
  status: i < 3 ? 'flown' : i < 6 ? 'borrowed' : 'never',
}));

// ---------- Prognostics / RUL ----------
// The active scenario now supplies {p05, median, p95, status, note} —
// see mock/scenarios.js. This builds the "last 6h" funnel replay so it
// narrows into whichever endpoint the active scenario reports, instead
// of always animating toward one hardcoded example.
export function buildRulFunnel(p05, median, p95) {
  const [p05Start, medianStart, p95Start] = [p05 * 1.43, median * 1.33, p95 * 1.27];
  return Array.from({ length: 24 }, (_, i) => {
    const shrink = i / 24;
    return {
      t: i,
      p05: +(p05Start - shrink * (p05Start - p05)).toFixed(1),
      median: +(medianStart - shrink * (medianStart - median)).toFixed(1),
      p95: +(p95Start - shrink * (p95Start - p95)).toFixed(1),
    };
  });
}
export const failLimit = 20;

// ---------- Mission & Probe ----------
// risk + note now come from the active scenario's `mission` field
// (mock/scenarios.js) — the alternatives/probe pool below stay static,
// they're the pre-simulated option set, not fault-specific.
export const missionAlternatives = {
  nominal: { risk: 'HIGH', endurance: '4.2h', note: 'baseline plan, no derate' },
  derate: { risk: 'MEDIUM', endurance: '4.6h', note: 'N capped at 92%, +0.4h endurance' },
  altitude: { risk: 'MEDIUM', endurance: '4.4h', note: 'FL80 cruise, lower EGT stress' },
  return: { risk: 'LOW', endurance: '1.1h', note: 'RTB now, mission aborted' },
};
export const lifeBudget = { spentHours: 0.6, probesUsed: 2, sessionCapHours: 2.0 };

export const probeCandidates = Array.from({ length: 60 }, (_, i) => {
  const G = +(rand() * 0.9 + 0.05).toFixed(2);
  const C = +(rand() * 0.9 + 0.05).toFixed(2);
  const feasible = C < 0.75 && G > 0.15;
  return { id: `P${i + 1}`, G, C, feasible };
});
probeCandidates[0] = { id: 'P1', G: 0.41, C: 0.20, feasible: true, best: true };

export const probeRanked = [
  { id: 'P1', name: '4min climb to FL60', G: 0.41, C: 0.20, value: 0.21 },
  { id: 'P7', name: 'level cruise hold 3min', G: 0.29, C: 0.18, value: 0.11 },
  { id: 'P12', name: 'partial-throttle step', G: 0.22, C: 0.24, value: -0.02 },
];

// ---------- Fleet ----------
export const fleetShape = {
  epoch: 214,
  contributors: Array.from({ length: 11 }, (_, i) => ({
    id: `AC-${100 + i}`,
    weight: +(0.3 + rand() * 0.7).toFixed(2),
  })),
  r2: 0.94,
  admissible: true,
};

export const guardrailCandidates = [
  { fault: 'theta_cool', r2: 0.94, admissible: true },
  { fault: 'theta_comb', r2: 0.41, admissible: false },
  { fault: 'theta_inj[3]', r2: 0.88, admissible: true },
  { fault: 'theta_fric', r2: 0.35, admissible: false },
];

// ---------- Safety Validation ----------
export const safetyScenarios = ['sensor dropout', 'model saturation', 'ambiguous fault', 'fleet reject', 'probe abort'];
export const safetyHeatCalendar = safetyScenarios.map((scenario) => ({
  scenario,
  cells: Array.from({ length: 40 }, () => (rand() > 0.985 ? 'fail' : 'pass')),
}));

export const provenance = {
  dataset: 'ngafid_915is_v3',
  model: '2.4.1',
  regime: 'V1',
  latencyMs: 84,
};
