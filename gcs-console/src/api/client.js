// Real backend client -- calls the actual Stage3Orchestrator via the
// FastAPI app in src/replan_to_learn/api/app.py (see scripts/run_api.sh,
// launch.json's "api" configuration). Distinct from mock/: everything
// here returns genuinely computed physics/gate/RUL/mission output, not
// scripted animation data.
//
// NOTE on latency: a single POST /telemetry call is currently ~10-60+
// real seconds (gate.update()'s regime-jacobian computation, and, when
// the gate is AMBIGUOUS, probe_selector's candidate-manoeuvre physics
// simulation -- both real, profiled, unresolved performance costs, not
// network latency). Callers MUST show a loading state; this is not
// something to poll at animation frame rates.

// http by default (zero-friction for a live demo). TLS is opt-in on the
// backend (USE_TLS=1, see scripts/run_api.sh/.bat) -- if you enable it
// there, also set VITE_API_BASE=https://localhost:8000 (e.g. in
// gcs-console/.env.local) before starting the console, and expect a
// one-time browser cert-warning click-through on https://localhost:8000
// (self-signed cert; no way to skip that from code).
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

async function apiFetch(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${options?.method || 'GET'} ${path} -> ${res.status}: ${text}`);
  }
  return res.json();
}

export function checkHealth() {
  return apiFetch('/health');
}

export function postTelemetry(frame) {
  return apiFetch('/telemetry', { method: 'POST', body: JSON.stringify(frame) });
}

export function getState(engineId) {
  return apiFetch(`/engine/${encodeURIComponent(engineId)}/state`);
}

// A real, physically-plausible telemetry frame -- same nominal cruise
// values used throughout the backend's own tests
// (tests/gate/test_gate_naming_accuracy.py's REGIME_CENTERS[3]).
export function buildSampleFrame(engineId, overrides = {}) {
  return {
    t: 0.0,
    egt: [950.0, 955.0, 960.0, 965.0],
    cht: 360.0,
    p_oil: 3.5e5,
    t_oil: 350.0,
    n_rpm: 2000.0,
    mdot_f: 0.05,
    map_pa: 90000.0,
    tps: 0.5,
    t_im: 295.0,
    p_amb: 84700.0,
    t_amb: 278.15,
    v_tas: 55.0,
    h_p: 1500.0,
    engine_id: engineId,
    ...overrides,
  };
}
