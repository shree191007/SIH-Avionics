# Replan to Learn

**SIH26054 — Adaptive Hybrid Twin Engine Health Management** for the Rotax
915iS. A physics-twin + gate + ML + RUL + mission-planner pipeline that
diagnoses developing engine faults from real telemetry, refuses to guess
when the evidence is genuinely ambiguous, and recommends real mission
replans — surfaced through a GCS operator console.

The project is built in five stages, each with its own design doc at the
repo root (`01_...md` – `05_...md`). This README is the map; those docs are
the spec each stage was built against.

## Why this exists

Conventional engine monitoring alerts on raw sensor thresholds (EGT too
hot, oil pressure too low). That catches gross failures late and says
nothing about *which* subsystem is actually degrading. This project
instead maintains a calibrated physics twin of the engine, compares its
predictions against real telemetry, and uses Fisher-information-based
identifiability analysis to answer three questions honestly:

1. **Is something wrong?** — physics-residual anomaly detection, which
   catches deviations raw thresholds miss entirely (see
   `scripts/demo_1_early_detection.py`, run against a real NTSB accident
   flight).
2. **Do we actually know what?** — the gate either NAMES a specific
   subsystem or honestly reports AMBIGUOUS, never guesses. When
   ambiguous, it tries to borrow evidence from the fleet, or proposes a
   real in-flight probe manoeuvre to resolve it.
3. **What should the pilot do about it?** — physics-derived remaining
   useful life (RUL) and a mission risk assessment that can recommend a
   real replan (derate, altitude change, return).

## Architecture

```
TelemetryFrame (real sensors)
      │
      ▼
PhysicsTwin              L1-L2   real ODE-integrated engine model,
      │                          residual = telemetry - prediction
      ▼
IdentifiabilityGate      L3-L4   Fisher-information separability test:
      │                          NAMED / AMBIGUOUS / BORROWED / INVALID
      ├──► FleetNode             cross-aircraft fingerprint borrowing
      │
      ▼
MLClassifier              L5     gate-masked GBT fault classifier
      │                          (never overrides the gate's verdict)
      ▼
RULParticleFilter          L6    physics-derived failure-threshold
      │                          Monte Carlo remaining-useful-life
      ▼
MissionSimulator/          L7    mission risk assessment + probe-
ProbeSelector                    manoeuvre selection (value-of-information)
      │
      ▼
GCSPanelBuilder                  operator-facing panel (engine health,
      │                          diagnosis, prognostics, mission, fleet)
      ▼
FastAPI (api/app.py)  ◄────────  gcs-console (React/Vite operator UI)
```

Orchestrated end-to-end by `Stage3Orchestrator.process_residual_frame()`
(`src/replan_to_learn/stage3.py`) — every layer above is a real,
independently-testable module, not a mock.

## Repo layout

```
src/replan_to_learn/
  contracts/      telemetry/residual data contracts, regime grid
  physics_twin/    calibrated ODE engine model (L1-L2)
  estimation/      UKF state estimation
  gate/            identifiability gate, Fisher information (L3-L4)
  fleet/           cross-aircraft fingerprint fusion (L3)
  ml/              gate-masked GBT fault classifier (L5)
  rul/             particle-filter remaining useful life (L6)
  planner/         mission simulator, probe selector (L7)
  gcs/             operator-panel data assembly
  api/             FastAPI orchestration layer
  replay/          deterministic end-to-end replay with provenance
  eval/            six-arm evaluation harness
  provenance/, registry/, reproducibility/   data/model lineage tracking

gcs-console/       React/Vite operator console (real backend + scripted demo mode)
model/             calibrated parameters (thermal, gaspath, noise, RUL thresholds, trained classifier)
scripts/           demo scripts, safety validation, calibration fitting, dev-server launchers
tests/             pytest suite, mirrors src/ layout
0X_*.md            per-stage design specs (the actual project requirements)
```

`data/` (real NGAFID/NTSB flight telemetry, ~26GB) is intentionally **not**
committed — see [Data](#data) below.

## Setup

```bash
python3 -m pip install -e ".[dev]"
cd gcs-console && npm install
```

Requires Python ≥3.10. The project is not currently PyPI-packaged beyond
local editable install (`pyproject.toml` at the repo root).

## Running things

**Tests:**
```bash
pytest tests/ -q --ignore=tests/real_data   # real_data needs data/ (see below)
```

**The four required demonstrations** (spec `05_gcs_integration_validation_deployment.md`
§6 — each runs against the real pipeline, no mocking):
```bash
python3 scripts/demo_1_early_detection.py       # real NTSB accident flight replay
python3 scripts/demo_2_honest_refusal_probe.py  # AMBIGUOUS -> probe -> new evidence -> diagnosis
python3 scripts/demo_3_fleet_borrow.py          # admissible borrow, then a rejected mismatch
python3 scripts/demo_4_mission_save.py          # healthy vs. degraded engine changes the mission call
```

**Safety validation** (spec §7 — randomized edge-case validation: sensor
dropout, missing exogenous inputs, transient operation, version
mismatch, simultaneous/ambiguous faults, corrupted or absent fleet
data):
```bash
python3 scripts/safety_validation.py
```

**API + GCS console**, using `.claude/launch.json`'s dev-server configs
(or run the two commands directly):
```bash
bash scripts/run_api.sh          # FastAPI on :8000
npm --prefix gcs-console run dev # Vite on :5173
```
Open the console, and press the **LIVE** button to send a real telemetry
frame through the actual backend (the scripted "1–6" scenario buttons are
a separate, pre-authored demo mode that needs no backend). A live call
genuinely takes tens of seconds — see [Known limitations](#known-limitations).

## Data

Real data lives in `data/` (NGAFID cross-fleet telemetry, and the NTSB
WPR22LA211 Rotax 915iS accident fixture used by Demo 1) and is not
committed here — it's large (~26GB) and not this project's to redistribute
wholesale. `tests/real_data/` documents provenance and loading; regenerate
or source it separately before running anything that touches real flight
data end-to-end (`tests/real_data/`, `scripts/fit_phase_*.py`,
`scripts/demo_1_early_detection.py`).

`model/` (the *calibrated outputs* fit from that data — thermal/gaspath
parameters, noise model, RUL failure thresholds, the trained ML
classifier) **is** committed, since the pipeline needs it to run and it's
small.

## Known limitations

Documented honestly rather than silently papered over:

- **Two safety-validation scenarios are real, un-implemented gaps**, not
  just untested: `model_saturation` is never actually triggered
  (`PhysicsTwin._compute_residuals` hardcodes `is_saturated=False`
  regardless of real state-clamp events), and fleet-epoch staleness is
  tracked (`FleetShape.epoch`) but never checked in
  `gate.try_borrow()`. Both need a real threshold/design decision, not a
  guess — see `scripts/safety_validation.py`'s own report output.
- **`probe_selector.select_probe()` is slow** (tens of seconds per call)
  when the gate is AMBIGUOUS — it's genuinely simulating ~24 candidate
  manoeuvres through the physics twin, not a bug. Partial parallelization
  (`concurrent.futures.ProcessPoolExecutor`) is in place for the
  per-candidate cost and regime-Jacobian computation
  (`planner/probe_selector.py`); an unresolved ~60s cost remains
  unaccounted for by that fix and needs a `cProfile` pass to actually
  locate before claiming the parallelization solved it.
- **`ProbeSelectionResult` has no `to_dict()`**, so `/probe/select`'s
  metadata field in `api/app.py` is silently always `None`.
- The six-arm evaluation harness (`eval/`) has been fixed for a real bug
  (synthetic fault cases weren't injecting any fault signal at all — see
  `eval/harness.py`'s `injected_theta`) but has not been run end-to-end
  against real NGAFID data.

## License

Not yet decided — treat as all-rights-reserved until a LICENSE file is added.
