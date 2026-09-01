# Original User Request

## Initial Request — 2026-08-26T15:29:06Z

Build Phase 1 (Foundation, Data Contracts & Existing-Solution Baseline) for SIH26054 "Replan to Learn" in Python using Polars / PyArrow for high-performance data contracts, schema validation, and artifact management.

Working directory: /Users/vatxn1907__/Desktop/V1
Integrity mode: development

## Requirements

### R1. Foundation Data Contracts & Schemas
Define and freeze data contracts in Python (using Polars / PyArrow / dataclasses):
- 6 primary telemetry channels: EGT[1..4], CHT, oil pressure, oil temperature, engine speed, fuel flow.
- Exogenous inputs: MAP, TPS, intake temperature, ambient pressure, ambient temperature, TAS, pressure altitude.
- Explicit contract rule: Exogenous inputs are model drivers, NOT health signals. Missing exogenous inputs MUST suppress affected residuals rather than computing from guessed/imputed values.
- Health parameters schema: \theta_{vol}, \theta_{comb}, \theta_{cool}, \theta_{inj[1..4]}, \theta_{oilp}, \theta_{fric}, and sensor biases. Maintain cooling/combustion (\theta_{cool}, \theta_{comb}) as an ambiguous pair.
- Regime contract: REGIME_GRID_V1 grid definition, versioned permanently.
- Injected-fault harness: Support non-uniform cooling degradation baseline.

### R2. Data Versioning & Provenance Metadata
Implement dataset and derived artifact provenance tracking. Every dataset/artifact manifest must include:
`dataset_id`, `flight_id`, `aircraft_id`, `engine_id`, `model_version`, `regime_grid_version`, `calibration_manifest_hash`, `schema_version`.
Enforce train/test splits strictly grouped by aircraft and flight (never by individual samples).

### R3. Artifact Registry Structure & Reproducibility
Establish directory hierarchy and manifest loaders under `artifacts/`:
- `artifacts/raw/`, `artifacts/processed/`, `artifacts/calibration/`, `artifacts/models/`, `artifacts/fingerprints/`, `artifacts/replay/`, `artifacts/evaluation/`
- Every artifact must be fully reproducible from its metadata manifest.

## Acceptance Criteria

### Data Contracts & Verification
- [ ] Programmatic schema validator ensures missing exogenous telemetry fields trigger residual suppression rather than value imputation.
- [ ] Fault injection harness reproduces non-uniform cooling degradation without simplifying to uniform spatial faults.
- [ ] Metadata manifest generator produces valid manifests containing all 8 required provenance fields for raw and processed datasets.
- [ ] Unit test suite verifies train/test splitter strictly prevents sample leakage between flights or aircraft.
- [ ] Automated test suite verifies artifact registry creation, path resolution, and manifest reproducibility.

## Follow-up — 2026-08-26T16:56:35Z

Build Stage 2 (Physics Twin & Residual Generation L1-L2) for SIH26054 "Replan to Learn" in Python: implement mean-value engine model, 2-node + 4 cylinder thermal network, regime classifier (`REGIME_GRID_V1`), whitened & normalized residual generation (`ResidualFrame`), analytical/FD Jacobian matrix, calibration pipeline using NGAFID flight dataset under `data/`, and fault injection harness.

Working directory: /Users/vatxn1907__/Desktop/V1
Integrity mode: demo

## Requirements

### R1. Mean-Value Engine Model & Thermal Network (L1)
Implement continuous-time lumped-parameter engine model (`PhysicsTwin`):
- State vector $x = [p_{im}, T_{hd}, T_{oil}, T_{exh}, \omega, T_{cyl[1..4]}]$ using RK4 / semi-implicit integration.
- Gas path (Rotax 915 iS A 1352 cm³), Wiebe-free efficiency surface, friction (Chen-Flynn FMEP), EGT per cylinder.
- Thermal network: 2 main nodes + 4 cylinder nodes with fixed cylinder asymmetry factor $a_i$.
- Health parameters $\theta$: $\theta_{vol}, \theta_{comb}, \theta_{cool}, \theta_{inj[1..4]}, \theta_{oilp}, \theta_{fric}$, sensor biases. Keep $(\theta_{cool}, \theta_{comb})$ ambiguous.
- Implement `predict(u, theta)` and `jacobian(u, theta)`.

### R2. Regime Classification & Quasi-Steady Detector
Implement `REGIME_GRID_V1` classifier (12 fingerprint bins: 3 flight phases $\times$ 3 power bands + Idle/Transient/Invalid):
- Enforce quasi-steady stability checks ($|dN/dt| < 30 \text{ rpm/s}$, $|dp_{im}/dt| < 2 \text{ kPa/s}$, $|dT_{hd}/dt| < 0.15 \text{ K/s}$ for $\ge 20 \text{ s}$).
- Track regime encounter counter $n_i(r)$ per aircraft/regime.

### R3. Residual Generation & Spatial Decomposition (L2)
Implement raw residual calculation $e_t$, AR(1) whitening filter, and regime-conditioned normalization to produce normalized residual vector $z_t$ (9 channels: $z_{EGT[1..4]}, z_{CHT}, z_{p_{oil}}, z_{T_{oil}}, z_{\dot{m}_f}, z_{N\text{(power balance)}}$).
Project $z_{EGT}$ onto geometric basis $(\alpha_t \mathbf{1} + \beta_t \mathbf{g} + \mathbf{s}_t)$ emitting spatial components $(\alpha, \beta, \|s\|_\infty, \text{argmax}_i |s_i|)$ in `ResidualFrame`.

### R4. Calibration Procedure & Fault Injection Harness
Implement 3-phase calibration workflow using real NGAFID data located in `data/`:
- Phase A: gas path/power vs Rotax curves.
- Phase B: thermal/oil vs NGAFID flight data.
- Phase C: regime-conditioned noise model $\sigma_{r_t}$.
Implement fault injection harness `inject(theta_nom, fault, t)` supporting non-uniform cooling degradation.

## Acceptance Criteria

### Verification & Quality Gates
- [ ] Gate G1.1: Gas path fit $\le 3\%$ power, $\le 4\%$ BSFC vs Rotax curves.
- [ ] Gate G1.2: Thermal fit CHT RMSE $\le 6^\circ\text{C}$, oil-temp RMSE $\le 4^\circ\text{C}$, oil-pressure RMSE $\le 0.25\text{ bar}$ on held-out flight data.
- [ ] Gate G1.3: Residual whiteness Ljung-Box $p > 0.05$ at lag 20 across channels.
- [ ] Gate G1.4: Regime independence $\|mean(z)\| \le 0.25\sigma$ in all regime bins ($\le 0.4\sigma$ worst bin).
- [ ] Gate G1.5: Bit-identical $z_t$ determinism given same inputs and seed.
- [ ] Gate G1.6: Execution time $\le 8\text{ ms}$ per step.
- [ ] Jacobian vs finite differences relative error $\le 10^{-6}$.
