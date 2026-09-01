# Stage 1 --- Foundation, Data Contracts & Existing-Solution Baseline

**SIH26054 · Replan to Learn · Modified implementation plan**

## Purpose

This stage does not replace the existing solution. It freezes its
interfaces, provenance, datasets, configuration, test fixtures, and
acceptance criteria so the existing L1--L7 implementation can be
extended without breaking scientific assumptions.

The existing Stage 1 already defines the Physics Twin, telemetry, health
parameters, regimes, residual generation, calibration, fault injection,
and the `ResidualFrame` contract. Preserve those definitions as the
baseline. fileciteturn1file2

## What changes

The previous three-stage implementation becomes a five-stage
implementation:

``` text
Stage 1  Foundation + contracts + data
Stage 2  Physics Twin + residual generation
Stage 3  State estimation + identifiability + fleet
Stage 4  ML + RUL + mission/probe
Stage 5  GCS + integration + validation + deployment
```

## 1. Freeze the existing contracts

### Telemetry

Retain the six primary channels:

-   EGT\[1..4\]
-   CHT
-   oil pressure
-   oil temperature
-   engine speed
-   fuel flow

Retain exogenous inputs:

-   MAP
-   TPS
-   intake temperature
-   ambient pressure
-   ambient temperature
-   TAS
-   pressure altitude

The existing document explicitly states that exogenous inputs are model
drivers rather than health signals and that missing inputs must cause
affected residuals to be suppressed rather than computed from guessed
values. Preserve this behavior. fileciteturn1file2

### Health parameters

Retain:

``` text
θ_vol
θ_comb
θ_cool
θ_inj[1..4]
θ_oilp
θ_fric
sensor biases
```

The cooling/combustion pair remains the deliberately ambiguous pair.

### Regime contract

Retain `REGIME_GRID_V1` and version it permanently. A regime-grid change
invalidates stored fleet fingerprints.

### Fault injection

Retain the existing injected-fault harness, including non-uniform
cooling degradation. Do not simplify cooling into a uniform fault
because that would artificially improve the spatial diagnostic result.
fileciteturn1file2

## 2. Data versioning

Every dataset and derived artifact receives:

``` text
dataset_id
flight_id
aircraft_id
engine_id
model_version
regime_grid_version
calibration_manifest_hash
schema_version
```

Train/test splits are by aircraft and flight, never by individual
samples.

## 3. Artifact registry

``` text
artifacts/
├── raw/
├── processed/
├── calibration/
├── models/
├── fingerprints/
├── replay/
└── evaluation/
```

Each artifact must be reproducible from its manifest.

## 4. Existing-solution audit

Before modifying code, map the existing implementation:

  Existing component      Action
  ----------------------- -----------------------------------------
  PhysicsTwin             Preserve and refactor only where needed
  Residual generation     Preserve contract
  UKF                     Preserve estimator semantics
  Identifiability gate    Preserve central refusal behavior
  Fleet federation        Preserve transfer guardrails
  GBT/ML                  Preserve as primary model
  Particle RUL            Preserve
  Mission/probe planner   Preserve
  GCS                     Refactor around final contracts
  Tests                   Expand rather than discard

The original solution is already unusually specific about acceptance
thresholds, interfaces, fleet guardrails, and evaluation arms. The goal
is to integrate those pieces, not throw them away.

## 5. Stage 1 exit criteria

-   Contracts frozen.
-   Existing code mapped.
-   Dataset manifests created.
-   Model/calibration provenance defined.
-   Versioning enforced.
-   Five-stage dependency graph accepted.
-   No scientific algorithm is changed merely for architectural
    cleanliness.
