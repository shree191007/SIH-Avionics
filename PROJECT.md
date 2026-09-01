# Project: SIH26054 Phase 1 — Foundation Data Contracts, Provenance & Artifact Registry

## Architecture Overview
Phase 1 establishes the rock-solid foundation, data contracts, versioning/provenance tracking, grouped anti-leakage splitting, and content-addressable artifact registry for the SIH26054 "Replan to Learn" architecture.

```
                     ┌─────────────────────────────────────────────────────────┐
                     │             TELEMETRY INGESTION & CONTRACTS             │
                     │  - 6 Primary Channels (9 scalar signals) [VERIFIED]     │
                     │  - 7 Exogenous Inputs (Model Drivers) [VERIFIED]        │
                     │  - Residual Suppression Engine (Zero Imputation) [VERIFIED]│
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │            HEALTH PARAMETERS & REGIME GRID              │
                     │  - Health Parameter Schema (\theta_vol..\theta_fric) [VERIFIED]
                     │  - Cooling/Combustion Ambiguous Pair Derivation [VERIFIED]
                     │  - REGIME_GRID_V1 (12 Core + 3 Non-Fingerprint Bins) [VERIFIED]
                     │  - Non-Uniform Cooling Degradation Fault Harness [VERIFIED]
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │            DATA VERSIONING & PROVENANCE (R2)            │
                     │  - 8-Field Manifest Schema (RFC 8785 Canonical JSON)    │
                     │  - Deterministic SHA256 Data Stream & Manifest Hashing  │
                     │  - Grouped Anti-Leakage Splitters (Aircraft & Flight)   │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │        ARTIFACT REGISTRY & REPRODUCIBILITY (R3)         │
                     │  - 7 Artifact Categories under artifacts/               │
                     │  - Atomic Multi-Process FileLock Persistence            │
                     │  - Manifest Loaders & Merkle Lineage DAG Traversal      │
                     │  - Deterministic Replay & Reproducibility Verifier      │
                     └─────────────────────────────────────────────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Primary Telemetry Channels Contract | 6 physical quantities across 9 scalar channels (EGT[1..4], CHT, p_oil, T_oil, N, mdot_f) with physical bounds and units in Polars, PyArrow, Dataclasses | M1 | R1 |
| F2 | Exogenous Inputs Contract | 7 exogenous channels (MAP, TPS, T_im, p_amb, T_amb, V_TAS, h_p) as boundary drivers | M1 | R1 |
| F3 | Programmatic Residual Suppression | Rule: Exogenous inputs are model drivers, not health signals. Missing inputs deterministically suppress affected residuals (NaN/masked) without imputation | M1 | R1 |
| F4 | Health Parameters Schema & Ambiguity Pair | $\theta_{vol}, \theta_{comb}, \theta_{cool}, \theta_{inj[1..4]}, \theta_{oilp}, \theta_{fric}$, sensor biases; physical proof & modeling of ($\theta_{cool}, \theta_{comb}$) ambiguity pair | M1 | R1 |
| F5 | Regime Contract (`REGIME_GRID_V1`) | 12-core + 3 operational regime bins, 20s quasi-steady stability criteria, encounter counters | M1 | R1 |
| F6 | Non-Uniform Cooling Fault Harness | Non-uniform cooling degradation baseline ($w \ge 0.35$) with spatial EGT decomposition ($\alpha_t \mathbf{1} + \beta_t \mathbf{g} + \mathbf{s}_t$) | M1 | R1 |
| F7 | 8-Field Provenance Manifest Schema | `dataset_id`, `flight_id`, `aircraft_id`, `engine_id`, `model_version`, `regime_grid_version`, `calibration_manifest_hash`, `schema_version` validation & generation | M2 | R2 |
| F8 | Deterministic Content Hashing | Canonical JSON (RFC 8785), Arrow RecordBatch stream hashing, self-hash computation, and Parquet metadata embedding | M2 | R2 |
| F9 | Grouped Anti-Leakage Splitters | `GroupedFlightSplitter`, `GroupedAircraftSplitter`, `HierarchicalGroupSplitter`, `TemporalFlightSplitter` strictly preventing sample leakage | M2 | R2 |
| F10 | Artifact Registry Hierarchy | 7 directories under `artifacts/` (`raw/`, `processed/`, `calibration/`, `models/`, `fingerprints/`, `replay/`, `evaluation/`) with path resolution | M3 | R3 |
| F11 | Manifest Loaders & Atomic Storage | `ArtifactRegistry`, `ArtifactManifest`, `ManifestLoader`, atomic write with `filelock`, zero-copy Polars/PyArrow loading | M3 | R3 |
| F12 | Reproducibility & Lineage Engine | Lineage DAG resolution (NetworkX), deterministic re-execution dry-runs, parameter/seed freezing, and hash verification | M3 | R3 |
| F13 | Full E2E Test Suite & Hardening | Complete Tiers 1-5 test coverage, adversarial stress tests, and validation of all acceptance criteria | M4 | E2E |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Foundation Data Contracts & Schemas | F1, F2, F3, F4, F5, F6 (Telemetry, Exogenous, Residual Suppression, Health, Regimes, Fault Harness) | None | DONE |
| M2 | Data Versioning & Provenance Metadata | F7, F8, F9 (8-Field Manifests, Deterministic SHA256, Grouped Anti-Leakage Splitters) | M1 | IN_PROGRESS |
| M3 | Artifact Registry Structure & Reproducibility | F10, F11, F12 (Hierarchy under `artifacts/`, Loaders, Atomic Persistence, Lineage DAG, Reproducibility Engine) | M1, M2 | PLANNED |
| M4 | Final Milestone: 100% E2E Pass & Adversarial Hardening | F13 (Tiers 1-4 E2E test pass, Tier 5 white-box adversarial stress testing & coverage hardening) | M1, M2, M3 | PLANNED |

## Interface Contracts

### M1 ↔ M2 Interface Contract
- `contracts/telemetry.py` exports `TelemetryFrame`, `TelemetrySchema` (Polars & PyArrow).
- `contracts/residuals.py` exports `ResidualFrame`, `ResidualSchema`, `ResidualSuppressionEngine`.
- `contracts/regimes.py` exports `REGIME_GRID_V1`, `RegimeClassifier`, `REGIME_GRID_VERSION = "REGIME_GRID_V1"`.
- `provenance/manifest.py` consumes schema version strings and regime grid versions to validate manifests.
- `provenance/splitters.py` accepts Polars DataFrames conforming to `TelemetrySchema` or `ResidualSchema` and returns partitioned indices/frames grouped by `aircraft_id` and `flight_id`.

### M2 ↔ M3 Interface Contract
- `provenance/manifest.py` exports `ProvenanceManifest`, `ProvenanceMetadata`, `compute_canonical_data_hash`, `compute_manifest_hash`.
- `registry/registry.py` uses `ProvenanceManifest` as the canonical manifest envelope for all stored artifacts.
- `registry/loaders.py` uses manifest hashes and embedded Parquet metadata to load and validate artifacts.
- `reproducibility/engine.py` traverses lineage hashes defined in `ProvenanceManifest` to verify reproducibility.

## Code Layout
```text
/Users/vatxn1907__/Desktop/V1/
├── pyproject.toml
├── src/
│   └── replan_to_learn/
│       ├── __init__.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   ├── telemetry.py
│       │   ├── residuals.py
│       │   ├── health.py
│       │   ├── regimes.py
│       │   └── faults.py
│       ├── provenance/
│       │   ├── __init__.py
│       │   ├── manifest.py
│       │   ├── hasher.py
│       │   └── splitters.py
│       ├── registry/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── paths.py
│       │   ├── loaders.py
│       │   └── atomic.py
│       ├── reproducibility/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   ├── lineage.py
│       │   └── verifier.py
│       └── models/
│           ├── __init__.py
│           ├── baseline.py
│           └── ukf.py
├── tests/
│   ├── conftest.py
│   ├── e2e/
│   │   ├── test_tier1_features.py
│   │   ├── test_tier2_boundaries.py
│   │   ├── test_tier3_combinations.py
│   │   └── test_tier4_applications.py
│   ├── test_contracts/
│   ├── test_provenance/
│   ├── test_registry/
│   └── test_reproducibility/
└── artifacts/
    ├── raw/
    ├── processed/
    ├── calibration/
    ├── models/
    ├── fingerprints/
    ├── replay/
    └── evaluation/
```
