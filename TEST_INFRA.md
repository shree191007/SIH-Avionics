# E2E Test Infra: SIH26054 Phase 1

## Test Philosophy
- Opaque-box, requirement-driven derived strictly from `ORIGINAL_REQUEST.md`.
- Zero dependency on private implementation internals — tests interact purely via public Python APIs, data contracts, and manifest definitions.
- Methodology: 4-Tier Systematic Testing (Category-Partition, Boundary Value Analysis, Pairwise Combinatorial, Real-World Application Workloads).

## Feature Inventory & Test Target Mapping
| # | Feature | Requirement | Tier 1 (Min ≥5) | Tier 2 (Min ≥5) | Tier 3 (Pairwise) | Tier 4 (Workloads) |
|---|---------|-------------|:---------------:|:---------------:|:-----------------:|:------------------:|
| F1 | 6 Primary Telemetry Channels Contract | R1 | 5 | 5 | ✓ | ✓ |
| F2 | 7 Exogenous Inputs Contract | R1 | 5 | 5 | ✓ | ✓ |
| F3 | Programmatic Residual Suppression (No Imputation) | R1 | 5 | 5 | ✓ | ✓ |
| F4 | Health Parameters & Ambiguity Pair | R1 | 5 | 5 | ✓ | ✓ |
| F5 | Regime Contract (REGIME_GRID_V1) | R1 | 5 | 5 | ✓ | ✓ |
| F6 | Non-Uniform Cooling Fault Harness | R1 | 5 | 5 | ✓ | ✓ |
| F7 | 8-Field Provenance Manifest Schema | R2 | 5 | 5 | ✓ | ✓ |
| F8 | Deterministic SHA256 Data & Manifest Hashing | R2 | 5 | 5 | ✓ | ✓ |
| F9 | Grouped Anti-Leakage Splitters | R2 | 5 | 5 | ✓ | ✓ |
| F10 | Artifact Registry Directory Hierarchy | R3 | 5 | 5 | ✓ | ✓ |
| F11 | Manifest Loaders & Atomic Persistence | R3 | 5 | 5 | ✓ | ✓ |
| F12 | Reproducibility Engine & Lineage DAG | R3 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Test Runner**: Pytest (`pytest -v tests/`)
- **Coverage Target**: 100% acceptance criteria validated across all 4 tiers
- **Test Structure**:
  - `tests/e2e/test_tier1_features.py`: Happy-path feature verification across all 12 features (≥60 tests).
  - `tests/e2e/test_tier2_boundaries.py`: Boundary, corner, clamp, and corrupt/missing data conditions (≥60 tests).
  - `tests/e2e/test_tier3_combinations.py`: Cross-feature interaction tests (e.g. residual suppression + manifest generation + grouped splitting + artifact save/load) (≥12 tests).
  - `tests/e2e/test_tier4_applications.py`: End-to-end real-world flight scenarios (NTSB accident replay, healthy baseline, multi-aircraft fleet splitting, full artifact lineage roundtrip) (≥6 tests).
  - `tests/e2e/test_tier5_adversarial.py`: White-box adversarial coverage hardening and stress testing (dynamic generation).

## Acceptance Criteria Verification Checklist
- [x] Programmatic schema validator ensures missing exogenous telemetry fields trigger residual suppression rather than value imputation.
- [x] Fault injection harness reproduces non-uniform cooling degradation without simplifying to uniform spatial faults.
- [x] Metadata manifest generator produces valid manifests containing all 8 required provenance fields for raw and processed datasets.
- [x] Unit test suite verifies train/test splitter strictly prevents sample leakage between flights or aircraft.
- [x] Automated test suite verifies artifact registry creation, path resolution, and manifest reproducibility.
