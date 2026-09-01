# Stage 5 --- GCS, End-to-End Integration, Validation & Deployment

**SIH26054 · Replan to Learn · Modified implementation plan**

## Purpose

The existing solution contains strong individual L1--L7 specifications,
but it needs a final integration stage that proves the complete system
behaves correctly as one product.

Stage 5 is where we connect the existing pieces, build the operational
GCS, run the six-arm evaluation, harden the interfaces, and prepare
deployment.

## 1. Final architecture

``` text
                    UAV / ENGINE
                         │
                    Telemetry
                         ↓
              ┌────────────────────┐
              │ Stage 1 Foundation │
              │ schemas + version  │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Stage 2 Physics    │
              │ Twin + Residuals   │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Stage 3 State +    │
              │ Gate + Fleet       │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Stage 4 ML + RUL + │
              │ Mission + Probe    │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │ Stage 5 GCS + API  │
              │ + Validation       │
              └────────────────────┘
```

## 2. GCS

The GCS must show reasoning, not merely a collection of gauges.

### Engine health

-   health summary
-   RPM
-   EGT\[1..4\]
-   CHT
-   oil pressure
-   oil temperature
-   fuel flow
-   vibration/misfire proxy
-   residual trends

### Diagnosis

For a named fault:

``` text
Primary issue
Confidence / calibrated probability
Gate verdict
Top supporting evidence
TreeSHAP features
Fingerprint separation
CRLB interval
```

For ambiguity:

``` text
PRIMARY ISSUE
AMBIGUOUS — cooling OR combustion

Why:
|cos| = ...
CRLB = ...
Fleet transfer:
accepted / rejected

Next action:
probe ...
```

Never render an ambiguous candidate as a definitive subsystem.

### Prognostics

Display:

-   p05/p50/p95 RUL
-   failure probability before mission completion
-   degradation trajectory
-   uncertainty

### Mission

Display:

-   mission risk
-   predicted operating margins
-   recommendation
-   alternative mission simulations

### Fleet

Display:

-   fleet epoch
-   contributor count
-   borrowed regimes
-   R²
-   transfer status

## 3. API contracts

Expose the final system through versioned endpoints:

``` text
POST /telemetry
POST /replay

GET /engine/{id}/state
GET /engine/{id}/health
GET /engine/{id}/evidence
GET /engine/{id}/diagnosis
GET /engine/{id}/rul

POST /mission/simulate
POST /probe/select

GET /fleet/{id}/fingerprint
```

The API contains orchestration only. Scientific logic remains in the
stage modules.

## 4. End-to-end replay

One command must be able to run:

``` text
dataset
 ↓
Stage 2
 ↓
Stage 3
 ↓
Stage 4
 ↓
GCS
 ↓
evaluation report
```

Replay must be deterministic and record:

-   dataset hash
-   model versions
-   configuration
-   random seeds
-   software version
-   result artifacts

## 5. Six-arm evaluation

Retain the existing evaluation arms:

  Arm   System
  ----- ------------------------
  A     Raw threshold baseline
  B     Pure ML
  C     Physics + ML, no gate
  D     Physics + ML + gate
  E     D + fleet
  F     E + probe

All arms use identical data splits and evaluation windows.

The decisive metric remains:

> over-confident misattribution rate

The system is successful only if the guarded architecture materially
reduces this error without achieving the reduction by simply refusing
every case.

The original solution explicitly requires refusal rate and resolution
rate to be reported alongside accuracy. Preserve that.
fileciteturn1file0

## 6. Required demonstrations

### Demo 1 --- Early detection

Replay a known maintenance-event flight.

Show:

``` text
threshold alert
vs
physics residual alert
```

and lead time.

### Demo 2 --- Honest refusal + probe

Use the deliberately ambiguous fault pair.

Show:

``` text
AMBIGUOUS
  ↓
probe candidates
  ↓
selected manoeuvre
  ↓
new evidence
  ↓
diagnosis
```

### Demo 3 --- Fleet borrow

Show:

``` text
local ambiguity
  ↓
admissible fleet shape
  ↓
BORROWED
```

Then show a mismatched fault where the transfer is rejected.

### Demo 4 --- Mission save

Show a mission where the current engine state changes the
recommendation:

``` text
nominal mission
      ↓
risk assessment
      ↓
derate / altitude change / return
```

## 7. Safety validation

Run at least:

-   10\^4 randomized mission feasibility checks for probe safety
-   sensor dropout
-   missing MAP
-   transient operation
-   model saturation
-   stale fleet epoch
-   version mismatch
-   unknown fault
-   ambiguous fault
-   simultaneous faults
-   corrupted fleet contribution
-   disconnected GCS/fleet link

Required behavior:

``` text
invalid input        → INVALID / DEGRADED_INPUT
model saturation     → MODEL_SATURATED
unknown anomaly      → ANOMALOUS_UNKNOWN
ambiguous diagnosis  → ABSTAIN / PROBE_REQUIRED
unsafe mission       → REPLAN / ABORT
```

## 8. Deployment architecture

### Level 1 --- Offline research

Historical data only.

### Level 2 --- Engine test rig

Live telemetry and real-time twin.

### Level 3 --- GCS shadow mode

Live inference without operational control.

### Level 4 --- Decision support

Operator receives recommendations.

### Level 5 --- Fleet deployment

Fleet fingerprint exchange and maintenance intelligence.

### Level 6 --- Edge-assisted

Move validated low-latency functions toward onboard SBC deployment.

## 9. Final repository

``` text
sih26054/
├── contracts/
├── data/
├── physics_twin/
├── residuals/
├── estimation/
├── identifiability/
├── fleet/
├── ml/
├── rul/
├── mission/
├── planner/
├── gcs/
├── api/
├── replay/
├── evaluation/
├── tests/
├── configs/
├── models/
└── docs/
```

## 10. Final acceptance gate

The project is complete only when:

1.  Stage 2 physics passes G1.
2.  Stage 3 estimator/gate/fleet passes G2.
3.  Stage 4 ML/RUL/planner passes G3.
4.  Stage 5 integration tests pass.
5.  End-to-end replay is deterministic.
6.  The six-arm comparison is reproducible.
7.  The GCS exposes the evidence chain.
8.  Ambiguous cases are visibly refused.
9.  Fleet transfer is visibly guarded.
10. Mission recommendations are traceable to simulation evidence.

## Final system behavior

``` text
What should the engine be doing?
            ↓
What is it actually doing?
            ↓
What hidden health state explains the difference?
            ↓
Can the available sensors distinguish the candidates?
       ┌──────────────┴──────────────┐
      YES                            NO
       ↓                              ↓
diagnose                     fleet borrow / probe
       ↓                              ↓
degradation                    new evidence
       ↓                              ↓
RUL                           re-estimate
       └──────────────┬───────────────┘
                      ↓
               mission simulation
                      ↓
               SAFE / REPLAN / ABORT
                      ↓
                    GCS
```

The existing solution is therefore **not discarded**. Its detailed
L1--L7 scientific work is preserved and reorganized into a cleaner
five-stage implementation pipeline, with Stage 1 providing the missing
foundation and Stage 5 providing the missing system-level integration,
validation, and deployment layer.
