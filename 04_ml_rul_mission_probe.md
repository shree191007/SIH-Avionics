# Stage 3 --- ML Layer, Particle-Filter RUL, Mission Planner & GCS (L5--L7)

**SIH26054 · Replan to Learn** · implementation spec, v1.0 **Owns:** L5
(ML layer), L6 (particle-filter RUL), L7 (mission planner + probe
selection), GCS **Consumes:** `AttributionFrame` from Stage 2 ·
**Produces:** operator recommendation + probe commands

L7's probe-scoring block is the contribution highlighted in Fig 3 of the
pitch. Everything else in this stage is deliberately established method,
chosen for auditability over sophistication.

------------------------------------------------------------------------

## 0. Definition of done

  -----------------------------------------------------------------------
  \#                      Gate                    Threshold
  ----------------------- ----------------------- -----------------------
  G3.1                    ML never contradicts    0 instances of L5
                          the gate                naming a subsystem L4
                                                  declared inseparable
                                                  --- enforced
                                                  structurally, not
                                                  statistically

  G3.2                    ML calibration          Expected Calibration
                                                  Error ≤ 0.05 on
                                                  held-out; reliability
                                                  diagram published

  G3.3                    RUL accuracy            α-λ accuracy ≥ 0.6 at λ
                                                  = 0.5 with α = 0.2 (PHM
                                                  Society convention)

  G3.4                    RUL honesty             realised failure time
                                                  inside the reported
                                                  \[5th, 95th\] interval
                                                  ≥ 90 % of runs

  G3.5                    Probe value             selected probe achieves
                                                  ≥ 80 % of the *best
                                                  available* separability
                                                  gain, at ≤ 120 % of the
                                                  best available cost

  G3.6                    Probe restraint         0 probes flown when an
                                                  admissible fleet borrow
                                                  existed (Stage 2 §4
                                                  ordering contract)

  G3.7                    Safety                  0 probe proposals
                                                  violating the flight
                                                  envelope, fuel reserve,
                                                  or return-margin
                                                  constraints, across 10⁴
                                                  randomised missions

  G3.8                    Edge budget             full L5+L6+L7 cycle ≤
                                                  400 ms at 0.1 Hz
                                                  planning rate; ≤ 150 MB
                                                  RSS total for all seven
                                                  levels
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 1. L5 --- the ML layer

### 1.1 Position in the architecture

L5 receives **nine clean residuals**, not a thousand raw numbers. That
is the entire reason it needs a fraction of the usual training data: the
physics has already removed the operating condition, so the model learns
fault signatures rather than re-learning how a climb works.

**Structural constraint (G3.1).** L5's output is masked by L4's verdict:

``` python
posterior = model.predict(features)
posterior = mask(posterior, gate.separable_classes())   # inseparable classes collapsed
if gate.verdict is AMBIGUOUS:
    report = AmbiguousReport(gate.ambiguous_set, posterior_over_set)
```

The ML layer can rank candidates *within* an ambiguous set --- that is
useful and honest --- but it can never break a tie the physics says is
unbreakable. A tie broken by a learned prior is precisely the "89 %
confidence" failure on page 1 of the pitch, relocated one level up.

### 1.2 Features

Computed from `ResidualFrame` and `AttributionFrame` over three windows
(60 s / 300 s / 1800 s):

  -----------------------------------------------------------------------
  Group                   Features                Count
  ----------------------- ----------------------- -----------------------
  Per-channel statistics  mean, robust slope      108
                          (Theil--Sen), IQR,      
                          max\|z\| --- 9 channels 
                          × 4 stats × 3 windows   

  Spatial                 `α`, `β`, `‖s‖_∞`,      15
                          `argmax_i s`, `β/α`     
                          ratio --- × 3 windows   

  Cross-modal             per-cylinder EGT/CHT    6
                          coherence, mean and min 

  Misfire proxy           cycle-to-cycle variance 4
                          of the 10 Hz RPM        
                          channel, and its trend  

  Regime context          fraction of window per  14
                          regime bin,             
                          regime-transition count 

  Estimator context       `θ̂`, `diag(P_θ)`,       25
                          innovation NIS          

  **Total**                                       **\~172**
  -----------------------------------------------------------------------

Feature computation is streaming with fixed-size ring buffers; no
re-scan of history.

### 1.3 Model

**Gradient-boosted decision trees** (LightGBM / XGBoost), depth ≤ 6, ≤
300 trees, exported to a dependency-free C inference stub for edge.

Chosen over a deep model for three reasons that should be stated in the
submission: it trains on the data volume we actually have;
per-prediction feature attribution is exact and cheap (TreeSHAP), which
a maintenance crew can be shown; and it costs almost nothing on the
target SBC. A 1D-CNN over the residual window is implemented as a
**comparison arm only** --- if it does not beat GBT by a margin that
survives the variance across seeds, GBT ships.

Classes: `HEALTHY`, `INJECTOR_i` (×4), `COOLING`, `COMBUSTION`,
`OIL_SYSTEM`, `SENSOR_EGT_i`, `UNKNOWN`.

`UNKNOWN` is a real class trained on held-out fault types the model
never saw --- a novelty-detection arm (one-class SVM on the healthy
residual manifold, thresholded on Mahalanobis distance). A monitoring
system that has no way to say "this is something I have not seen" will
confidently mislabel the one failure mode nobody anticipated.

### 1.4 Training data

  -----------------------------------------------------------------------
  Source                              Role
  ----------------------------------- -----------------------------------
  NGAFID healthy flights              negative class, and the
                                      noise/variability model

  Physics-injected faults (Stage 1    positive classes, swept over
  §7)                                 magnitude, onset rate, and regime
                                      exposure

  Fleet-heterogeneous variants        build gain `α_i` sampled per
                                      synthetic unit, so the model cannot
                                      learn a single unit's scale
  -----------------------------------------------------------------------

Split by **aircraft and by flight**, never by sample. A random sample
split leaks adjacent seconds of the same flight into train and test and
inflates every metric.

### 1.5 Calibration

Isotonic regression on a held-out calibration fold. Report ECE, Brier
score, and the reliability diagram. **G3.2.**

A confidence figure that does not survive a reliability diagram is
exactly the object this project criticises. Publishing ours is the
credibility move.

------------------------------------------------------------------------

## 2. L6 --- particle-filter RUL

### 2.1 Why a particle filter

The degradation state is 1--2 dimensional per fault, the failure
threshold is a non-linear function of the parameter and the operating
condition, and the output needed is a *first-passage time distribution*
--- not a point estimate. A particle filter gives that distribution
directly; a Kalman filter would give a mean and a variance and would
then need a Gaussian assumption on crossing time that is empirically
false when the drift rate is uncertain.

### 2.2 Degradation state

    d_k = [ θ_j,k , ρ_k ]          ρ = drift rate (per flight hour)
    θ_j,k+1 = θ_j,k + ρ_k · Δt_fh + w_θ
    ρ_k+1   = ρ_k · (1 − a_ρ) + a_ρ·ρ̄ + w_ρ        # mean-reverting, prevents runaway extrapolation

Optional non-linear form `ρ_k+1 = ρ_k · exp(γ·(1 − θ_j,k))` for
accelerating wear; selected per fault class by held-out likelihood, not
by preference.

### 2.3 Measurement update

The measurement is L3's posterior, not the raw residual:

    w_k^(p) ∝ w_{k−1}^(p) · N( θ̂_j,k ; θ_j,k^(p) , P_θ,jj,k )

-   `N_p = 500` particles (edge budget); systematic resampling when
    `ESS < N_p/2`.
-   Roughening after resample: add `N(0, ε·std)` with `ε = 0.05` to
    prevent particle collapse on long healthy stretches.
-   **Freeze policy:** no update while `verdict == AMBIGUOUS` for the
    parameter in question. Propagating an RUL for a fault you have
    refused to name is incoherent; the interval widens on its own, which
    is the correct behaviour and shows up in the GCS.

### 2.4 Failure threshold --- physics-derived, not assumed

The threshold is computed from the twin, not picked:

    θ_j^fail = min{ θ_j : any operating limit is violated somewhere in the
                    required mission envelope, on an ISA+20 day }

Limits evaluated: CHT/coolant limit at max continuous power in climb;
knock/misfire margin at the leanest commanded cruise mixture; minimum
oil pressure at hot idle; minimum power for required rate of climb at
maximum take-off weight.

This is why Fig 4 of the pitch can say "physics-derived misfire
threshold" rather than "threshold from failure data" --- there is no
failure data, and the twin supplies the limit.

### 2.5 Output

    RUL_hours = first-passage time of θ_j^(p) through θ_j^fail, per particle
    report: (p05, p50, p95), plus P(fail before end of current sortie)

An ETA of 4:15--4:35 is honesty, not vagueness; "4:23:47" is precisely
wrong every time.

### 2.6 Metrics

α-λ accuracy (α = 0.2, λ = 0.5), RMSE of `p50`, prognostic horizon at α
= 0.2, convergence rate, and **interval coverage** --- the fraction of
runs where truth falls in \[p05, p95\]. Coverage is the one that catches
a filter that is confidently wrong. **G3.3, G3.4.**

------------------------------------------------------------------------

## 3. L7 --- mission planner and probe selection

### 3.1 Two jobs

1.  **Nominal:** fly the sortie in simulation on today's degraded
    engine, using `PhysicsTwin.predict` with the current `θ̂`, and report
    mission risk and a power/altitude recommendation.
2.  **When the gate refuses:** find the cheapest manoeuvre that would
    separate the ambiguous candidates, price it, and decide whether to
    fly it. **This is the contribution.**

### 3.2 Ordering contract (restated, because it is a correctness requirement)

    gate.verdict == AMBIGUOUS
          │
          ├─ admissible fleet borrow exists (Stage 2 §3.6, rules A1–A6)?  ──► BORROW. No probe. Cost: zero.
          │
          └─ no admissible borrow  ──► probe scoring (§3.3–3.6)
                                            ├─ best probe passes value test?  ──► fly it
                                            └─ otherwise ──► report AMBIGUOUS to the operator, plainly

**G3.6** tests this. A system that flies a manoeuvre for information the
fleet already had is burning engine life for nothing.

### 3.3 Candidate manoeuvre set

Parameterised, not free-form --- a continuous trajectory optimisation is
not needed and would not be auditable.

  -----------------------------------------------------------------------
  Manoeuvre               Parameters              Regime bins excited
  ----------------------- ----------------------- -----------------------
  Step climb              Δh ∈ {500, 1000, 2000,  CLIMB × {MID, HIGH}
                          3000} m, IAS ∈ {80, 85, 
                          90} kt                  

  Power derate            ΔN ∈ {−100, −150, −250} CRUISE × LOW
                          rpm, dwell ∈ {60, 180,  
                          300} s                  

  Mixture enrichment      Δλ ∈ {−0.05, −0.10},    current phase, shifts
                          dwell ∈ {60, 120} s     EGT/CHT coupling

  Airspeed step           ΔIAS ∈ {±10, ±15} kt,   changes cooling airflow
                          dwell ∈ {120, 300} s    at constant power ---
                                                  the direct `θ_cool`
                                                  excitation

  Descent segment         ΔROD, dwell             DESCENT × {LOW, MID}
  -----------------------------------------------------------------------

Roughly 60 discrete candidates. Exhaustive evaluation is affordable; no
search heuristic needed, which keeps the selection explainable.

### 3.4 Predicted information gain

For candidate `m`, the predicted post-probe regime set is `R_i ∪ ΔR(m)`
with dwell times `τ_r(m)`. The predicted fingerprint entries come from
the twin's jacobian, with variance scaled by dwell:

    Var[ f̂|_r ] = σ_r² / τ_r(m)          (a 60 s dwell is worth less than a 300 s dwell)

Build the post-probe information matrix `F'(m)` exactly as in Stage 2
§2.4, using the inflated variances for predicted rows.

**Primary objective --- targeted separability.** The mission does not
need global identifiability; it needs *this* ambiguous pair resolved:

    G(m) = max over (j,k) ∈ ambiguous_set  [ |cos(f_j, f_k)| − |cos(f'_j(m), f'_k(m))| ]

**Secondary objective --- conditioning, used only to break ties:**

    G_D(m) = log det F'(m) − log det F        (D-optimality, restricted to the ambiguous subspace)

**Feasibility filter, applied before scoring:** flight envelope
(altitude, IAS, power limits at current weight and ISA deviation),
airspace/exposure constraints from the mission plan, remaining fuel with
reserve intact, and the **return-margin constraint** --- after the
probe, endurance to base must still exceed required time to base plus
reserve. A probe that resolves the diagnosis and strands the aircraft is
a failure, not a trade-off.

### 3.5 Cost model

    C(m) = w_L·ΔLife(m) + w_T·ΔTime(m) + w_F·ΔFuel(m) + w_E·ΔExposure(m)

-   **ΔLife** --- cumulative damage integrated over the probe
    trajectory: an Arrhenius thermal term on head/oil temperature plus a
    thermal-cycle term. Reported in engine-hours-equivalent so the
    operator sees "0.2 h of life" rather than a unitless penalty. This
    is the term that makes the trade-off real and it must come from the
    twin's simulated temperatures, not a lookup.
-   **ΔTime** --- added sortie time, in hours.
-   **ΔFuel** --- kg, and its knock-on effect on endurance.
-   **ΔExposure** --- mission-specific: time at altitudes or positions
    that raise detection or threat risk. Weight supplied by the mission
    plan, zero in a benign profile.

Weights `w_*` come from the mission profile, are shown in the GCS, and
are operator-editable. They are a policy input, not a tuned
hyperparameter, and should never be silently fitted.

### 3.6 Selection and the value test

    m* = argmax_{m feasible}  G(m) / ( C(m) + c_0 )

`c_0` regularises against zero-cost candidates dominating the ratio.

Fly `m*` only if it clears a **value-of-information** test:

    E[benefit] = P(misattribution | current gate state) × Cost(wrong maintenance action)
    fly  ⟺  E[benefit] > C(m*)  AND  G(m*) ≥ G_min

`Cost(wrong maintenance action)` --- the cost of stripping the wrong
subsystem --- is an operator input with a documented default. `G_min`
ensures a probe that only marginally improves separability is not flown
at all.

Post-probe: re-estimate (L3), re-gate (L4), and if the verdict is still
`AMBIGUOUS`, **do not automatically escalate to a second probe** ---
report to the operator with the achieved-vs-predicted separability. A
system that keeps spending engine life chasing a diagnosis it is not
getting is worse than one that says so.

### 3.7 Nominal mission risk

Simulate the planned sortie with `θ̂` and its uncertainty (sample from
`P_θ`, 100 draws):

    Risk = P( any operating limit violated | remaining sortie )

Reported as a band (LOW / MEDIUM / HIGH) with the underlying probability
available, for today's sortie and for the maximum-endurance sortie.
Recommendations (derate, altitude change, return) come from
re-simulating candidate mitigations and picking the one that meets the
mission with the largest limit margin.

------------------------------------------------------------------------

## 4. GCS --- operator view

### 4.1 Contract

  --------------------------------------------------------------------------
  Field                   Source                     Rendering rule
  ----------------------- -------------------------- -----------------------
  Engine health %         scalar summary of `θ̂` vs   integer, no decimals
                          admissible range           

  Status                  trend on health            HEALTHY / DEGRADING /
                                                     ACTION REQUIRED

  Primary issue           `AttributionFrame.named`   **If verdict is
                                                     AMBIGUOUS, render the
                                                     candidate set, not a
                                                     best guess**

  Attribution basis       verdict                    `LOCAL` / `PROBE 03:24`
                                                     / `FLEET R²=0.94` /
                                                     `MODEL-SHAPE`

  Life remaining          L6 `(p05, p95)`            always an interval,
                                                     never a point

  Mission risk            L7 §3.7                    per-sortie and
                                                     per-endurance bands

  Recommendation          L7                         concrete and
                                                     actionable: "Derate 150
                                                     rpm, return at 6 000
                                                     ft"

  Fleet epoch age         Stage 2 `FleetShape.epoch` shown when attribution
                                                     basis is FLEET
  --------------------------------------------------------------------------

### 4.2 The refusal state

Where the gate cannot isolate a subsystem, no probe is affordable and
the fleet shape does not fit, the panel says so plainly:

    PRIMARY ISSUE    AMBIGUOUS — cooling OR combustion
                     cannot separate at cruise (|cos| = 0.97)
                     probe available: 4 min climb, 0.2 h life — awaiting authorisation
                     fleet transfer rejected: R² = 0.41 (shape mismatch)

Never a confidence bar on a guess. This is a hard product requirement: a
maintenance crew sent to strip the wrong subsystem twice stops reading
the dashboard permanently, and at that point the entire system is
worthless regardless of its accuracy.

### 4.3 Explainability panel

For any `NAMED` verdict: TreeSHAP top-5 features from L5, the
fingerprint cosine values from L4, and --- when `BORROWED` --- which
regimes were borrowed, the R², and the fleet contributor count. An
engineer must be able to reconstruct why the system said what it said.

------------------------------------------------------------------------

## 5. Evaluation harness

### 5.1 Arms

  -----------------------------------------------------------------------
  Arm                                 Description
  ----------------------------------- -----------------------------------
  A · Threshold                       fixed limits on raw EGT/CHT/oil ---
                                      current practice

  B · Pure ML                         GBT on raw telemetry windows, no
                                      physics, no gate

  C · Physics + ML, no gate           our L1--L3 + L5, unguarded
                                      attribution (the "89 % confidence"
                                      baseline)

  D · + gate                          C plus L4

  E · + fleet                         D plus L4F

  F · + probe                         E plus L7 probe selection --- the
                                      full system
  -----------------------------------------------------------------------

All arms run on **identical data**, identical splits, identical seeds.

### 5.2 Metrics

  -----------------------------------------------------------------------
  Metric                              Definition
  ----------------------------------- -----------------------------------
  **Over-confident misattribution     fraction of cases where a subsystem
  rate**                              is named with stated confidence ≥
                                      0.8 and the named subsystem is not
                                      the faulted one. **Decisive.**

  Lead time                           hours between first alert and the
                                      maintenance event

  False-positive rate                 alerts per 100 healthy flight hours

  Precision / recall                  per fault class, macro-averaged

  Refusal rate                        fraction of cases reported
                                      `AMBIGUOUS` --- reported
                                      *alongside* accuracy, because a
                                      system that refuses everything is
                                      trivially never wrong

  Resolution rate                     fraction of refusals subsequently
                                      resolved, split by probe vs borrow

  Cost of resolution                  engine-hours-equivalent and minutes
                                      spent per resolution

  RUL error                           α-λ accuracy, `p50` RMSE, interval
                                      coverage

  Risk calibration                    reliability diagram on mission-risk
                                      predictions

  Latency, edge CPU, RSS              on the target SBC
  -----------------------------------------------------------------------

**Refusal rate and resolution rate must be reported together with
accuracy.** Reporting accuracy alone on a system whose contribution is
*refusing to answer* would be dishonest, and a competent judge will ask
for exactly this pair.

If arm F does not beat arm C on over-confident misattribution, we report
that. The claim is falsifiable by construction, and saying so is a
strength.

------------------------------------------------------------------------

## 6. Demonstration --- the four moments

Scripted, reproducible from a single command, ≤ 6 minutes total.

**1 · Lead time.** A real NGAFID flight that ended in a maintenance
event, replayed. Threshold and residual on one chart; the residual
crosses hours earlier. *Establishes L1--L2.*

**2 · The refusal, then the probe.** Two injected faults producing
near-identical cruise residuals. The system declines to name one --- the
panel shows the candidate set and `|cos|`. The planner scores 60
candidates on screen, picks the 4-minute climb, prices it at 0.2 h of
life, flies it; the signatures separate. *The contribution, visible in
sixty seconds.*

**3 · The borrow.** Same ambiguity, on an aircraft whose history
contains no climb at all. No probe is flown. The fleet shape completes
the fingerprint, R² = 0.94, the gate resolves --- zero fuel, zero engine
life, datalink down. Then the counter-demo: inject a *different* fault
on the same aircraft, R² collapses to 0.41, the transfer is refused, and
the system falls back to honest local ambiguity. **Show both.** The
refusal is what makes the borrow trustworthy.

**4 · The save.** 3.5 h to base against 6.0 h derated and 2.0 h at climb
power. Derate, turn back, aircraft returns.

------------------------------------------------------------------------

## 7. Risks

  --------------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- --------------------------
  ML learns the injection High                    Vary injection onset rate,
  artefact rather than                            magnitude, and build gain;
  the fault                                       hold out entire
                                                  fault-parameterisations;
                                                  test on a fault class
                                                  never seen in training via
                                                  the `UNKNOWN` arm

  Probe scoring is        Medium                  Rank-1 update of `F` per
  expensive at 60                                 added regime row rather
  candidates × FIM                                than full rebuild;
  rebuild                                         measured against G3.8

  Probe                   Medium                  Log both every time; if
  predicted-vs-achieved                           systematically optimistic,
  separability diverges                           the jacobian is wrong and
                                                  it is a Stage 1 defect,
                                                  not a planner one

  Operator ignores an     High (product)          Pair every refusal with a
  `AMBIGUOUS` panel                               concrete next action ---
                                                  the priced probe, or the
                                                  specific reason a borrow
                                                  was refused

  Cost weights tuned to   High (integrity)        Weights are operator
  make probes look cheap                          policy inputs, shown in
                                                  the GCS, never fitted;
                                                  state this in the
                                                  submission

  RUL extrapolates a      Medium                  Mean-reverting `ρ` (§2.2);
  drift rate seen only in                         freeze on `AMBIGUOUS`;
  cruise                                          interval widens rather
                                                  than the point estimate
                                                  drifting
  --------------------------------------------------------------------------

------------------------------------------------------------------------

## 8. Deliverables

1.  `ml/` --- feature builder, GBT training and calibration pipeline, C
    inference stub, TreeSHAP export.
2.  `rul/` --- particle filter (§2), physics-derived threshold solver,
    PHM metric suite.
3.  `planner/` --- mission simulator, candidate generator,
    information-gain scorer, cost model, value-of-information test,
    feasibility filter.
4.  `gcs/` --- operator panel implementing §4, including the refusal
    state.
5.  `eval/` --- six-arm harness, metric suite, and the reliability/ROC
    figures.
6.  The four-moment demo script, runnable end to end from one command.

# Modification applied in the five-stage plan

The original L5--L7 specification is retained as the scientific source
of truth for ML, particle-filter RUL, mission planning, probe selection,
and operator recommendation. It is now implemented as **Stage 4**.

The important architectural addition is that Stage 4 receives only the
Stage 3 `AttributionFrame` and does not reach backward into raw
telemetry.

## Stage 4 implementation order

### 4.1 ML

1.  Build the \~172-feature streaming feature vector.
2.  Train GBT baseline.
3.  Train 1D-CNN comparison arm.
4.  Calibrate the selected model with isotonic regression.
5.  Enforce structural masking by the Stage 3 gate.
6.  Implement `UNKNOWN` novelty detection.

### 4.2 RUL

1.  Initialize particles from Stage 3 health estimates and covariance.
2.  Propagate degradation state.
3.  Freeze updates during ambiguity.
4.  Use physics-derived failure thresholds.
5.  Produce p05/p50/p95 RUL.
6.  Evaluate alpha-lambda accuracy and interval coverage.

### 4.3 Mission planner

1.  Simulate current mission with `PhysicsTwin.predict`.
2.  Generate feasible candidate manoeuvres.
3.  Apply flight-envelope and return-margin filters before scoring.
4.  Score targeted separability gain.
5.  Apply life/time/fuel/exposure cost.
6.  Apply value-of-information test.
7.  Return SAFE / REPLAN / ABORT or probe recommendation.

### 4.4 Probe ordering

This ordering is mandatory:

``` text
AMBIGUOUS
   ↓
admissible fleet borrow?
   ├── yes → BORROW
   └── no  → score probes
```

The original solution explicitly makes this a correctness requirement
because a probe should never be flown when the fleet already contains an
admissible shape. fileciteturn1file0

### 4.5 Stage 4 exit criteria

-   G3.1--G3.8 satisfied.
-   ML calibration published.
-   RUL interval coverage validated.
-   Probe safety tested across randomized missions.
-   Fleet-borrow-before-probe behavior tested.
-   Nominal mission risk simulation validated.
