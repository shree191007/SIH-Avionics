# Stage 1 --- Physics Twin & Residual Generation (L1--L2)

**SIH26054 · Replan to Learn** · implementation spec, v1.0 **Owns:**
levels L1 (mean-value engine model) and L2 (residual generation)
**Consumes:** raw telemetry + ambient state · **Produces:** normalised
residual vector `z_t` and regime label `r_t`

Everything downstream --- estimation, the identifiability gate, the
fleet fingerprint, RUL, the planner --- reads Stage 1's output and
nothing else. If Stage 1's residuals are biased by operating condition,
every claim on pages 2 and 3 of the pitch collapses. This stage is
therefore specified tighter than the rest.

------------------------------------------------------------------------

## 0. Definition of done

  -------------------------------------------------------------------------
  \#                      Gate                      Threshold
  ----------------------- ------------------------- -----------------------
  G1.1                    Steady-state gas-path fit ≤ 3 % on power, ≤ 4 %
                          vs Rotax published curves on BSFC across the
                                                    published RPM/MAP grid

  G1.2                    Thermal sub-model fit vs  CHT RMSE ≤ 6 °C,
                          held-out healthy flight   oil-temp RMSE ≤ 4 °C,
                          data                      oil-pressure RMSE ≤
                                                    0.25 bar

  G1.3                    Residual whiteness on     Ljung--Box p \> 0.05 at
                          healthy data              lag 20 for every
                                                    channel

  G1.4                    **Regime-independence**   \|mean(z)\| ≤ 0.25 σ in
                          (the load-bearing one)    *every* regime bin, and
                                                    ≤ 0.4 σ in the worst
                                                    bin

  G1.5                    Determinism               Bit-identical `z_t` for
                                                    identical input on two
                                                    runs, same seed

  G1.6                    Edge budget               ≤ 8 ms per 1 Hz step, ≤
                                                    40 MB RSS on the target
                                                    SBC
  -------------------------------------------------------------------------

G1.4 is the acceptance criterion that matters. A residual that still
moves with climb has not removed the "working hard" component and Fig 1
of the pitch is not true.

------------------------------------------------------------------------

## 1. Plant, sensors, and what is actually available

### 1.1 Engine

Rotax 915 iS A --- 4-cylinder, 4-stroke, turbocharged and intercooled,
1352 cm³, dual-redundant EMS, dry-sump forced lubrication, liquid-cooled
heads with ram-air cooled cylinders.

> **Constant policy.** Every engine constant in this spec is a *named
> symbol with a source tag*, never an inline literal. Values marked
> `[ROTAX]` come from the operator's manual / published performance
> curves; `[FIT]` are identified from data; `[ASSUMED]` are engineering
> estimates that must appear in the limitations table of any report. No
> constant may be `[ASSUMED]` and also have a sensitivity above 0.1 in
> the Stage 2 sensitivity matrix --- if one does, it gets promoted to
> `[FIT]` or the model is restructured.

### 1.2 Sensor set (the six that carry the claim)

  -----------------------------------------------------------------------------
  Channel         Symbol         Rate           Nominal σ      Notes
  --------------- -------------- -------------- -------------- ----------------
  Exhaust gas     `EGT[1..4]`    1 Hz           8 °C           4 channels; the
  temperature,                                                 spatial axis
  per cylinder                                                 Stage 2 exploits

  Cylinder-head / `CHT`          1 Hz           2.5 °C         On the 915 iS
  coolant                                                      this is the
  temperature                                                  liquid-cooled
                                                               head circuit;
                                                               treated as the
                                                               head thermal
                                                               node. **Document
                                                               the
                                                               substitution**
                                                               --- it is not a
                                                               classical
                                                               air-cooled CHT

  Oil pressure    `p_oil`        1 Hz           0.12 bar       

  Oil temperature `T_oil`        1 Hz           1.5 °C         

  Engine speed    `N`            10 Hz          4 rpm          Downsampled to 1
                                                               Hz for L1; the
                                                               10 Hz stream is
                                                               retained for the
                                                               misfire proxy

  Fuel flow       `ṁ_f`          1 Hz           0.6 L/h        
  -----------------------------------------------------------------------------

### 1.3 Exogenous inputs (model drivers, not health signals)

`MAP`, throttle position `TPS`, airbox/intake air temperature `T_im`,
ambient static pressure `p_amb`, ambient temperature `T_amb`, true
airspeed `V_TAS`, pressure altitude `h_p`.

Inputs are *never* residualised. If an input channel drops out, L1 emits
`status = DEGRADED_INPUT` and L2 suppresses the affected residuals
rather than emitting a residual computed against a guessed input.

### 1.4 Data sources

  -----------------------------------------------------------------------
  Source                  Used for                Caveat carried forward
  ----------------------- ----------------------- -----------------------
  Rotax published power & gas-path and torque     steady-state only,
  BSFC curves             calibration `[ROTAX]`   sea-level standard day

  NGAFID flight archive   thermal, oil, and       **general-aviation
                          dynamic calibration     trainer data, not
                          `[FIT]`                 MALE-UAV** ---
                                                  transferability of
                                                  physics is argued, not
                                                  assumed

  Physics-injected        fault demonstrations    no public UAV
  degradation                                     piston-engine failure
                                                  corpus exists
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 2. L1 --- mean-value engine model

Continuous-time, lumped-parameter, mean-value (cycle-averaged). No
crank-angle resolution: the sensor set cannot observe it and the edge
budget cannot afford it.

### 2.1 State vector

    x = [ p_im,     intake-manifold pressure            (Pa)
          T_hd,     head / coolant node temperature     (K)
          T_oil,    oil node temperature                (K)
          T_exh,    exhaust manifold gas temperature    (K)
          ω,        crankshaft speed                    (rad/s)   — driven, see §2.7
          T_cyl[4]  per-cylinder head temperature       (K) ]

`T_cyl[i]` are the four per-cylinder nodes that make the spatial
decomposition in Stage 2 possible. They are weakly coupled to `T_hd`
through the coolant circuit.

### 2.2 Health parameter vector θ

These are the quantities the whole system exists to estimate. Nominal
value 1.0 = as-new.

  -----------------------------------------------------------------------------------
  j              Symbol         Physical meaning        Nominal        Admissible
                                                                       range
  -------------- -------------- ----------------------- -------------- --------------
  1              `θ_vol`        volumetric-efficiency   1.0            \[0.80, 1.05\]
                                multiplier                             
                                (induction/valve                       
                                health)                                

  2              `θ_comb`       combustion /            1.0            \[0.80, 1.05\]
                                indicated-efficiency                   
                                multiplier                             

  3              `θ_cool`       head heat-transfer      1.0            \[0.70, 1.10\]
                                coefficient multiplier                 
                                (cooling path)                         

  4              `θ_inj[i]`     per-cylinder injector   1.0            \[0.80, 1.10\]
                                delivery multiplier, i                 
                                = 1..4                                 

  5              `θ_oilp`       oil-pump/clearance      1.0            \[0.75, 1.05\]
                                multiplier                             

  6              `θ_fric`       friction (FMEP)         1.0            \[0.95, 1.30\]
                                multiplier                             

  7              `b_EGT[i]`,    additive sensor biases  0.0            ±3 σ
                 `b_CHT`,                                              
                 `b_poil`                                              
  -----------------------------------------------------------------------------------

**θ_cool and θ_comb are the ambiguous pair** the whole contribution is
built around. They must be separate parameters in the model, not a
lumped "thermal health" scalar --- lumping them would make the pitch's
central claim unfalsifiable.

### 2.3 Induction and gas path

    ṁ_air = θ_vol · η_vol(N, p_im) · V_d · (N / (2·60)) · p_im / (R_air · T_im)

-   `η_vol(N, p_im)` --- 2-D lookup, bicubic, 8×8 breakpoints, `[FIT]`
    against `[ROTAX]` power curve
-   `V_d = 1.352e-3 m³` `[ROTAX]`
-   Turbo/wastegate is **not** modelled dynamically. `p_im` is taken
    from the measured `MAP` channel and treated as an input. Rationale:
    the wastegate controller is proprietary, modelling it adds three
    `[ASSUMED]` constants, and `MAP` is measured anyway. Documented
    limitation: a boost-control fault will present as an unexplained
    input, not as a θ.

Commanded fuel and lambda:

    ṁ_f,cyl[i] = θ_inj[i] · ṁ_air / (4 · λ_cmd(N, p_im) · AFR_st)
    ṁ_f        = Σ_i ṁ_f,cyl[i]
    λ_i        = ṁ_air / (4 · ṁ_f,cyl[i] · AFR_st)

`λ_cmd` is the ECU's commanded mixture map, `[FIT]` from cruise data at
known `θ = 1`.

### 2.4 Combustion, torque, power

    Q̇_fuel   = ṁ_f · LHV
    η_ind     = θ_comb · η_ind,0(λ, N, p_im)
    P_ind     = θ_comb · η_ind,0 · Q̇_fuel
    FMEP      = θ_fric · (c_0 + c_1·N + c_2·N²)
    P_brake   = P_ind − FMEP · V_d · N/(2·60)

`η_ind,0` is a Wiebe-free efficiency surface `[FIT]`; `c_0..c_2` are
Chen--Flynn form `[FIT]`.

### 2.5 Exhaust temperature (the EGT model)

Energy not converted to work and not lost to the head goes out the
exhaust:

    T_exh = T_im + (1 − η_ind − f_ht(N, p_im)) · Q̇_fuel / (ṁ_air · c_p,exh)
    EGT_i = T_exh,i − k_probe · (T_exh,i − T_amb_duct) + b_EGT[i]

with the per-cylinder split driven by `θ_inj[i]` through `λ_i`.
`k_probe` is the probe recovery/immersion loss factor `[FIT]`. This is
where a lean cylinder shows up as *hotter* EGT and a rich one as cooler
--- the standard piston-engine signature the demo depends on.

### 2.6 Thermal network (two-node + four cylinder nodes)

    C_hd  dT_hd/dt  = Q̇_to_head − θ_cool · UA_hd(V_TAS, ρ_amb, N) · (T_hd − T_amb)
                                  − UA_ho · (T_hd − T_oil)

    C_oil dT_oil/dt = Q̇_fric + UA_ho·(T_hd − T_oil)
                                  − θ_cool_oil · UA_oc(V_TAS, ρ_amb) · (T_oil − T_amb)

    C_c   dT_cyl[i]/dt = Q̇_to_head,i − UA_c,i(V_TAS, ρ_amb) · (T_cyl[i] − T_amb)
                                  − UA_cc · (T_cyl[i] − T_hd)

`UA_hd`, `UA_c,i`, `UA_oc` are cooling-airflow conductances, `[FIT]`
with a `V_TAS^0.8 · ρ^0.8` forced-convection form. `UA_c,i` carries a
**fixed per-cylinder asymmetry factor** `a_i` `[FIT]` --- the rear
cylinders genuinely run hotter, and Stage 2's spatial decomposition is
invalid unless that nominal asymmetry is modelled here, not attributed
to a fault.

> This is the single most important correction carried over from the
> cross-cylinder novelty review: cylinders are **not exchangeable**, and
> `d_ij` has a non-zero, regime-dependent nominal value. Stage 1 owns
> that nominal; Stage 2 owns only the deviation from it.

### 2.7 Oil pressure

    p_oil = θ_oilp · [ k_pump · N · μ(T_oil) / (k_leak + k_clear·N) ] + b_poil
    μ(T_oil) = Vogel / Walther fit to the certified oil grade      [ASSUMED → FIT]

### 2.8 Crankshaft speed

`ω` is **not** integrated. The propeller governor closes a loop we do
not model; `N` is taken as a measured input to the gas-path model and
simultaneously residualised against a propeller-load torque balance:

    r_N :  P_brake(model) − P_prop(N, V_TAS, ρ, β_blade)

This yields a *power-balance* residual in kW, which is the channel most
sensitive to `θ_comb` and `θ_fric` and least sensitive to thermal
parameters --- deliberately, because it breaks the cooling/combustion
degeneracy partially even before the gate runs.

### 2.9 Numerics

-   Integrator: fixed-step RK4, `h = 0.1 s`, output decimated to 1 Hz.
-   Thermal states are stiff relative to gas path --- head and oil nodes
    use a semi-implicit (backward-Euler) update to allow the 0.1 s step
    without instability.
-   All state clamped to physically admissible ranges; a clamp event
    raises `status = MODEL_SATURATED` and invalidates that step's
    residuals.
-   Float64 in offline identification; float32 permitted on edge after a
    G1.5 re-check.

------------------------------------------------------------------------

## 3. Calibration procedure

Two-source, run in this order, and **reported as two sources** in every
paper and slide.

### Phase A --- gas path and power, against `[ROTAX]`

1.  Build the published grid: for each `(N, p_im)` breakpoint, published
    shaft power and BSFC.
2.  Fit `η_vol`, `η_ind,0`, `c_0..c_2` by weighted least squares, with
    monotonicity and physical-bound constraints (`η_vol ∈ [0.5, 1.15]`,
    `η_ind,0 ∈ [0.20, 0.42]`).
3.  Optimiser: trust-region reflective, 20 random restarts, keep best by
    held-out grid RMSE.
4.  **Gate G1.1.**

### Phase B --- thermal, oil, and dynamics, against `[FIT]` flight data

1.  Select healthy flight segments (§3.1) from the NGAFID corpus.
2.  Freeze Phase A parameters. Fit `C_hd`, `C_oil`, `C_c`, `UA_*`,
    `a_i`, `k_probe`, `μ(T_oil)` coefficients by prediction-error
    minimisation over full flights.
3.  **Gate G1.2.**

### Phase C --- noise characterisation

1.  On held-out healthy segments, compute residual time series per
    channel.
2.  Estimate `σ_r` **per regime bin r**, not globally --- sensor noise
    and model error are both regime-dependent, and Stage 2's fingerprint
    normalises by exactly this quantity.
3.  Fit an AR(1) whitening filter per channel if G1.3 fails on the raw
    residual.
4.  Persist as `noise_model.json` (§6.3). **Gates G1.3, G1.4.**

### 3.1 "Healthy" segment selection

A segment is admissible for calibration if all hold:

-   no maintenance event in the following 100 flight hours (from the
    maintenance log)
-   no ECU fault code active
-   all six channels present, no gap \> 3 s
-   length ≥ 300 s
-   at least one steady sub-segment of ≥ 120 s

Selection is committed as a manifest file with a content hash so
calibration is reproducible.

------------------------------------------------------------------------

## 4. Regime definition --- the axis everything else is indexed by

The regime label `r` is the index of the sensitivity fingerprint in
Stage 2, the thing an aircraft "has or has not flown", and the unit of
fleet federation. Its definition is therefore **frozen and versioned**
--- a change to the regime grid invalidates every stored fleet
fingerprint.

### 4.1 Grid `REGIME_GRID_V1` (12 bins)

  -----------------------------------------------------------------------
  Axis                                Bins
  ----------------------------------- -----------------------------------
  Phase                               `CLIMB` (ROC \> +1.5 m/s), `CRUISE`
                                      (\|ROC\| ≤ 1.5 m/s), `DESCENT` (ROC
                                      \< −1.5 m/s)

  Power                               `LOW` (P_brake \< 45 % MCP), `MID`
                                      (45--75 %), `HIGH` (\> 75 %)
  -----------------------------------------------------------------------

12 bins = 3 phases × 3 power bands, plus `IDLE` and `TRANSIENT` and
`INVALID` as non-fingerprint labels (a total of 15 labels, 12 of which
are fingerprint-bearing).

### 4.2 Stability requirement

A sample is assigned a regime only if the engine is quasi-steady:

    |dN/dt| < 30 rpm/s   AND   |dp_im/dt| < 2 kPa/s   AND   |dT_hd/dt| < 0.15 K/s
    sustained for ≥ 20 s

Otherwise `r = TRANSIENT` and the sample contributes to detection but
**not** to any sensitivity estimate. Thermal states have long time
constants; a fingerprint computed mid-transient is dominated by thermal
lag, not by health.

### 4.3 Regime encounter counter

Per aircraft, per regime, maintain `n_i(r)` = seconds of quasi-steady
time accumulated. This is the sample count that sets `v_i(θ_j, r)` in
Stage 2's fusion weighting, and it is what makes "this aircraft has
never flown a climb" a machine-checkable statement rather than a
narrative one.

------------------------------------------------------------------------

## 5. L2 --- residual generation

### 5.1 Raw residual

    e_t(c) = y_t(c) − ŷ_t(c ; u_t, x̂_t, θ_nom)      for each channel c

Computed against **nominal θ**, not estimated θ. Stage 3 (L3) is the
thing that moves θ; if L2 residualised against the current estimate it
would cancel the very signal L3 needs.

### 5.2 Whitening and normalisation

    w_t(c) = e_t(c) − a_c · e_{t−1}(c)            AR(1) whitening, a_c from Phase C
    z_t(c) = w_t(c) / σ_{r_t}(c)                  regime-conditioned normalisation

`z_t` is dimensionless and comparable across channels. **Every
downstream consumer uses `z`, never `e`.** This is what makes the cosine
metric in Stage 2 meaningful: the fingerprint is defined on
noise-normalised sensitivities, so its geometry is only stable if the
residual is normalised the same way.

### 5.3 Channel set of `z`

    z = [ z_EGT[1..4], z_CHT, z_poil, z_Toil, z_mf, z_N(power balance) ]     → 9 elements

### 5.4 Spatial pre-decomposition (handed to Stage 2)

The EGT block is additionally projected onto the fixed engine-geometry
basis:

    z_EGT = α_t · 1 + β_t · g + s_t

-   `1` = ones(4) --- common mode (global degradation, ECU-wide bias)
-   `g` = cylinder cooling-order vector, unit-norm, orthogonalised
    against `1`, `[FIT]` in §2.6
-   `s_t` = residual sparse component, `s_t ⊥ span{1, g}`

`(α_t, β_t, ‖s_t‖_∞, argmax_i |s_t,i|)` are emitted alongside `z`. This
is the "magnitude × spatial pattern" part of the fingerprint; the
cross-modal EGT/CHT coherence term is computed in Stage 2.

> Recorded limitation: `{d_ij}` spans only N−1 = 3 dimensions, so a
> perfectly uniform degradation is invisible on the spatial axis by
> construction. That class still needs the regime axis --- which is
> exactly the class the fleet federation in Stage 2 serves.

------------------------------------------------------------------------

## 6. Interfaces

### 6.1 Input frame

``` python
@dataclass(frozen=True)
class TelemetryFrame:
    t: float                    # s, monotonic, aircraft clock
    egt: tuple[float,...]       # K, len 4
    cht: float                  # K
    p_oil: float                # Pa
    t_oil: float                # K
    n_rpm: float                # rpm
    mdot_f: float               # kg/s
    # exogenous
    map_pa: float; tps: float; t_im: float
    p_amb: float; t_amb: float; v_tas: float; h_p: float
    valid: int                  # bitmask, 1 = channel present
```

### 6.2 Output frame --- the Stage 1 → Stage 2 contract

``` python
@dataclass(frozen=True)
class ResidualFrame:
    t: float
    z: np.ndarray               # float32[9], normalised residuals
    regime: int                 # index into REGIME_GRID_V1, or TRANSIENT/INVALID
    regime_stable_s: float      # how long the current regime has held
    spatial: tuple[float,float,float,int]   # (alpha, beta, |s|_inf, argmax_i)
    sigma: np.ndarray           # float32[9], sigma used, for audit
    x_hat: np.ndarray           # float32[9], model state, for the planner's simulator
    status: int                 # OK | DEGRADED_INPUT | MODEL_SATURATED | INVALID
    model_version: str          # semver of the calibrated model
    regime_grid_version: str    # "REGIME_GRID_V1"
```

`model_version` and `regime_grid_version` travel with every frame and
are checked by the fleet node in Stage 2. A fingerprint computed under a
different model version is **not** fusable and must be rejected, not
silently averaged.

### 6.3 Persisted calibration artefacts

    model/
      gaspath_params.json       # Phase A, with [ROTAX] provenance per constant
      thermal_params.json       # Phase B, with [FIT] provenance + source manifest hash
      noise_model.json          # Phase C: sigma[channel][regime], AR(1) coeffs
      regime_grid_v1.json       # frozen bin edges
      MANIFEST.sha256

### 6.4 Public API

``` python
class PhysicsTwin:
    def __init__(self, model_dir: Path, dt: float = 0.1) -> None: ...
    def reset(self, frame: TelemetryFrame) -> None: ...
    def step(self, frame: TelemetryFrame) -> ResidualFrame: ...
    def predict(self, u: InputSequence, theta: np.ndarray) -> np.ndarray:
        """Open-loop forward simulation. Used by L7's mission simulator and by
        L4's finite-difference sensitivity computation. Must be side-effect free."""
    def jacobian(self, u: InputFrame, theta: np.ndarray) -> np.ndarray:
        """d(y)/d(theta), shape (9, n_theta). Central differences, step 1e-3,
        or analytic where available. Stage 2 depends on this being consistent
        with predict() to 1e-6 relative."""
```

`jacobian` is a Stage 1 deliverable even though only Stage 2 uses it,
because it must be consistent with the same model instance that produced
the residual.

------------------------------------------------------------------------

## 7. Fault injection harness (needed by every later stage)

``` python
def inject(theta_nom, fault: FaultSpec, t: float) -> np.ndarray
```

  ----------------------------------------------------------------------------
  Fault class             Parameterisation             Profile
  ----------------------- ---------------------------- -----------------------
  Injector degradation,   `θ_inj[i] : 1.0 → 1.0 − a`   ramp over 5--40 flight
  cyl i                                                hours

  Cooling degradation     `θ_cool : 1.0 → 1.0 − a`,    ramp; `w ≠ 0`
                          with a front/rear weighting  deliberately, because
                          `w`                          real cooling loss is
                                                       **not** uniform

  Combustion degradation  `θ_comb : 1.0 → 1.0 − a`     ramp

  EGT probe drift, cyl i  `b_EGT[i] : 0 → b`           ramp or step

  Oil-system degradation  `θ_oilp : 1.0 → 1.0 − a`     ramp
  ----------------------------------------------------------------------------

The cooling fault **must** be injected with non-uniform cylinder
weighting. A uniform injection would make the spatial axis look better
than it is and the demo would not survive a knowledgeable judge.

------------------------------------------------------------------------

## 8. Test plan

### 8.1 Unit

-   Every sub-model against a hand-computed case (energy balance closes
    to 1e-9).
-   `jacobian` vs `predict` finite differences, all θ, 1e-6 relative.
-   Regime classifier against a labelled 200-segment fixture, ≥ 98 %
    agreement.

### 8.2 Property-based

-   Monotonicity: `θ_cool ↓` ⇒ `T_hd ↑` at fixed input, all regimes.
-   Scale: doubling `dt` changes `z` by \< 1 % (integrator convergence).
-   Idempotence: `reset` + replay reproduces a run bit-identically
    (G1.5).

### 8.3 Statistical (on held-out healthy data)

-   G1.3 whiteness, G1.4 regime-independence, per channel and per regime
    bin.
-   Report the **regime-independence table** as a figure: mean(z) by
    channel × regime, with the ±0.25 σ band drawn. This figure is the
    evidence for Fig 1 of the pitch and should appear in the submission.

### 8.4 Fault-recovery smoke test

Inject each fault class at a known magnitude, confirm the corresponding
residual channels move in the predicted direction and the uninvolved
ones do not. This is not a diagnosis test --- that is Stage 2 --- it
only proves the model is wired correctly.

------------------------------------------------------------------------

## 9. Risks

  -----------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- -----------------------
  CHT-vs-coolant          High                    Model the head node as
  substitution                                    what it physically is;
  invalidates the thermal                         state the substitution
  fault signature                                 in every limitations
                                                  section; validate
                                                  `θ_cool` sensitivity on
                                                  flight data before
                                                  Stage 2 depends on it

  NGAFID engines are not  High                    Calibrate *structure*
  915 iS                                          on NGAFID and
                                                  *magnitudes* on Rotax;
                                                  report cross-engine
                                                  transfer as an argued
                                                  assumption, never as a
                                                  validated one

  Wastegate/boost fault   Medium                  Detect via `MAP` vs
  presents as unmodelled                          `TPS` consistency
  input                                           check; raise
                                                  `DEGRADED_INPUT` rather
                                                  than attributing to θ

  `η_vol` and `θ_comb`    Medium                  Deliberately retain the
  partially collinear at                          power-balance residual
  fixed λ                                         `r_N`, which loads them
                                                  differently; hand the
                                                  collinearity to Stage
                                                  2's gate rather than
                                                  hiding it

  Regime grid changes     High                    `regime_grid_version`
  after fleet data exists                         in every frame; fleet
                                                  node rejects mismatched
                                                  versions
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 10. Deliverables

1.  `physics_twin/` package implementing §2, §5, §6.4.
2.  Calibrated `model/` artefacts with provenance tags and manifest
    hash.
3.  Fault-injection harness (§7).
4.  Test suite covering §8, running in CI on the fixture corpus.
5.  **Regime-independence figure** (G1.4) and the two-source calibration
    report.

# Modification applied in the five-stage plan

The original L1--L2 specification is retained as the scientific source
of truth for the Physics Twin. Its detailed model, calibration
procedure, regime grid, residual definition, interfaces, and
fault-injection harness are now implemented as **Stage 2** rather than
Stage 1.

Stage 2 must therefore deliver:

1.  `PhysicsTwin`
2.  calibration artifacts
3.  `REGIME_GRID_V1`
4.  normalized residual generation
5.  spatial EGT decomposition
6.  `jacobian()`
7.  fault-injection harness
8.  Stage 2 CI tests
9.  G1.1--G1.6 evidence

The key dependency remains:

``` text
raw telemetry + ambient state
          ↓
      PhysicsTwin
          ↓
predicted measurements
          ↓
       residual
          ↓
whitening + regime normalization
          ↓
     ResidualFrame
```

The existing requirement that every downstream consumer uses `z`, never
the unnormalized residual `e`, remains unchanged. fileciteturn1file2
