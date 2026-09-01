# Stage 2 --- State Estimation, Identifiability Gate & Fleet Federation (L3, L4, L4F)

**SIH26054 · Replan to Learn** · implementation spec, v1.0 **Owns:** L3
(parameter estimation), L4 (identifiability gate), L4F (fleet
fingerprint node) **Consumes:** `ResidualFrame` from Stage 1 ·
**Produces:** `AttributionFrame` for Stages 3 and 7

This is where the project's contribution lives. L4 is the runtime gate
that refuses to name a subsystem it cannot separate; L4F is the
mechanism that lets an aircraft resolve that refusal using regimes it
has never flown, by borrowing a *shape* from the fleet.

------------------------------------------------------------------------

## 0. Definition of done

  ------------------------------------------------------------------------
  \#                      Gate                    Threshold
  ----------------------- ----------------------- ------------------------
  G2.1                    θ estimation converges  \|θ̂ − θ_true\| ≤ 0.02
                          on injected single      within 30 min of flight,
                          faults                  all fault classes

  G2.2                    Gate **refuses** on the refusal rate ≥ 95 % for
                          designed-ambiguous pair cooling-vs-combustion,
                          at cruise-only          cruise-only exposure

  G2.3                    Gate **resolves** on    resolution rate ≥ 90 %,
                          the same pair given     misattribution ≤ 5 %
                          climb+cruise+descent    

  G2.4                    **Over-confident        strictly lower at every
                          misattribution rate**   operating point of the
                          vs unguarded baseline   ROC; this is the
                                                  decisive metric of the
                                                  whole project

  G2.5                    Fleet transfer:         resolution rate ≥ 80 %
                          hybrid-fingerprint gate of the full-regime rate,
                          on cruise-only aircraft with false-borrow ≤ 5 %

  G2.6                    Guardrail discriminates R² separates
                                                  matched-shape from
                                                  mismatched-shape
                                                  transfers with AUC ≥ 0.9

  G2.7                    Uplink budget           ≤ 2 kB per aircraft per
                                                  fusion round; **no raw
                                                  flight data in any
                                                  message**

  G2.8                    Edge budget             gate evaluation ≤ 50 ms
                                                  per invocation, ≤ 60 MB
                                                  RSS
  ------------------------------------------------------------------------

**G2.4 is the metric the submission is judged on.** If the guarded
system does not beat the unguarded one on over-confident misattribution,
we report that and the project's central claim is falsified. This is
written into the spec deliberately.

------------------------------------------------------------------------

## 1. L3 --- parameter estimation

### 1.1 Estimator choice

Joint UKF over the augmented state `[x ; θ]`, with θ modelled as a slow
random walk.

Rationale, briefly: the model is mildly non-linear in θ (thermal
conductances multiply temperature differences), the dimension is small
(\~19 states + \~12 parameters), and the UKF gives a covariance `P_θ`
for free --- which the gate in §2 needs and which an optimisation-based
estimator would not supply. EKF was rejected because the
finite-difference Jacobian of the thermal network is poorly conditioned
near saturation; a particle filter was rejected on the edge CPU budget
(it is used in Stage 3 for RUL, where the dimension is 1--2).

### 1.2 Process model

    x_{k+1} = f(x_k, u_k, θ_k) + w_k          w ~ N(0, Q_x)
    θ_{k+1} = θ_k + q_k                        q ~ N(0, Q_θ)
    y_k     = h(x_k, u_k, θ_k) + v_k           v ~ N(0, R_k)

-   `Q_θ = diag(σ_drift²)·Δt` with `σ_drift` set so that a parameter can
    move its full admissible range over \~500 flight hours. Too large
    and the filter chases noise; too small and it will not track a
    genuine ramp. Tuned on injected ramps, reported.
-   `R_k` is **regime-conditioned**, taken directly from Stage 1's
    `noise_model.json`.
-   UKF scaling: `α = 1e-3`, `β = 2`, `κ = 0`. Square-root formulation
    (Cholesky-update) for numerical stability in float32 on edge.

### 1.3 Constraints

θ is box-constrained (Stage 1 §2.2). Constraint handling by projection
of the posterior mean with covariance inflation, not by clipping sigma
points (clipping biases the covariance and would silently corrupt the
gate's CRLB).

### 1.4 Freeze policy

θ is **not** updated during `TRANSIENT` or `MODEL_SATURATED` samples.
The filter propagates covariance and skips the measurement update. This
matters: thermal lag during a transient looks exactly like a cooling
parameter change.

### 1.5 Outputs

    θ̂_t        posterior mean, n_θ vector
    P_θ,t      posterior covariance, n_θ × n_θ
    ν_t        normalised innovation sequence (used for model-validity monitoring)

------------------------------------------------------------------------

## 2. L4 --- the identifiability gate

The gate answers one question, every time attribution is about to be
reported: **given the regimes this aircraft has actually flown, can
these candidate faults be told apart at all?** It runs *before* a
subsystem is named, and its refusal is a first-class output, not an
error.

### 2.1 The sensitivity fingerprint

For aircraft *i*, candidate parameter θ_j, and observed regime set
`R_i`:

    f_i(θ_j) = [ (∂y(r)/∂θ_j) / σ_r ]        for r ∈ R_i

-   `∂y(r)/∂θ_j` comes from `PhysicsTwin.jacobian` evaluated at the
    regime-representative operating point and current θ̂.
-   `σ_r` is the regime-conditioned noise scale from Stage 1. **The
    normalisation is what makes the geometry meaningful** --- an
    unnormalised fingerprint would rank a noisy channel's large raw
    sensitivity above a quiet channel's decisive one.
-   Layout: `f_i(θ_j) ∈ R^(n_regime × n_channel)`, flattened. With
    `REGIME_GRID_V1` (12 bins) and 9 channels that is a 108-vector, of
    which only the rows for flown regimes are populated.

### 2.2 Separability metric

    cos(θ_j, θ_k) = ⟨ f_i(θ_j), f_i(θ_k) ⟩ / ( ‖f_i(θ_j)‖ · ‖f_i(θ_k)‖ )

Cosine, not correlation and not Euclidean distance, because the true
magnitude of a degradation is unknown --- only its direction of effect
is observable. Scale-invariance is the required property, and it is the
same property that makes the fingerprint federatable (§3).

**Sign matters.** `|cos|` near 1 means inseparable; `cos` near −1 means
the two faults push the sensors in opposite directions and are
*maximally* separable. The gate therefore tests `|cos|`, and any code
that tests `cos` alone is wrong.

### 2.3 Two orthogonal axes beyond regime

The cross-cylinder work established that regime is not the only axis
available. The gate consumes three, concatenated into the fingerprint
before the cosine is taken:

  ----------------------------------------------------------------------------------------
  Axis                    Content                 What it separates without excitation
  ----------------------- ----------------------- ----------------------------------------
  **Regime** (§2.1)       sensitivity across the  the classical axis; needs the aircraft
                          12 regime bins          to have flown the regimes

  **Spatial**             sensitivity of          global bias → pure α; cooling → α +
                          `(α, β, s)` from Stage  strong β; single injector → sparse s.
                          1 §5.4                  Cooling-vs-sensor-bias separates on β
                                                  **from one sample**

  **Cross-modal**         sensitivity of          genuine combustion faults move EGT and
                          per-cylinder EGT/CHT    CHT together in a physically coupled
                          coherence               way; a failing probe moves one and not
                                                  the other. Closes the
                                                  localized-sensor-vs-localized-hardware
                                                  ambiguity the spatial axis opens
  ----------------------------------------------------------------------------------------

Recorded honestly in the limitations: the spatial axis spans only N−1 =
3 dimensions, so a *perfectly uniform* degradation is invisible on it.
That class is precisely the one the regime axis --- and therefore the
fleet federation of §3 --- exists to serve.

### 2.4 Fisher information and the confidence claim

The fingerprint matrix `S_i ∈ R^(m × n_θ)` stacks the fingerprints
columnwise. Then

    F = Sᵀ S                       (already noise-normalised, so Σ⁻¹ is folded in)
    CRLB(θ_j) = sqrt( [F⁻¹]_jj )

Both tests must pass before a subsystem is named:

    SEPARABLE(j,k)   ⟺   |cos(θ_j, θ_k)| < τ_sep
    IDENTIFIABLE(j)  ⟺   CRLB(θ_j) < δ_j

-   `τ_sep = 0.90` default. Calibrate on the fixture corpus; the
    reference demo separates cooling-vs-sensor-bias at `cos = 0.818`
    once climb and descent are present.
-   `δ_j` is a **physically meaningful** bound, not a statistical one:
    the smallest change in θ_j that would alter a maintenance decision.
    Set per parameter from the maintenance manual, not from the data.

`F⁻¹` is computed by Cholesky with a rank check; if `F` is
rank-deficient the gate returns `AMBIGUOUS` for the whole deficient
subspace rather than pseudo-inverting. Pseudo-inverting here is exactly
the "returns something anyway" failure the project exists to attack.

### 2.5 Gate output

    NAMED      — separable and identifiable; report subsystem + CRLB interval
    AMBIGUOUS  — the inseparable set {θ_j, θ_k, ...} is reported explicitly, with the
                 pairwise |cos| values, and handed to L7 as a probe request (Stage 3)
    BORROWED   — resolved on a hybrid fingerprint (§3); carries R², |R_i|, fleet size,
                 and an inflated CRLB
    INVALID    — model or input status prevents any claim

`AMBIGUOUS` is never rendered as a low-confidence guess in the GCS. The
panel states the candidate set. This is a product requirement, not a
display preference.

------------------------------------------------------------------------

## 3. L4F --- fleet-federated identifiability

### 3.1 The claim, precisely

An aircraft that has only flown cruise has an empty fingerprint in the
climb and descent rows, so `F` is rank-deficient over `{θ_cool, θ_comb}`
and the gate must refuse. A sister aircraft has flown a hundred climbs.
What can legitimately be transferred between two *different physical
engines* is not data and not a model: it is the **shape** of the fault's
sensitivity across regimes.

**Why the shape, and only the shape.** Cosine is scale-invariant ---
`cos(αv, v) = 1` for any `α > 0`. That is what made it the right metric
*locally*: the true magnitude of θ_j is never known, only its direction
of effect. It is the same property that makes it right *across a
heterogeneous fleet*. Two Rotax units with different build tolerance,
wear state or calibration drift respond with a different **magnitude**
to the same fault, but the same fault produces the same **shape** across
regimes --- because that shape comes from shared physics (how heat load
varies with regime), not from unit-to-unit manufacturing variance. Scale
is the part that does not transfer; direction is the part that does; and
direction is exactly the invariant the gate already relies on.

### 3.2 Transfer model

    f_i(θ_j)|_r = α_i(θ_j) · u_j(r) + ε_i(r)

  -----------------------------------------------------------------------
  Symbol                  Meaning                 Where it lives
  ----------------------- ----------------------- -----------------------
  `u_j`                   fleet-canonical         fleet node
                          **unit-norm shape** for 
                          fault θ_j over the full 
                          regime set `R`          

  `α_i(θ_j)`              aircraft *i*'s own      on-aircraft
                          scalar gain --- build   
                          tolerance, wear,        
                          calibration drift       

  `ε_i(r)`                aircraft-specific       not modelled, bounded
                          residual / idiosyncrasy by the guardrail
  -----------------------------------------------------------------------

### 3.3 Sign convention (required, easy to get wrong)

A unit-norm shape is defined only up to sign. Without a convention, two
fleet nodes can converge to `u_j` and `−u_j`, and every borrowed
fingerprint flips. Fix:

    r_ref(j) = argmax_r |u_j^prior(r)|          # from the twin's own jacobian
    u_j      ← u_j · sign( u_j(r_ref(j)) )      # applied after every fusion round

`r_ref(j)` is frozen per parameter at model-version time and stored in
the fleet artefact.

### 3.4 On-aircraft: estimate the gain, then fill in what was never flown

Least-squares projection of the local fingerprint onto the fleet shape,
restricted to the regimes this aircraft actually has:

    α̂_i(θ_j) = ⟨ f_i(θ_j), u_j|R_i ⟩ / ‖ u_j|R_i ‖²

Then predict the entries the aircraft has never flown:

    f̂_i(θ_j)|_r = α̂_i(θ_j) · u_j(r)          for r ∉ R_i

and run the L4 gate of §2 on the **hybrid fingerprint** --- real local
entries where they exist, fleet-predicted entries where they do not.
That is what lets an aircraft that has only ever seen cruise make a
confident call using climb and descent it never flew itself.

### 3.5 Honest uncertainty on borrowed entries

A borrowed entry is **not** as good as a flown one, and the gate must
not pretend otherwise. Each predicted entry carries an inflated
variance:

    Var[ f̂_i(θ_j)|_r ] = α̂_i² · Var[u_j(r)]  +  u_j(r)² · Var[α̂_i]  +  σ_ε²

    Var[α̂_i] = ( Σ_{r ∈ R_i} u_j(r)² · Var[f_i|_r] ) / ‖u_j|R_i‖⁴
    σ_ε²      = (1 − R²_i) · ‖f_i(θ_j)‖² / |R_i|        # empirical idiosyncrasy floor

The FIM in §2.4 is then built with these inflated variances on borrowed
rows, so the resulting CRLB widens honestly and `BORROWED` decisions
carry visibly wider intervals than `NAMED` ones. **Skipping this
inflation would manufacture exactly the false confidence the project is
built to eliminate.**

### 3.6 The guardrail --- check the transfer assumption before trusting it

    R²_i(θ_j) = 1 − ‖ f_i(θ_j) − α̂_i(θ_j)·u_j|R_i ‖² / ‖ f_i(θ_j) ‖²

Because `α̂` is the least-squares optimum, this reduces exactly to

    R²_i(θ_j) = cos²( f_i(θ_j), u_j|R_i )

--- the same metric the gate already uses, reused as an admissibility
test. Nice property, and worth stating in the paper: the guardrail is
not a bolted-on heuristic, it is the gate's own geometry applied to the
transfer.

**Admissibility rules --- all must hold before any entry is borrowed:**

  --------------------------------------------------------------------------------------------
  \#                      Rule                                         Reason
  ----------------------- -------------------------------------------- -----------------------
  A1                      `|R_i| ≥ 3` distinct flown regimes for θ_j   with `|R_i| = 1` the
                                                                       projection is trivially
                                                                       exact and `R² ≡ 1` ---
                                                                       the guardrail has **no
                                                                       power at all**. With 2
                                                                       it is nearly powerless.
                                                                       This is the rule most
                                                                       likely to be omitted by
                                                                       accident

  A2                      `R²_i(θ_j) ≥ τ_R2` (default 0.85)            shape mismatch:
                                                                       different failure mode,
                                                                       or a second
                                                                       co-occurring fault
                                                                       distorting the
                                                                       signature

  A3                      `α̂_i(θ_j) > 0`                               a negative gain means
                                                                       this unit responds in
                                                                       the *opposite*
                                                                       direction to the fleet
                                                                       --- physically
                                                                       incoherent, never
                                                                       borrow

  A4                      `α̂_i(θ_j) ∈ [0.25, 4.0] × median_fleet(α)`   outlier gain; likely a
                                                                       mis-scaled sensor
                                                                       rather than a build
                                                                       difference

  A5                      fleet contributor count for `u_j(r)` ≥ 3     do not borrow from a
                          aircraft, or `n_fleet(r) ≥ 200 s`            shape nobody has really
                                                                       measured

  A6                      `model_version` and `regime_grid_version`    fingerprints from a
                          match                                        different calibration
                                                                       are not comparable
  --------------------------------------------------------------------------------------------

If any rule fails, **do not borrow**: the honest output is `AMBIGUOUS`
with the reason code, and L7 is asked for a probe instead. That is the
difference between principled transfer learning and quietly papering
over a real discrepancy.

### 3.7 Fusion at the fleet node --- a federated Kalman filter, not an average

Each aircraft's local estimate of the sensitivity `ŝ_i(θ_j, r)` carries
its own uncertainty `v_i(θ_j, r)` --- from sensor noise, and from how
many times that regime has actually been encountered. Information
(inverse covariance) adds linearly across independent sources, so the
correct fusion is precision-weighted:

    ŝ_fleet(θ_j, r) = [ Σ_i ŝ_i(θ_j,r) / v_i(θ_j,r) ] / [ Σ_i 1 / v_i(θ_j,r) ]
    v_fleet(θ_j, r) = 1 / [ Σ_i 1 / v_i(θ_j,r) ]

Each aircraft therefore transmits only two scalars per cell ---
`(ŝ_i/v_i, 1/v_i)` --- and the node sums them. This is Carlson's
federated Kalman architecture (IEEE T-AES, 1990) applied to fingerprint
fusion instead of state fusion: a well-trodden estimation-theory
pattern, pointed at a new object.

**One correction the naive form needs.** The raw `ŝ_i` already contains
aircraft *i*'s own gain `α_i`, so summing raw sensitivities across a
heterogeneous fleet returns a gain-weighted blend, not the canonical
shape. The fusion is therefore run as a weighted rank-1 factorisation
--- alternating least squares on

    min over {α_i}, u   Σ_i Σ_r ( ŝ_i(r) − α_i·u(r) )² / v_i(r)      s.t. ‖u‖ = 1

whose two update steps are:

    α_i ← [ Σ_r ŝ_i(r)·u(r) / v_i(r) ] / [ Σ_r u(r)² / v_i(r) ]        # gain step
    u(r) ← [ Σ_i α_i·ŝ_i(r) / v_i(r) ] / [ Σ_i α_i² / v_i(r) ]         # shape step
    u    ← u / ‖u‖ ;  α ← α · ‖u‖ ;  apply sign convention §3.3

The shape step **is** the precision-weighted sum above, in the special
case `α_i ≡ 1`; and the gain step is the on-aircraft projection of §3.4
with per-regime weights. Nothing new is introduced --- the
information-filter update is the inner loop of a rank-1 fit. Converges
in 3--8 iterations in practice; cap at 20 and log non-convergence.

**Cold start / small fleets.** Initialise `u_j^(0)` from the physics
twin's own jacobian --- the twin *is* the shared physics, so its shape
is a principled prior --- and shrink toward it:

    u_j ← normalize( (1 − λ_shrink)·u_j^ALS + λ_shrink·u_j^prior )
    λ_shrink = κ / (κ + N_eff),   N_eff = Σ_i 1/v_i(θ_j,r),  κ tuned on the simulated fleet

A fleet of one degrades gracefully to "use the model's own shape", which
is the correct behaviour and not a special case in the code.

### 3.8 Variance model for `v_i`

    v_i(θ_j, r) = σ_reg²(θ_j, r) / n_i(r)  +  v_floor(θ_j)

-   `σ_reg²` from the covariance of the local regression that produced
    `ŝ_i` (§3.9).
-   `n_i(r)` = seconds of quasi-steady time in regime r, from Stage 1
    §4.3.
-   `v_floor` is a non-zero model-mismatch floor. Without it, an
    aircraft with a very long history in one regime would acquire
    unbounded precision and dominate the fleet --- which is wrong,
    because its error is model bias, not noise, and model bias does not
    average out.

### 3.9 Where `ŝ_i` actually comes from

Two sources, combined, and the distinction matters for what the fleet is
really contributing:

1.  **Model-derived (prior).** `PhysicsTwin.jacobian` at the regime
    operating point. Always available, zero flight hours required, but
    carries no information the twin did not already have.

2.  **Empirically identified (the real fleet contribution).** Weighted
    regression of observed residual movement on estimated parameter
    movement over a sliding window:

        Δz_t(r) ≈ Σ_j ŝ_i(θ_j, r) · Δθ̂_j,t + noise

    ridge-regularised toward the model-derived prior, with `σ_reg²`
    taken from the regression covariance. Requires accumulated flight
    time and genuine parameter movement.

The fleet fuses (2), shrunk toward (1). Stating this plainly in the
paper pre-empts the obvious reviewer question --- *if everything is
model-derived, what does the fleet add?* The answer is: the fleet
supplies empirical operating-point diversity that no single aircraft's
history contains, and corrects the twin's shape where reality disagrees
with it.

### 3.10 Robustness

-   **Trimming.** Drop the highest and lowest `α̂_i` contributions per
    cell when fleet size ≥ 7.
-   **Precision cap.** No single aircraft may contribute more than 30 %
    of `Σ 1/v_i` for a cell.
-   **Faulted-unit exclusion.** An aircraft whose own gate is `NAMED`
    with a large \|θ̂ − 1\| contributes to `u_j` only for parameters
    *other* than its own active fault.
-   **Staleness.** Contributions older than `T_stale` (default 200
    flight hours) are decayed by a forgetting factor, so the fleet shape
    tracks fleet-wide wear rather than freezing at as-new.

### 3.11 Messages

Uplink, per aircraft, per fusion round:

``` python
@dataclass(frozen=True)
class FingerprintContribution:
    aircraft_id: str            # opaque; no tail number, no position, no route
    model_version: str
    regime_grid_version: str
    theta_index: int            # j
    info_vector: np.ndarray     # float32[n_regime]  = s_hat_i / v_i
    info_scalar: np.ndarray     # float32[n_regime]  = 1 / v_i
    n_seconds: np.ndarray       # uint32[n_regime]   = n_i(r), for audit only
    epoch: int
```

Downlink, per fusion round:

``` python
@dataclass(frozen=True)
class FleetShape:
    theta_index: int
    u: np.ndarray               # float32[n_regime], unit norm, sign-fixed
    var_u: np.ndarray           # float32[n_regime]
    contributors: np.ndarray    # uint16[n_regime]
    alpha_median: float
    epoch: int
    model_version: str
    regime_grid_version: str
```

Size: with `n_θ = 12` and `n_regime = 12`, one uplink is \~1.7 kB ---
**G2.7** satisfied with margin. **No raw flight data, no positions, no
timestamps of manoeuvres leave the aircraft.** For a defence programme
this is not incidental; it is the reason a fleet-level scheme is
deployable at all, and it should be stated in the pitch.

### 3.12 Operating modes

  -----------------------------------------------------------------------
  Mode                                Behaviour
  ----------------------------------- -----------------------------------
  **Connected**                       contributions uplinked each round;
                                      `FleetShape` refreshed

  **Disconnected**                    last-known `FleetShape` used from
                                      local cache; `epoch` age surfaced
                                      in the GCS; all admissibility rules
                                      unchanged

  **Never-connected**                 `u_j^prior` from the twin only;
                                      borrowing permitted but
                                      `λ_shrink = 1` and the GCS labels
                                      attribution `MODEL-SHAPE`, not
                                      `FLEET`
  -----------------------------------------------------------------------

Borrowing works with the datalink down. That is the whole point --- it
is a cached shape, not a live query.

------------------------------------------------------------------------

## 4. Interfaces

``` python
@dataclass(frozen=True)
class AttributionFrame:
    t: float
    verdict: Verdict                 # NAMED | AMBIGUOUS | BORROWED | INVALID
    theta_hat: np.ndarray            # n_theta
    crlb: np.ndarray                 # n_theta, inflated on borrowed rows
    named: int | None                # theta index, if NAMED/BORROWED
    ambiguous_set: tuple[int, ...]   # populated if AMBIGUOUS
    cos_matrix: np.ndarray           # n_theta x n_theta, |cos| upper triangle
    borrowed_regimes: tuple[int,...]
    r2: float | None                 # guardrail value, if a transfer was attempted
    alpha_hat: float | None
    fleet_epoch: int | None
    reason: ReasonCode               # why refused, or why a borrow was rejected (A1..A6)
```

``` python
class IdentifiabilityGate:
    def update(self, rf: ResidualFrame) -> AttributionFrame: ...
    def fingerprint(self, theta_index: int) -> np.ndarray: ...
    def try_borrow(self, theta_index: int, shape: FleetShape) -> BorrowResult: ...
    def probe_request(self) -> ProbeRequest | None:
        """Emitted when verdict == AMBIGUOUS and no admissible borrow exists.
        Consumed by L7 (Stage 3)."""

class FleetNode:
    def ingest(self, c: FingerprintContribution) -> None: ...
    def fuse(self, theta_index: int) -> FleetShape: ...
```

**Ordering contract:** the gate attempts a borrow *before* raising a
probe request. A probe costs engine life and mission time; a borrow
costs nothing. L7 must never be asked to fly a manoeuvre for a regime
the fleet could have supplied admissibly.

------------------------------------------------------------------------

## 5. Test plan

### 5.1 Unit / property

-   `R²_i == cos²(f_i, u_j|R_i)` to 1e-9 for random inputs --- the
    identity in §3.6 is asserted in code, not just in prose.
-   `|R_i| == 1` ⇒ `R² == 1` exactly ⇒ rule A1 must fire. Regression
    test, because this is the failure that would silently break the
    guardrail.
-   Sign convention: fusing `{ŝ_i}` and `{−ŝ_i}` yields the same `u_j`.
-   Fusion equals the closed-form precision-weighted mean when all
    `α_i = 1`.
-   Fleet of one ⇒ `u_j == u_j^prior` to 1e-6.
-   Gate invariance: scaling all `f` by a positive constant leaves every
    verdict unchanged.

### 5.2 Simulated fleet (the main evidence)

Fleet of `N ∈ {1, 2, 4, 8, 16}` units. Each unit gets independently
sampled build gains `α_i ~ LogNormal(0, 0.25)` and an independently
sampled regime history --- including at least one unit with
**cruise-only** exposure, which is the demo aircraft.

Measurements:

  -----------------------------------------------------------------------
  Experiment                          Reports
  ----------------------------------- -----------------------------------
  **E1 · Resolution vs fleet size**   gate resolution rate for the
                                      cruise-only aircraft, vs N.
                                      Expected: rises and saturates; the
                                      saturation point is a deployable
                                      recommendation

  **E2 · False-borrow ROC**           sweep `τ_R2` ∈ \[0.5, 0.99\]; plot
                                      false-borrow rate vs resolution
                                      rate. **G2.6**

  **E3 · Guardrail power**            inject a *different* fault on the
                                      borrowing aircraft than the fleet
                                      shape describes; confirm R²
                                      collapses and A2 fires. Also inject
                                      two co-occurring faults, the harder
                                      case

  **E4 · Heterogeneity stress**       widen `α_i` spread until transfer
                                      breaks; report the build-tolerance
                                      range over which the claim holds

  **E5 · Honest confidence**          compare CRLB on `BORROWED` verdicts
                                      against realised error. If borrowed
                                      intervals are not calibrated, §3.5
                                      inflation is wrong

  **E6 · Baseline comparison**        unguarded estimator,
                                      guarded-local-only, guarded+fleet.
                                      Over-confident misattribution rate
                                      for each. **G2.4**
  -----------------------------------------------------------------------

### 5.3 Adversarial

-   One aircraft reports a corrupted contribution (wrong sign, 100×
    magnitude, all zeros). Confirm trimming and the precision cap
    contain the damage and the fleet shape stays usable.
-   Version skew: a contribution with a stale `model_version` must be
    rejected, not fused.

------------------------------------------------------------------------

## 6. Risks

  -----------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- -----------------------
  Shape is not actually   **Fatal to the claim**  E4 measures it directly
  unit-invariant for the                          and reports the
  ambiguous pair                                  tolerance range; if it
                                                  fails, the fleet layer
                                                  is reported as
                                                  not-supported rather
                                                  than quietly retained

  Everything is           High (reviewer          §3.9 separates prior
  model-derived, so the   question)               from empirical
  fleet adds nothing                              contribution; E1 must
                                                  show resolution rising
                                                  with N, which cannot
                                                  happen if the fleet
                                                  adds no information

  Borrowing manufactures  High                    §3.5 variance
  false confidence                                inflation + E5
                                                  calibration check

  `|R_i| = 1` makes the   High, easy to miss      Rule A1, asserted in a
  guardrail vacuous                               dedicated regression
                                                  test

  Fleet shape drifts as   Medium                  Staleness forgetting
  the whole fleet wears                           factor §3.10; `epoch`
                                                  age shown in GCS

  Nobody deploys a MALE   Certain                 Simulated fleet is the
  UAV fleet for a                                 evidence and is
  hackathon                                       labelled as such in
                                                  every figure caption
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Deliverables

1.  `estimation/` --- UKF implementing §1, with the covariance output
    the gate depends on.
2.  `gate/` --- L4 implementing §2, including the three fingerprint
    axes.
3.  `fleet/` --- L4F: contribution builder, fusion node, admissibility
    rules A1--A6.
4.  Simulated-fleet harness and the E1--E6 result set.
5.  **Two figures for the submission:** resolution-vs-fleet-size (E1)
    and the false-borrow ROC (E2). These are the direct evidence for §4
    of the pitch document.

# Modification applied in the five-stage plan

The original L3/L4/L4F specification is retained as the scientific
source of truth for state estimation, identifiability, and fleet
federation. It is now implemented as **Stage 3**.

The central contribution remains:

> The system refuses to name a subsystem when the available evidence
> cannot distinguish the competing physical hypotheses.

The UKF, three-axis fingerprint, Fisher information/CRLB, ambiguity
state, fleet shape transfer, A1--A6 admissibility rules, rank-1 fleet
fusion, uncertainty inflation, and test plan are retained.

The key implementation chain is:

``` text
ResidualFrame
     ↓
UKF
     ↓
θ̂ + Pθ + innovations
     ↓
Sensitivity fingerprint
     ↓
Local identifiability
     ↓
      ┌───────────────┐
      │               │
   identifiable    ambiguous
      │               │
      ↓               ↓
   Attribution     fleet borrow
                        │
                 ┌──────┴──────┐
              admissible     rejected
                 │               │
                 ↓               ↓
              BORROWED       PROBE_REQUEST
```

Stage 3 must produce a stable `AttributionFrame` consumed by Stage 4.
fileciteturn1file1
