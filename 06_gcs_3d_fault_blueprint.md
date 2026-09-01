# Stage 5 Addendum — GCS 3D Fault Blueprint

**SIH26054 · Replan to Learn · GCS presentation layer, detailed spec**
Companion to `gcs-evidence-console-design-spec.md` and the uploaded
`01_foundation_data_contracts.md` … `05_gcs_integration_validation_deployment.md`

---

## Purpose

The Evidence Console (the earlier design spec) gives the operator the
*numeric* truth — θ̂, CRLB, the cosine matrix. This addendum specifies
a companion view: a schematic 3D engine that gives the *spatial* truth
— exactly where, on the physical engine, the named or ambiguous fault
sits. It documents the working prototype (`engine-fault-blueprint.html`)
in full: every component's coordinates, every fault's visual rule, and
the parts of the state space it doesn't cover yet.

Same governing rule as the rest of the console applies here without
exception: **nothing highlights that isn't traceable to a field in
`AttributionFrame`.**

---

## 0. What this is, and what it is not

- It is a **schematic** representation of the Rotax 915 iS architecture
  described in Stage 1 §1.1 — four cylinders, a liquid-cooled head
  circuit, dry-sump oil system, turbocharged induction. It is not built
  from real CAD data; no such data was available, and a true CAD import
  would bury the one part that matters under thousands that don't. This
  is a documented limitation, in the same spirit as the `[ASSUMED]`
  provenance tags Stage 1 requires for every engineering estimate.
- It is a **companion** to the Diagnosis panel (Evidence Console §5.2),
  not a replacement. The panel is where an engineer reads the exact
  CRLB interval and TreeSHAP ranking; the blueprint is where anyone —
  operator or judge — sees *where* that finding lives in under a
  second.
- It does **not** render mission-level verdicts (`SAFE` / `REPLAN` /
  `ABORT`). Those belong to the Mission & Probe screen and describe the
  sortie, not a subsystem — keeping that boundary explicit avoids
  conflating "what's wrong with the engine" with "what the aircraft
  should do about it."

---

## 1. Why 3D, why transparent, why no gradients

- **Transparent housing** is the literal answer to "show the fault
  exactly": the block, cylinder walls, cooling jacket, and turbo shell
  are all rendered at low, flat opacity so the crankshaft, injectors,
  and combustion markers are visible *through* them, not hidden behind
  solid metal the way a real engine would hide them.
- **Flat, unlit materials** (`MeshBasicMaterial`, no lights in the
  scene at all) — not a stylistic choice on top of the "no gradients"
  rule from the Evidence Console spec, but the same rule applied to 3D:
  a Lambert/Phong-shaded engine would pick up soft lighting falloff
  across every curved surface, which is exactly the kind of
  manufactured polish the rest of this project refuses to apply to its
  confidence estimates. A blueprint is a flat line drawing for the same
  reason a CRLB interval is a hard bracket, not a blurred cloud.
- **Pattern over color intensity** for anything provisional: reticles
  and highlight fills use the same six semantic hex values as the
  Evidence Console (`--critical`, `--caution`, `--ambiguous`,
  `--borrowed`, `--nominal`, `--probe`), never a shade in between.

---

## 2. Scene composition — component → physics binding

Every mesh in the scene maps to either a health parameter, a sensor
bias, or a structural part with no direct θ. Coordinates are the actual
values in `engine-fault-blueprint.html`.

| Component | Geometry (Three.js) | Position (x, y, z) | Represents | Source equation |
|---|---|---|---|---|
| Crankcase block | `BoxGeometry(4.4, 1.1, 1.6)`, transparent fill + wireframe | (0, 0.55, 0) | housing, no direct θ | Stage 1 §1.1 |
| Crankshaft | `CylinderGeometry(0.09, 0.09, 4.0)`, rotated 90° | (0, 0.55, 0) | **θ_fric** | `FMEP = θ_fric·(c₀+c₁N+c₂N²)` — Stage 1 §2.4 |
| Cylinder 1–4 | `CylinderGeometry(0.32, 0.34, 1.3)` | x ∈ {−1.6, −0.55, 0.55, 1.6}, y=1.75 | `T_cyl[i]` head node | Stage 1 §2.1, §2.6 |
| Cylinder head | `BoxGeometry(0.62, 0.22, 0.62)` | (xᵢ, 2.5, 0) | head cap, no direct θ | Stage 1 §2.6 |
| Injector marker *i* | `ConeGeometry(0.07, 0.22)` | (xᵢ, 2.35, −0.42) | **θ_inj[i]** | `ṁf,cyl[i] = θ_inj[i]·ṁair/(4·λcmd·AFRst)` — §2.3 |
| EGT probe marker *i* | `SphereGeometry(0.06)` | (xᵢ, 2.1, 0.5) | **b_EGT[i]** | `EGT_i = T_exh,i − k_probe·(…) + b_EGT[i]` — §2.5 |
| Combustion marker *i* | `SphereGeometry(0.10)` | (xᵢ, 2.03, 0) | **θ_comb** | `η_ind = θ_comb·η_ind,0(λ,N,p_im)` — §2.4 |
| Cooling jacket | `BoxGeometry(4.8, 0.55, 1.0)`, transparent + wireframe | (0, 2.75, 0) | **θ_cool** | `C_hd·dT_hd/dt = … − θ_cool·UA_hd·(T_hd−T_amb) …` — §2.6 |
| CHT sensor | `SphereGeometry(0.07)` | (2.1, 2.75, 0.42) | **b_CHT** | offset from the jacket deliberately — sensor bias ≠ the physical cooling path (Stage 3 §2.3 cross-modal axis) |
| Oil pan | `BoxGeometry(3.8, 0.4, 1.3)` | (0, −0.2, 0) | sump, no direct θ | §2.7 |
| Oil pump marker | `CylinderGeometry(0.16, 0.16, 0.3)`, rotated 90° | (−2.1, 0, 0.5) | **θ_oilp** | `p_oil = θ_oilp·[k_pump·N·μ(T_oil)/(k_leak+k_clear·N)] + b_poil` — §2.7 |
| *(not yet built)* oil pressure sensor marker | — | offset from pump | **b_poil** | same equation, additive term — see §8 |
| Exhaust segments ×4 + rear pipe | `CylinderGeometry` | per cylinder → z=1.35 rear line | exhaust path, no direct θ | §2.5 |
| Turbo | `SphereGeometry(0.28)`, transparent + wireframe | (2.35, 1.15, 1.35) | boost path — **deliberately not a θ** | §2.3: "turbo/wastegate is not modelled dynamically… a boost-control fault will present as an unexplained input, not as a θ" |
| Intake manifold | `CylinderGeometry(0.14, 0.14, 2.0)`, rotated 90° | (−2.75, 1.0, −0.62) | **θ_vol** | `ṁair = θ_vol·ηvol(N,p_im)·Vd·(N/120)·p_im/(R_air·T_im)` — §2.3 |

---

## 3. Complete fault taxonomy → visual rule

Every entry in the Stage 1 §2.2 health-parameter vector, mapped to a
target and a default rendering color. Items marked **built** are wired
to a button in the current prototype; **documented** are specified here
for the next pass.

| j | Symbol | Nominal | Admissible range | 3D target | Color | Status |
|---|---|---|---|---|---|---|
| 1 | θ_vol | 1.0 | [0.80, 1.05] | intake manifold | `--critical` | built |
| 2 | θ_comb | 1.0 | [0.80, 1.05] | combustion markers ×4 | `--critical` alone / `--ambiguous` paired with θ_cool | built (ambiguous case) |
| 3 | θ_cool | 1.0 | [0.70, 1.10] | cooling jacket | `--critical` alone / `--ambiguous` paired / `--borrowed` if fleet-resolved | built |
| 4 | θ_inj[1..4] | 1.0 each | [0.80, 1.10] | injector marker, matching cylinder index | `--critical` | built (cyl 3 example; generalizes to any index) |
| 5 | θ_oilp | 1.0 | [0.75, 1.05] | oil pump marker | `--critical` | built |
| 6 | θ_fric | 1.0 | [0.95, 1.30] | crankshaft | `--critical` | built |
| 7a | b_EGT[1..4] | 0.0 | ±3σ | EGT probe marker, matching index | `--caution` | built (probe 2 example) |
| 7b | b_CHT | 0.0 | ±3σ | CHT sensor marker | `--caution` | built |
| 7c | b_poil | 0.0 | ±3σ | oil pressure sensor marker | `--caution` | documented, not built — §8 |

**Generalization note.** The prototype demonstrates one index per class
(injector 3, EGT probe 2) rather than building 8 near-identical buttons.
In production this isn't a button lookup at all — `applyScenario()` is
replaced by a function that reads `AttributionFrame.named` directly and
indexes into the same `cylinders[]` / marker arrays already built. No
new geometry is needed to cover the remaining three injectors or three
EGT probes; only the binding function changes. See §7.

---

## 4. Verdict rendering — the full state matrix

`AttributionFrame.verdict` and the Stage 5 §7 safety states, each with
an explicit, distinct rule. The point of keeping these visually
distinct is the same point the spec makes about the GCS panel: an
ambiguous refusal must never look like a low-confidence guess, and a
model fault must never look like a hardware fault.

| State | Source | Rendering rule | Color | Built? |
|---|---|---|---|---|
| `NAMED` | AttributionFrame.verdict | single target solid-highlighted, reticle + label | `--critical` | ✓ |
| `AMBIGUOUS` | AttributionFrame.verdict | **every** candidate in `ambiguous_set` highlighted simultaneously, own reticle each | `--ambiguous` | ✓ |
| `BORROWED` | AttributionFrame.verdict | same single target as NAMED, label shows R² + fleet epoch instead of CRLB | `--borrowed` | ✓ |
| `INVALID` | AttributionFrame.verdict | whole engine desaturated (all fills → `--ink-400` at low opacity, no reticles) | neutral | documented |
| `DEGRADED_INPUT` | ResidualFrame.status | highlight the **input's** marker (e.g. intake/turbo for MAP/TPS dropout), never a θ | `--caution` | ✓ (boost example) |
| `MODEL_SATURATED` | ResidualFrame.status | whole-engine wireframe pulses once, no single part singled out — a clamp event invalidates the *step*, not one subsystem | `--caution` | documented |
| `ANOMALOUS_UNKNOWN` | L5 novelty class | dashed outline pulse around the whole engine, **no marker highlighted** — by construction this class has no known location | `--ambiguous` | documented |
| `ABSTAIN` / `PROBE_REQUIRED` | Gate + planner | same as `AMBIGUOUS`, plus a small badge on the label: "probe available — Nh, X% life" | `--ambiguous` + `--probe` badge | documented |

**Why `INVALID` and `ANOMALOUS_UNKNOWN` never light up a specific
part.** Lighting a component implies the system knows where to look.
Both states are, by definition, cases where it doesn't — rendering a
whole-engine treatment instead of guessing a location is the 3D
equivalent of the panel rule "never render an ambiguous candidate as a
definitive subsystem."

---

## 5. Simultaneous faults

Stage 5 §7 requires testing simultaneous faults explicitly. The
blueprint handles this by construction rather than as a special case:
each active fault owns its own target list, reticle set, and label.
Two concurrent findings — say `NAMED θ_inj[3]` and `AMBIGUOUS
cooling|combustion` — render as two independent highlight groups in
their own colors, at their own (naturally non-overlapping) 3D
locations, with two labels. Nothing about the rendering path assumes a
single active fault; `applyScenario()` in the prototype clears and
re-applies one scenario at a time only because the demo drives it from
buttons — the production binding (§7) accepts a *list* of findings per
frame instead of a single key.

---

## 6. Interaction & camera model

Exact math, for anyone extending the prototype:

**Camera** — spherical coordinates around a fixed look-at target:

```
target = (0, 1.3, 0)
camera.x = target.x + radius·sin(phi)·sin(theta)
camera.y = target.y + radius·cos(phi)
camera.z = target.z + radius·sin(phi)·cos(theta)
```

Defaults: `radius=9.5`, `theta=π/3.6`, `phi=π/2.7`. Drag updates
`theta -= dx·0.006` and `phi -= dy·0.006`, clamped to `[0.35, π−0.35]`
so the camera can't flip past the poles. Scroll/pinch updates `radius`,
clamped to `[4.5, 18]`.

**Reticle pulse** — every active target's ring scales by
`1 + 0.12·sin(t·0.003)`, `t` in milliseconds — a slow, even pulse, not
a decorative bounce.

**Label placement** — screen-space projection of the active targets'
centroid each frame:

```
ndc = worldPosition.clone().project(camera)
screenX = (ndc.x·0.5 + 0.5) · canvasWidth
screenY = (−ndc.y·0.5 + 0.5) · canvasHeight
```

No dependency beyond Three.js core (r128) — no OrbitControls addon, no
CSS3DRenderer for labels, just a plain absolutely-positioned `<div>`
repositioned every frame.

---

## 7. Live data binding — from demo buttons to a real stream

The prototype's `applyScenario(key)` is a stand-in for what a deployed
GCS would call once per `AttributionFrame`. Sketch of the real binding:

```js
function renderAttribution(frame) {
  // frame: AttributionFrame — verdict, named, ambiguous_set, crlb,
  //        r2, fleet_epoch, reason

  const findings = [];

  if (frame.verdict === 'NAMED' || frame.verdict === 'BORROWED') {
    findings.push({
      target: paramToMesh(frame.named),      // index → the same
                                              // marker/jacket/pump/etc
                                              // objects built in §2
      color: frame.verdict === 'BORROWED' ? colors.borrowed : colors.critical,
      label: describeNamed(frame)            // "θ_inj[3]=0.88, CRLB[0.84,0.92]"
                                              // or "BORROWED · R²=0.94 · epoch 214"
    });
  }

  if (frame.verdict === 'AMBIGUOUS') {
    frame.ambiguous_set.forEach(j => findings.push({
      target: paramToMesh(j),
      color: colors.ambiguous,
      label: 'AMBIGUOUS — |cos| = ' + frame.reason.cos.toFixed(2)
    }));
  }

  applyFindings(findings);   // generalized version of applyScenario():
                              // clears previous highlights, applies N
                              // findings instead of exactly one
}

function paramToMesh(thetaIndex) {
  // one lookup table, built once from §2/§3 — no new geometry required
  // to support every injector or every EGT probe, only this mapping.
}
```

`paramToMesh` is the only new piece of code needed to move from "one
example injector, one example EGT probe" to the full parameter vector —
everything else in §2 already exists for all four cylinders.

---

## 8. Known limitations (stated plainly)

- **Schematic, not CAD-accurate.** Proportions and spacing are chosen
  for legibility, not measured from the real Rotax 915 iS.
- **b_poil has no marker yet.** The oil pressure sensor bias shares the
  pump's equation but is a distinct parameter (Stage 1 §2.2 item 7) and
  should get its own marker offset from the pump, the same way `b_CHT`
  is offset from the cooling jacket.
- **`INVALID`, `MODEL_SATURATED`, `ANOMALOUS_UNKNOWN`, and
  `ABSTAIN/PROBE_REQUIRED`** are specified in §4 but not yet wired to a
  button in the running prototype — `NAMED`, `AMBIGUOUS`, `BORROWED`,
  and one `DEGRADED_INPUT` example are.
- **No click-to-inspect on the model itself.** Fault selection is
  currently button-driven, not raycast-from-mouse-click on a part. See
  §9.
- **One example per parameter class**, not the full 4-injector /
  4-probe set, per the generalization note in §3.

---

## 9. Extension roadmap

1. Add `b_poil` marker and wire the four remaining `INVALID` /
   `MODEL_SATURATED` / `ANOMALOUS_UNKNOWN` / `ABSTAIN` rendering rules
   from §4.
2. Replace `paramToMesh` demo stub with the real lookup table so any
   injector or EGT index highlights correctly, not just the two
   examples.
3. Add `THREE.Raycaster`-based click-to-inspect: clicking a part in the
   3D view opens the Evidence Drawer (Evidence Console §6) for that
   parameter, the same way clicking a number does on the 2D panels.
4. Bind `renderAttribution()` (§7) to the real replay/telemetry stream
   instead of demo buttons, so the blueprint updates live during Demo 1
   of the required demonstrations.
5. If CAD or reference art for the Rotax 915 iS becomes available,
   swap the primitive geometry in §2 for it while keeping the highlight
   / reticle / label system unchanged — the mapping table is the
   contract, not the geometry.
