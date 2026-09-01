# GCS Evidence Console — Frontend Walkthrough

**Audience:** the agent/engineer wiring the real Stage 1–5 backend
(`PhysicsTwin`, `UKF`, `IdentifiabilityGate`, `FleetShape`, the
planner) into this frontend. Everything below describes what exists
today, in plain terms, so integration doesn't require re-reading every
component file. The frontend is 100% mock data right now — nothing
here talks to a real process. Your job is to replace the data source,
not the UI.

Stack: React 19 + Vite + Recharts, plain CSS (no Tailwind/CSS-in-JS),
no router, no state library beyond React Context. `npm run dev` from
this directory serves it on `:5173`.

---

## 1. The one-sentence architecture

**One page (`Overview`), one fault at a time, one context that owns
which fault is active and how far it's progressed — every screen and
tile just reads derived fields off that context.** There used to be a
10-page nav; it's gone. There's no router.

```
App.jsx
 ├─ ConsoleProvider      (Evidence Drawer + Plain-English toggle state)
 └─ ScenarioProvider      (which fault is active + simulation clock + liveFrame)
     ├─ ProvenanceStrip   (dataset/model/regime strip, top)
     ├─ ScenarioBar       (hotkeys 1-6, play/pause/replay, progress ticker)
     ├─ Overview           (the single dashboard, laid out as a 3-column grid)
     │   └─ Tile → Modal → <Screen/>   (click "expand" → full-detail screen in a modal)
     └─ EvidenceDrawer     (right-side panel, opens when any number is clicked)
```

Physical layout of `Overview` (`grid-template-areas` in `App.css`):

```
┌──────────────┬──────────────────────┬──────────────┐
│ 1 · 3 · 4 · 7│                      │ 2 Diagnosis   │
│ Engine Health,│   9 · 3D Blueprint    │ & Evidence    │
│ Prognostics,  │   (always live,      │ (reasoning    │
│ Mission &     │    center, no modal) │  chain)       │
│ Safety (one   │                      │               │
│ tall panel,   │                      │               │
│ 4 stacked     │                      │               │
│ subsections,  │                      │               │
│ spans both    ├──────────────────────┤               │
│ rows)         │      5 · Fleet       │               │
└──────────────┴──────────────────────┴──────────────┘
```

Panels 6 (Replay & Six-Arm Eval) and 8 (Deployment Ladder) were removed
from the UI entirely — their screen files, mock data, and the
`liveFrame.replayExample` field are gone. If they come back, treat them
as new work, not a "re-enable" (see §9). Panel 7 (Safety) is no longer
its own tile either — it's a fourth subsection stacked inside the left
panel, same pattern as 1/3/4.

The remaining 7 "screens" (`src/screens/*.jsx`) are not routes — they're
just components. Every one except `Blueprint.jsx` is rendered twice: a
condensed version inline in its `Tile` (or, for panels 1 and 3, inside
one shared tile with two labeled subsections, each with its own
"expand" button), and the full version inside a `Modal` when you click
"expand". `Blueprint.jsx` is the exception — it only renders once,
directly in the center column, never in a modal (see §5).

---

## 2. The two contexts

### `ConsoleContext` (`src/context/ConsoleContext.jsx`)
Trivial, not your concern for backend integration: holds the
Plain-English caption toggle and the Evidence Drawer's open/closed
state + contents (`{ field, equation, value }`). Any component can
call `openDrawer(field, equation, value)` via the `<Field>` wrapper
component to populate it. Nothing here needs backend data.

### `ScenarioContext` (`src/context/ScenarioContext.jsx`) — **this is the integration seam**
This owns:
- `scenarios` — the static list of 6 fault definitions (`mock/scenarios.js`)
- `scenario` — whichever one is currently selected (by hotkey 1-6 or a click)
- `progress` — a float 0→1, driven by a `setInterval` clock (150ms tick, 12s total duration)
- `liveFrame` — **the actual object every screen reads.** Computed each
  tick by `buildLiveFrame(scenario, progress)` (`mock/interpolate.js`),
  which interpolates the static `scenario` from a "nothing's wrong yet"
  state up to its final values as `progress` goes 0→1.

This is the mock simulation. **Right now, "the backend" is a fake
clock ticking forward and lerping between two hardcoded endpoints.**
Your integration replaces that clock+interpolator with real frames
arriving from the actual pipeline. Section 6 below is the concrete plan.

---

## 3. The `LiveFrame` shape — the actual wire contract

Every screen and tile reads `liveFrame`, not `scenario`, for anything
that should animate/update live. This is the shape your backend needs
to be able to produce, one way or another (see §6 for how):

```js
liveFrame = {
  id: string,                 // scenario id, e.g. "theta_inj3"
  label: string,               // short display label
  blueprintKey: string,        // which 3D Blueprint scenario-button this maps to (§5)
  progress: number,            // 0..1, how far through this frame's "development" we are
  resolved: boolean,           // progress >= 1 — true once the verdict has actually landed

  verdict: 'HEALTHY' | 'MONITORING' | 'NAMED' | 'AMBIGUOUS' | 'DEGRADED_INPUT',
  verdictSuffix: string,       // human-readable one-liner shown next to the verdict badge

  cos: number | null,          // |cos(θ_a, θ_b)| once the gate step has fired, else null
  named: string | undefined,        // e.g. "theta_inj[3]" — only set once resolved
  ambiguous_set: string[] | undefined, // e.g. ["theta_cool","theta_comb"] — only once resolved

  health: {
    status: 'HEALTHY' | 'DEGRADING',
    cylZ: [number,number,number,number] | null,  // EGT z-score per cylinder, or null if suppressed
    warnCyl: number | null,    // 1-4, which cylinder (if any) is the named culprit
    note: string,
  },

  rul: {
    p05: number, median: number, p95: number,   // hours
    status: 'TRACKING' | 'STALE',
    note: string,
  },

  mission: {
    risk: { sortie: 'LOW'|'MEDIUM'|'HIGH', endurance: 'LOW'|'MEDIUM'|'HIGH' },
    note: string,
    probeNeeded: boolean,
  },

  fleet: {
    attempted: boolean,
    r2: number | null,
    admissible: boolean | null,
    note: string,
  },

  chain: [ ReasoningStep & { revealed: boolean } ],  // see §4

  safetyCategory: string | null,      // matches a row in the Safety heat-calendar, or null
}
```

Nothing outside `mock/interpolate.js` and `ScenarioContext.jsx` needs
to know *how* this object is produced — every consumer just destructures
`liveFrame.health`, `liveFrame.rul`, etc. **That's the seam: if you
make something else produce an object shaped like this every ~150ms
(or on every real event), the rest of the app doesn't change.**

---

## 4. The Reasoning Chain — `chain[]`

This is the "gate said X, UKF said Y, physics said Z" trace
(`src/components/ReasoningChain.jsx`, rendered on the Diagnosis screen
and the Overview tile). Each scenario in `mock/scenarios.js` defines a
`chain` array of 3-5 steps; `buildLiveFrame` adds a `revealed: boolean`
to each one based on `progress` (step `i` of `n` reveals at
`progress >= (i+1)/n`). The UI only renders steps where `revealed` is
true — that's what makes the chain "grow" during the simulation instead
of appearing all at once.

```js
ReasoningStep = {
  id: 'physics' | 'ukf' | 'gate' | 'fleet' | 'verdict',   // fleet is optional, others aren't
  actor: string,        // "PhysicsTwin", "UKF", "Identifiability Gate", "Fleet Shape", "Verdict"
  title: string,
  statement: string,    // the actual finding, in prose
  field: string,         // dotted path into the real dataclass, e.g. "ResidualFrame.spatial"
  equation: string,      // the governing equation, shown in the Evidence Drawer
  value: string | number,
  color: string,         // CSS var, e.g. 'var(--caution)'
  outcome: string,       // "→ ..." handoff line to the next actor
}
```

**For a real backend**, each step's `field`/`equation`/`value` should
be sourced directly from the corresponding dataclass field
(`ResidualFrame.spatial`, `UKF.theta_hat`, `AttributionFrame.reason.cos`,
`IdentifiabilityGate.try_borrow(...).r2`, `AttributionFrame.verdict`) —
the mock already names the real field for every step, see
`mock/scenarios.js`.

---

## 5. The 3D Blueprint — `src/screens/Blueprint.jsx` + `public/engine-fault-blueprint.html`

The 3D engine view is a **separate, standalone HTML/Three.js file**,
embedded via `<iframe src="/engine-fault-blueprint.html">`. It is not
a React component and has its own internal state. The console drives
it entirely through `postMessage`:

```js
// Blueprint.jsx, on every liveFrame.resolved / liveFrame.blueprintKey change:
iframe.contentWindow.postMessage({ type: 'setScenario', key: liveFrame.blueprintKey }, '*');
// while liveFrame.resolved is false:
iframe.contentWindow.postMessage({ type: 'clear' }, '*');
```

`blueprintKey` must be one of the keys already wired inside
`engine-fault-blueprint.html`'s own scenario list (search for
`data-key` in that file) — e.g. `named_theta_vol`, `named_theta_fric`,
`named_theta_oilp`, `named_inj3`, `amb_cool_comb`, `degraded_boost`,
plus documented-but-not-yet-mapped ones (`borrowed_cool`, `invalid`,
`saturated`, `anomalous`, `abstain`, `named_egt2`, `named_cht`,
`named_poil`, `named_inj1`, `simultaneous`). If the backend can
produce a fault type not in this list, either add a new
`data-key`/scenario entry inside `engine-fault-blueprint.html` (the
`paramToMesh` lookup table already covers every θ/bias in the health
vector — see that file's own `PARAM_META`), or extend `renderAttribution()`
in that file directly, which is the documented "real binding" path
(`06_gcs_3d_fault_blueprint.md` §7).

The iframe detects `window.self !== window.top` to know it's embedded:
when embedded, its own demo picker panel is hidden and it does nothing
until it receives a `postMessage` — it never free-runs. When opened
standalone (`demos/engine-fault-blueprint.html`, or this file's own
public URL), the picker is fully interactive for demoing the 3D view
on its own.

**Important:** two identical copies of this file exist —
`gcs-console/public/engine-fault-blueprint.html` and
`demos/engine-fault-blueprint.html` (repo root). Keep them in sync if
you edit either; the console only ever loads the `public/` copy.

---

## 6. How to actually wire in the real backend

Three viable approaches, cheapest first:

### Option A — swap the data source, keep the simulation clock
Leave `ScenarioContext`'s `setInterval` clock as-is (it's just driving
a visual "evidence accumulating" reveal), but instead of interpolating
between two hardcoded endpoints in `mock/scenarios.js`, fetch or
subscribe to a real `AttributionFrame`/`ResidualFrame` for the
selected fault and use its *actual* values as the interpolation
target. Minimal change: `SCENARIOS` entries stop being hardcoded
objects and become `async` fetches or a WebSocket subscription keyed
by `hotkey`/`id`, resolved once before `buildLiveFrame` needs them.

### Option B — replace the clock with real event timestamps (recommended for a real demo)
This is the more honest integration, and closer to what
`06_gcs_3d_fault_blueprint.md` §7's `renderAttribution(frame)` already
sketches for the 3D view. Instead of a fake 0→1 progress bar:

1. Backend streams real frames (WebSocket or SSE) as they're produced
   — `ResidualFrame` each cycle, `AttributionFrame` when the gate/fleet
   actually run, particle-filter/RUL updates, planner output.
2. Frontend accumulates these into a `liveFrame`-shaped object
   *directly* — no interpolation needed, because the real pipeline is
   already the thing "developing over time." `chain` steps get
   `revealed: true` the moment their real event arrives, not on a
   fixed fraction of a fake duration.
3. `ScenarioContext.liveFrame` becomes `useState` fed by the stream
   instead of `useMemo(buildLiveFrame(...))`. Everything downstream
   (`Diagnosis.jsx`, `Overview.jsx`, `EngineHealth.jsx`, etc.) is
   unchanged, because they only ever read `liveFrame`.
4. `progress`/`resolved` become: `resolved = liveFrame.verdict !== 'MONITORING'`,
   `progress` can just drive the `ScenarioBar` ticker off wall-clock
   time since the fault was first flagged (cosmetic only at that point).

### Option C — full replay mode
`buildRulFunnel`, `buildEgtSeries` (`mock/data.js`) and the funnel/EGT
chart "reveal by slicing" trick exist so a chart can show a trace
growing over real time without re-fetching every frame. If the
backend can serve a historical window (last N samples) per request,
these two generator functions can be replaced 1:1 with real arrays of
`{t, predicted, actual}` / `{t, p05, median, p95}` and nothing else
in `EngineHealth.jsx` / `Prognostics.jsx` needs to change — they
already just slice/map over whatever array they're given.

**Whichever option you pick, the contract that must not break is the
`liveFrame` shape in §3.** Every screen was written against that shape,
not against `mock/scenarios.js` directly (the only exceptions are a
few places that intentionally read the *final* `scenario` object for
stable, non-animated metadata — search each screen for `scenario.` vs
`liveFrame.` to see exactly which).

---

## 7. Static (non-scenario) mock data — `src/mock/data.js`

These are **not** wired to the active fault at all; they're
either fleet-wide/session-wide context, or illustrative option pools.
Listed here so you know what's still fake and can decide whether it's
in scope:

| Export | Used by | What it represents |
|---|---|---|
| `regimeTimeline`, `regimeEncounters` | Engine Health | flight regime history — real source: `RegimeClassifier` / `REGIME_GRID_V1` (Stage 1) |
| `spatialDecomposition` | Engine Health | fixed example of `ResidualFrame.spatial` — not scenario-driven, always shows the cyl-3 ambiguous example |
| `cosMatrix`, `THETA_PARAMS` | Diagnosis (Cosine Heatmap) | full pairwise `AttributionFrame.cos_matrix` — real source exists, just not scenario-swapped |
| `crlbTrail` | Diagnosis | fixed illustrative CRLB widening trend — real source: `AttributionFrame.crlb` history |
| `resolvedByLog` | Diagnosis | fixed illustrative log — real source: attribution-basis field per Stage 5 GCS contract |
| `regimeLedgerCool` | Diagnosis, Fleet | fixed flown/borrowed/never grid for θ_cool specifically |
| `missionAlternatives`, `lifeBudget`, `probeCandidates`, `probeRanked` | Mission | pre-simulated planner output — real source: L7 planner (Stage 4) `simulate()` / candidate scoring |
| `fleetShape`, `guardrailCandidates` | Fleet | real source: `FleetShape`, `IdentifiabilityGate.try_borrow()` |
| `safetyHeatCalendar` | Safety | real source: the 10⁴ randomized-check results (Stage 5 §7) |
| `provenance` | Provenance Strip (top bar) | dataset/model/regime/latency — trivial to wire to real values |

None of these need to change for the console to "work" — they're
scoped out of the live-simulation feature on purpose (see prior
decisions in this repo's history), but they're obvious next targets if
you want more of the console driven by real data.

---

## 8. Component quick-reference

| File | Reads from `liveFrame`? | Notes |
|---|---|---|
| `screens/Diagnosis.jsx` | ✅ verdict, cos, chain, verdictSuffix | also reads static `cosMatrix`/`crlbTrail`/`resolvedByLog` |
| `screens/EngineHealth.jsx` | ✅ health | chart series generated once per **scenario** (`buildEgtSeries`, seeded RNG) then revealed by `liveFrame.progress` — see §6 Option C |
| `screens/Fleet.jsx` | ✅ fleet | guardrail-preview default selection reads final `scenario.named`/`ambiguous_set`, not live |
| `screens/Prognostics.jsx` | ✅ rul | funnel chart rebuilt from `liveFrame.rul` every tick via `buildRulFunnel` |
| `screens/Mission.jsx` | ✅ mission | probe table/scatter only shown when `mission.probeNeeded` |
| `screens/Safety.jsx` | ✅ safetyCategory | highlights a heat-calendar row when set |
| `screens/Blueprint.jsx` | ✅ resolved, blueprintKey | drives the iframe, see §5. Only ever rendered inline in the center column now — never in a modal |
| `components/ScenarioBar.jsx` | reads `progress`/`playing`, not `liveFrame` fields | the hotkey/play/pause/replay control surface |
| `components/EvidenceDrawer.jsx` | ❌ | generic — shows whatever `{field, equation, value}` was last clicked via `<Field>` |
| `components/VerdictBadge.jsx` | — | pure presentational; color/glyph map needs a new entry if you introduce a new verdict enum value |

---

## 9. Things that are deliberately *not* animated/live — or not present at all

- **Fleet constellation** (dot plot) and **epoch/contributors count** — fleet-wide, not per-fault.
- **Deployment Ladder and Replay & Six-Arm Eval were removed from the UI**
  (not just hidden — `screens/Deployment.jsx` and `screens/Replay.jsx`
  are deleted, and `mock/data.js`'s `deploymentLevels`,
  `currentDeploymentLevel`, `sixArm`, `determinism` exports, plus
  `mock/scenarios.js`'s `replayExample` field, are gone). Deployment
  was a certification/maturity level, never fault-dependent by design.
  Six-Arm was an aggregate evaluation-harness result, not per-fault. If
  either comes back, it's new work — there's nothing to "re-enable."

Don't feel obligated to wire up the remaining static bits (fleet
constellation); they were explicitly scoped out in favor of the
fault-attribution path (Engine Health → Diagnosis → Fleet →
Prognostics → Mission → Safety → Blueprint), which is the actual
product story this console tells.

---

## 10. File index

```
gcs-console/
├── public/engine-fault-blueprint.html   3D scene, standalone + embeddable (§5)
├── src/
│   ├── App.jsx                          shell: providers + layout
│   ├── App.css                          all layout/component CSS (token system in index.css)
│   ├── index.css                        design tokens (--critical, --caution, etc.)
│   ├── context/
│   │   ├── ConsoleContext.jsx           drawer + plain-english toggle
│   │   └── ScenarioContext.jsx          ★ integration seam — simulation clock + liveFrame
│   ├── mock/
│   │   ├── scenarios.js                 ★ the 6 fault definitions (final-state endpoints)
│   │   ├── interpolate.js               ★ buildLiveFrame() — replace/extend per §6
│   │   └── data.js                      everything NOT scenario-dependent (§7)
│   ├── components/                      shared widgets (Field, VerdictBadge, ReasoningChain, …)
│   └── screens/                         one file per tile/modal (§8)
└── WALKTHROUGH.md                       this file
```

★ = the three files that matter for backend integration. Everything
else is presentation and shouldn't need to change.
