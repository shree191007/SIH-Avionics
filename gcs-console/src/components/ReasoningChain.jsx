import Field from './Field';

// "Gate said this, ML model said this, physics engine says this" — the
// causal trace behind a verdict, one step per actor in the pipeline, in
// the order they actually fired. Every step is a Field: click it and the
// Evidence Drawer shows the same field/equation/value it always does —
// this widget narrates order and handoff, it doesn't add new evidence.
export default function ReasoningChain({ steps }) {
  return (
    <div className="reasoning-chain" role="list" aria-label="Reasoning chain">
      {steps.map((s, i) => (
        <div key={s.id} className="reasoning-step" role="listitem">
          <div className="reasoning-rail">
            <span className="reasoning-dot" style={{ background: s.color, borderColor: s.color }} />
            {i < steps.length - 1 && <span className="reasoning-connector" />}
          </div>
          <Field
            block
            className="reasoning-body"
            label={s.title}
            field={s.field}
            equation={s.equation}
            value={s.value}
          >
            <div className="reasoning-actor mono" style={{ color: s.color }}>{s.actor}</div>
            <div className="reasoning-title">{s.title}</div>
            <div className="reasoning-statement">{s.statement}</div>
            <div className="reasoning-outcome mono">→ {s.outcome}</div>
          </Field>
        </div>
      ))}
    </div>
  );
}
