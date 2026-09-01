const COLOR = {
  NAMED: 'var(--critical)',
  AMBIGUOUS: 'var(--ambiguous)',
  BORROWED: 'var(--borrowed)',
  INVALID: 'var(--ink-400)',
  DEGRADED_INPUT: 'var(--caution)',
  MODEL_SATURATED: 'var(--caution)',
  ANOMALOUS_UNKNOWN: 'var(--ambiguous)',
  ABSTAIN: 'var(--ambiguous)',
  PROBE_REQUIRED: 'var(--ambiguous)',
  HEALTHY: 'var(--nominal)',
  TRACKING: 'var(--nominal)',
  DEGRADING: 'var(--caution)',
  STALE: 'var(--ink-400)',
  MONITORING: 'var(--caution)',
};

const GLYPH = {
  NAMED: '●',
  AMBIGUOUS: '◐',
  BORROWED: '◆',
  INVALID: '○',
  DEGRADED_INPUT: '▲',
  MODEL_SATURATED: '▲',
  ANOMALOUS_UNKNOWN: '◇',
  ABSTAIN: '◐',
  PROBE_REQUIRED: '◐',
  HEALTHY: '●',
  TRACKING: '●',
  DEGRADING: '▲',
  STALE: '◌',
  MONITORING: '◔',
};

// Never a confidence indicator next to an AMBIGUOUS verdict — this badge
// renders the enum only, never a percentage.
export default function VerdictBadge({ verdict, suffix }) {
  const color = COLOR[verdict] || 'var(--ink-400)';
  return (
    <span className="badge" style={{ color }}>
      <span aria-hidden="true">{GLYPH[verdict] || '○'}</span>
      {verdict}
      {suffix ? <span style={{ color: 'var(--ink-400)', textTransform: 'none' }}>{suffix}</span> : null}
    </span>
  );
}
