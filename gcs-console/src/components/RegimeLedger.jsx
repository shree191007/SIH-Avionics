// █ solid = flown locally, ▨ hatch = borrowed from fleet, ░ dotted-outline = never observed.
// One visual grammar reused on Diagnosis, Fleet, and Probe screens.
const STYLE = {
  flown: { char: '████', color: 'var(--nominal)', opacity: 1 },
  borrowed: { char: '▨▨▨▨', color: 'var(--borrowed)', opacity: 0.9 },
  never: { char: '░░░░', color: 'var(--ink-400)', opacity: 0.5 },
};

export default function RegimeLedger({ entries, compact = false }) {
  return (
    <div className="regime-ledger" role="table" aria-label="Regime Ledger">
      {entries.map((e) => {
        const s = STYLE[e.status];
        return (
          <div
            key={e.regime}
            className="regime-cell mono"
            style={{ color: s.color, opacity: s.opacity }}
            title={`${e.regime}: ${e.status}`}
          >
            {!compact && <div className="regime-cell-label">{e.regime}</div>}
            <div>{s.char}</div>
          </div>
        );
      })}
    </div>
  );
}
