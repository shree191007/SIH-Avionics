// GitHub-style pass/fail grid over the randomized checks, grouped by
// scenario type — pure visualization of results already produced.
export default function SafetyHeatCalendar({ rows, highlight }) {
  return (
    <div className="heat-calendar">
      {rows.map((row) => {
        const failCount = row.cells.filter((c) => c === 'fail').length;
        const isHighlighted = row.scenario === highlight;
        return (
          <div key={row.scenario} className="heat-calendar-row" style={isHighlighted ? { outline: '1px solid var(--borrowed)', outlineOffset: 3 } : undefined}>
            <div className="heat-calendar-label mono">{row.scenario}</div>
            <div className="heat-calendar-cells">
              {row.cells.map((c, i) => (
                <span
                  key={i}
                  className="heat-cell"
                  style={{ background: c === 'pass' ? 'var(--nominal)' : 'var(--critical)' }}
                  title={c}
                />
              ))}
            </div>
            <div className="heat-calendar-status mono" style={{ color: failCount ? 'var(--critical)' : 'var(--nominal)' }}>
              {failCount ? `${failCount} FAIL` : 'all pass'}
            </div>
          </div>
        );
      })}
    </div>
  );
}
