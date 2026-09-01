// Discrete color steps, not a continuous gradient — this is a diagnostic
// overview, not a verdict, so it stays in neutral ink shades rather than
// the six semantic colors.
function bucket(v) {
  if (v === null) return null;
  if (v >= 0.9) return { step: 3, shade: '#EDEFF0' };
  if (v >= 0.75) return { step: 2, shade: '#8B99A6' };
  if (v >= 0.5) return { step: 1, shade: '#5A6672' };
  return { step: 0, shade: '#2A333D' };
}

export default function CosineHeatmap({ params, matrix }) {
  const n = params.length;
  const cell = 20;
  return (
    <div className="cosine-heatmap">
      <svg width={cell * n + 90} height={cell * n + 20}>
        {params.map((p, i) => (
          <text key={`r${i}`} x={85} y={20 + i * cell + cell / 2 + 3} fontSize="9" textAnchor="end" fill="var(--ink-400)" className="mono">{p}</text>
        ))}
        {Array.from({ length: n }).map((_, i) =>
          Array.from({ length: n }).map((_, j) => {
            if (j <= i) return null;
            const v = matrix[i][j];
            const b = bucket(v);
            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={90 + j * cell} y={20 + i * cell} width={cell - 1} height={cell - 1}
                  fill={b.shade}
                  data-field={`cos(${params[i]}, ${params[j]})`}
                />
                <title>{`|cos(${params[i]}, ${params[j]})| = ${v}`}</title>
              </g>
            );
          })
        )}
      </svg>
      <div className="cosine-legend mono">
        <span><i style={{ background: '#2A333D' }} />&lt;0.5</span>
        <span><i style={{ background: '#5A6672' }} />0.5–0.75</span>
        <span><i style={{ background: '#8B99A6' }} />0.75–0.9</span>
        <span><i style={{ background: '#EDEFF0' }} />≥0.9</span>
      </div>
    </div>
  );
}
