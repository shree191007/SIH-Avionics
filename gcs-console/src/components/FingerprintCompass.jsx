// The gate's cosine separability drawn as two vectors on a polar plot
// across the 12 regime bins — the literal geometry the gate computes.
export default function FingerprintCompass({ labelA = 'θ_cool', labelB = 'θ_comb', angleA = 20, angleB = 32, bins = 12 }) {
  const size = 220, cx = size / 2, cy = size / 2, r = 88;
  const toXY = (deg, len) => {
    const rad = (deg * Math.PI) / 180;
    return [cx + Math.cos(rad) * len, cy - Math.sin(rad) * len];
  };
  const ticks = Array.from({ length: bins }, (_, i) => (360 / bins) * i);
  const [ax, ay] = toXY(angleA, r);
  const [bx, by] = toXY(angleB, r);

  return (
    <svg width={size} height={size} role="img" aria-label="Fingerprint Compass">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--line-500)" strokeWidth="1" />
      <circle cx={cx} cy={cy} r={r * 0.5} fill="none" stroke="var(--line-500)" strokeWidth="1" opacity="0.5" />
      {ticks.map((deg) => {
        const [x1, y1] = toXY(deg, r * 0.96);
        const [x2, y2] = toXY(deg, r);
        return <line key={deg} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--ink-400)" strokeWidth="1" />;
      })}
      <line x1={cx} y1={cy} x2={ax} y2={ay} stroke="var(--critical)" strokeWidth="2" />
      <circle cx={ax} cy={ay} r="4" fill="var(--critical)" />
      <line x1={cx} y1={cy} x2={bx} y2={by} stroke="var(--borrowed)" strokeWidth="2" />
      <circle cx={bx} cy={by} r="4" fill="var(--borrowed)" />
      <text x={ax + 8} y={ay} fill="var(--critical)" fontSize="10" className="mono">{labelA}</text>
      <text x={bx + 8} y={by + 12} fill="var(--borrowed)" fontSize="10" className="mono">{labelB}</text>
    </svg>
  );
}
