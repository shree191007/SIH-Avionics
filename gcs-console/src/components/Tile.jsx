export default function Tile({ title, num, onExpand, children, area, className = '' }) {
  return (
    <div className={`tile ${className}`} style={area ? { gridArea: area } : undefined}>
      <div className="tile-header">
        <span className="mono tile-num">{num}</span>
        <span className="tile-title">{title}</span>
        {onExpand && <button className="tile-expand" onClick={onExpand}>expand ⤢</button>}
      </div>
      <div className="tile-body">{children}</div>
    </div>
  );
}
