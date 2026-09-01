import { useConsole } from '../context/ConsoleContext';

// Every number is clickable; the drawer shows the field, the equation,
// and the live value substituted in, in that order. Hard edge, no blur.
export default function EvidenceDrawer() {
  const { drawer, closeDrawer } = useConsole();
  return (
    <aside className={`evidence-drawer ${drawer ? 'open' : ''}`} aria-live="polite">
      {drawer && (
        <>
          <div className="evidence-drawer-header">
            <span className="mono">EVIDENCE</span>
            <button onClick={closeDrawer} aria-label="Close evidence drawer">✕</button>
          </div>
          <div className="evidence-drawer-body">
            <div className="evidence-row">
              <div className="evidence-label">field</div>
              <div className="mono">{drawer.field}</div>
            </div>
            {drawer.equation && (
              <div className="evidence-row">
                <div className="evidence-label">equation</div>
                <div className="mono evidence-eqn">{drawer.equation}</div>
              </div>
            )}
            <div className="evidence-row">
              <div className="evidence-label">live value</div>
              <div className="mono evidence-value">{String(drawer.value)}</div>
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
