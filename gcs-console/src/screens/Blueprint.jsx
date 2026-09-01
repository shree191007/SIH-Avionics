import { useRef, useEffect } from 'react';
import { useScenario } from '../context/ScenarioContext';

// Reuses the standalone Three.js prototype built for 06_gcs_3d_fault_blueprint.md
// rather than re-implementing the scene in React — the mapping table in that
// file is the contract, not the geometry (§9), so the geometry stays as-is.
// The active console-wide simulation is pushed into the iframe via
// postMessage: while the fault is still developing (liveFrame not yet
// resolved) the engine stays clear, exactly like the rest of the console
// — nothing highlights until AttributionFrame actually names it.
export default function Blueprint() {
  const iframeRef = useRef(null);
  const { liveFrame } = useScenario();

  const sync = () => {
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    if (liveFrame.resolved) win.postMessage({ type: 'setScenario', key: liveFrame.blueprintKey }, '*');
    else win.postMessage({ type: 'clear' }, '*');
  };

  useEffect(sync, [liveFrame.resolved, liveFrame.blueprintKey]);

  return (
    <div className="screen blueprint-screen">
      <iframe
        ref={iframeRef}
        title="Engine Fault Blueprint"
        src="/engine-fault-blueprint.html"
        className="blueprint-iframe"
        onLoad={sync}
      />
    </div>
  );
}
