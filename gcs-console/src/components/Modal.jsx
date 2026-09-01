import { useEffect } from 'react';

// Expands a tile's full-detail screen in place, over the dashboard —
// the "central window" stays the dashboard; this is how you drill in
// without leaving it for a separate page.
export default function Modal({ title, onClose, children }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label={title}>
        <div className="modal-header">
          <span className="mono">{title}</span>
          <button onClick={onClose} aria-label="Close">✕ close</button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}
