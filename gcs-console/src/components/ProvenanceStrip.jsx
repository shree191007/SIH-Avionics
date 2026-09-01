import { useConsole } from '../context/ConsoleContext';
import { provenance } from '../mock/data';

export default function ProvenanceStrip() {
  const { plainEnglish, setPlainEnglish } = useConsole();
  return (
    <div className="provenance-strip mono">
      <div className="provenance-fields">
        <span>dataset:{provenance.dataset}</span>
        <span>model:{provenance.model}</span>
        <span>regime:{provenance.regime}</span>
        <span>{provenance.latencyMs}ms/frame</span>
      </div>
      <label className="plain-english-toggle">
        <span>Plain-English</span>
        <input
          type="checkbox"
          checked={plainEnglish}
          onChange={(e) => setPlainEnglish(e.target.checked)}
          aria-label="Toggle plain-English captions"
        />
      </label>
    </div>
  );
}
