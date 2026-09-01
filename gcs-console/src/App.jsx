import { useState } from 'react';
import { ConsoleProvider } from './context/ConsoleContext';
import { ScenarioProvider } from './context/ScenarioContext';
import ProvenanceStrip from './components/ProvenanceStrip';
import ScenarioBar from './components/ScenarioBar';
import EvidenceDrawer from './components/EvidenceDrawer';
import LiveModal from './components/LiveModal';
import Overview from './screens/Overview';
import './App.css';

// One central window: the Overview dashboard is the whole app now.
// Every screen still exists as a component — Overview's tiles expand
// them in a modal — there's just no more page-per-screen navigation.
export default function App() {
  const [liveOpen, setLiveOpen] = useState(false);
  return (
    <ConsoleProvider>
      <ScenarioProvider>
        <div className="console-shell">
          <ProvenanceStrip />
          <ScenarioBar onOpenLive={() => setLiveOpen(true)} />
          <div className="console-body">
            <main className="console-canvas">
              <Overview />
            </main>
            <EvidenceDrawer />
          </div>
          {liveOpen && <LiveModal onClose={() => setLiveOpen(false)} />}
        </div>
      </ScenarioProvider>
    </ConsoleProvider>
  );
}
