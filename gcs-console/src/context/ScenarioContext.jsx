import { createContext, useContext, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { SCENARIOS, SCENARIO_BY_HOTKEY } from '../mock/scenarios';
import { buildLiveFrame } from '../mock/interpolate';

const ScenarioContext = createContext(null);

const DURATION_MS = 12000;
const TICK_MS = 150; // ~7fps — plenty smooth for gauges, cheap enough for 9 tiles re-rendering at once

export function ScenarioProvider({ children }) {
  const [activeId, setActiveIdRaw] = useState(SCENARIOS[0].id);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(true);
  const startRef = useRef(performance.now());
  const intervalRef = useRef(null);

  const scenario = SCENARIOS.find((s) => s.id === activeId) ?? SCENARIOS[0];

  const setActiveId = useCallback((id) => {
    setActiveIdRaw(id);
    setProgress(0);
    startRef.current = performance.now();
    setPlaying(true);
  }, []);

  const selectByHotkey = useCallback((key) => {
    const s = SCENARIO_BY_HOTKEY[key];
    if (s) setActiveId(s.id);
  }, [setActiveId]);

  const restart = useCallback(() => {
    setProgress(0);
    startRef.current = performance.now();
    setPlaying(true);
  }, []);

  const togglePlay = useCallback(() => {
    setPlaying((wasPlaying) => {
      if (!wasPlaying) {
        // resuming: shift the clock so progress continues from where it paused
        setProgress((p) => { startRef.current = performance.now() - p * DURATION_MS; return p; });
      }
      return !wasPlaying;
    });
  }, []);

  useEffect(() => {
    if (!playing) return;
    intervalRef.current = setInterval(() => {
      const elapsed = performance.now() - startRef.current;
      const p = Math.min(1, elapsed / DURATION_MS);
      setProgress(p);
      if (p >= 1) { clearInterval(intervalRef.current); setPlaying(false); }
    }, TICK_MS);
    return () => clearInterval(intervalRef.current);
  }, [playing, activeId]);

  useEffect(() => {
    const onKey = (e) => {
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (SCENARIO_BY_HOTKEY[e.key]) { selectByHotkey(e.key); return; }
      if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectByHotkey, togglePlay]);

  const liveFrame = useMemo(() => buildLiveFrame(scenario, progress), [scenario, progress]);

  return (
    <ScenarioContext.Provider value={{
      scenarios: SCENARIOS, scenario, liveFrame, progress, playing, durationMs: DURATION_MS,
      setActiveId, selectByHotkey, restart, togglePlay,
    }}>
      {children}
    </ScenarioContext.Provider>
  );
}

export function useScenario() {
  const ctx = useContext(ScenarioContext);
  if (!ctx) throw new Error('useScenario must be used within ScenarioProvider');
  return ctx;
}
