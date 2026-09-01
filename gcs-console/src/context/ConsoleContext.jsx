import { createContext, useContext, useState, useCallback } from 'react';

const ConsoleContext = createContext(null);

export function ConsoleProvider({ children }) {
  const [plainEnglish, setPlainEnglish] = useState(false);
  const [drawer, setDrawer] = useState(null); // { field, equation, value }

  const openDrawer = useCallback((field, equation, value) => {
    setDrawer({ field, equation, value });
  }, []);
  const closeDrawer = useCallback(() => setDrawer(null), []);

  return (
    <ConsoleContext.Provider value={{ plainEnglish, setPlainEnglish, drawer, openDrawer, closeDrawer }}>
      {children}
    </ConsoleContext.Provider>
  );
}

export function useConsole() {
  const ctx = useContext(ConsoleContext);
  if (!ctx) throw new Error('useConsole must be used within ConsoleProvider');
  return ctx;
}

// term -> one-line plain-English caption, written once, reused everywhere.
export const PLAIN_ENGLISH = {
  CRLB: 'the tightest possible uncertainty band this data can support',
  'cos': 'how visually similar two faults look to the estimator (1.0 = indistinguishable)',
  'R²': 'how well another aircraft’s fault shape predicts this one',
  'cos_matrix': 'pairwise similarity between every fault the estimator tracks',
  'CRLB trail': 'how the uncertainty band has widened or narrowed over the last frames',
  'value-of-information': 'expected diagnostic benefit of flying a probe, minus its cost',
  'ΔLife': 'engine life spent flying diagnostic probes this mission',
};

export function Caption({ term }) {
  const { plainEnglish } = useConsole();
  if (!plainEnglish || !PLAIN_ENGLISH[term]) return null;
  return <div className="caption">{PLAIN_ENGLISH[term]}</div>;
}
