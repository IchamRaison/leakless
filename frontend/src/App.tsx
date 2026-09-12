import { lazy, Suspense, useEffect, useRef, useState } from "react";

const DemoExperience = lazy(() => import("./demo/DemoExperience"));
const MonitorReplay = lazy(() => import("./demo/MonitorReplay"));
// Review notes: dev server only (?notes). Stripped from production builds.
const NotesLayer = import.meta.env.DEV
  ? lazy(() => import("./notes/NotesLayer"))
  : null;

const isMonitor = () => location.hash === "#monitor";

export function App() {
  const [monitor, setMonitor] = useState(isMonitor);
  const wasMonitor = useRef(isMonitor());
  useEffect(() => {
    // Only a switch between views resets scroll; in-page anchors (#signal…) keep the browser's jump.
    const change = () => {
      const next = isMonitor();
      if (wasMonitor.current && !next) window.scrollTo(0, 0);
      wasMonitor.current = next;
      setMonitor(next);
    };
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  const notes = NotesLayer && new URLSearchParams(location.search).has("notes");
  return (
    <Suspense fallback={<div className="app-loading">Loading LeakLess…</div>}>
      {monitor ? <MonitorReplay /> : <DemoExperience />}
      {notes && NotesLayer && (
        <Suspense fallback={null}>
          <NotesLayer />
        </Suspense>
      )}
    </Suspense>
  );
}
