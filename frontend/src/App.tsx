import { lazy, Suspense, useEffect, useState } from "react";

const DemoExperience = lazy(() => import("./demo/DemoExperience"));
const MonitorReplay = lazy(() => import("./demo/MonitorReplay"));
// Review notes: dev server only (?notes). Stripped from production builds.
const NotesLayer = import.meta.env.DEV
  ? lazy(() => import("./notes/NotesLayer"))
  : null;

const isMonitor = () => location.hash === "#monitor";

export function App() {
  const [monitor, setMonitor] = useState(isMonitor);
  useEffect(() => {
    const change = () => {
      setMonitor(isMonitor());
      if (!isMonitor()) window.scrollTo(0, 0);
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
