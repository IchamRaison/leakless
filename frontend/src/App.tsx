import { lazy, Suspense } from "react";

const DemoExperience = lazy(() => import("./demo/DemoExperience"));
// Review notes: dev server only (?notes). Stripped from production builds.
const NotesLayer = import.meta.env.DEV
  ? lazy(() => import("./notes/NotesLayer"))
  : null;

export function App() {
  const notes = NotesLayer && new URLSearchParams(location.search).has("notes");
  return (
    <Suspense fallback={<div className="app-loading">Loading LeakLess…</div>}>
      <DemoExperience />
      {notes && NotesLayer && (
        <Suspense fallback={null}>
          <NotesLayer />
        </Suspense>
      )}
    </Suspense>
  );
}
