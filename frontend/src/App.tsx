import { lazy, Suspense } from "react";

const DemoExperience = lazy(() => import("./demo/DemoExperience"));

export function App() {
  return (
    <Suspense fallback={<div className="app-loading">Loading LeakLess…</div>}>
      <DemoExperience />
    </Suspense>
  );
}
