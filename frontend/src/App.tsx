import {
  lazy,
  Suspense,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

const DemoExperience = lazy(() => import("./demo/DemoExperience"));
const MonitorReplay = lazy(() => import("./demo/MonitorReplay"));
// Review notes: dev server only (?notes). Stripped from production builds.
const NotesLayer = import.meta.env.DEV
  ? lazy(() => import("./notes/NotesLayer"))
  : null;

const isMonitor = () =>
  location.hash === "#monitor" || location.hash.startsWith("#monitor/");

export function App() {
  const [monitor, setMonitor] = useState(isMonitor);
  const wasMonitor = useRef(isMonitor());
  const demoScroll = useRef(0);
  const firstRender = useRef(true);
  useEffect(() => {
    // Only a switch between views touches scroll; in-page anchors (#signal…) keep the browser's jump.
    const change = () => {
      const next = isMonitor();
      if (!wasMonitor.current && next) demoScroll.current = window.scrollY;
      wasMonitor.current = next;
      setMonitor(next);
    };
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useLayoutEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    if (monitor)
      document
        .querySelectorAll<HTMLAudioElement>(".demo-view audio")
        .forEach((audio) => audio.pause());
    // The demo stays mounted while the monitor is shown: selection, recordings and uploads survive.
    // Leaving the monitor for a section anchor (#evidence…) opens that section, not the old position.
    const id = decodeURIComponent(location.hash.slice(1));
    const section =
      !monitor && id && id !== "demo" ? document.getElementById(id) : null;
    if (section) section.scrollIntoView?.({ block: "start" });
    else window.scrollTo(0, monitor ? 0 : demoScroll.current);
  }, [monitor]);
  const notes = NotesLayer && new URLSearchParams(location.search).has("notes");
  return (
    <Suspense fallback={<div className="app-loading">Loading LeakLess…</div>}>
      <div className="demo-view" hidden={monitor}>
        <DemoExperience />
      </div>
      {/* Own boundary: loading the monitor chunk must never suspend (and re-run) the mounted demo. */}
      {monitor && (
        <Suspense
          fallback={<div className="app-loading">Loading LeakLess…</div>}
        >
          <MonitorReplay />
        </Suspense>
      )}
      {notes && NotesLayer && (
        <Suspense fallback={null}>
          <NotesLayer />
        </Suspense>
      )}
    </Suspense>
  );
}
