import { useEffect, useState, type RefObject } from "react";
import { Maximize2, Minimize2 } from "lucide-react";

/** Native Fullscreen API on one panel; hidden where the browser does not allow it. */
export function FullscreenButton({
  target,
  label,
  className = "quiet-button",
}: {
  target: RefObject<HTMLElement | null>;
  label: string;
  className?: string;
}) {
  const [active, setActive] = useState(false);
  useEffect(() => {
    const change = () =>
      setActive(
        !!target.current && document.fullscreenElement === target.current,
      );
    document.addEventListener("fullscreenchange", change);
    return () => document.removeEventListener("fullscreenchange", change);
  }, [target]);
  if (!document.fullscreenEnabled) return null;
  return (
    <button
      className={className}
      aria-label={
        active ? `Exit fullscreen ${label}` : `Show ${label} fullscreen`
      }
      onClick={() =>
        void (active
          ? document.exitFullscreen()
          : target.current?.requestFullscreen())
      }
    >
      {active ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
      {active ? "Exit fullscreen" : "Fullscreen"}
    </button>
  );
}
