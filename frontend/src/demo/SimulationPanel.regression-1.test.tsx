// @vitest-environment jsdom
// Regression: ISSUE-003 — on a 1280×720 screen the alert preview and WhatsApp button opened below the fold
// Found by /qa on 2026-09-13
// Report: .gstack/qa-reports/qa-report-127-0-0-1-5188-2026-09-13-simulation.md
import "@testing-library/jest-dom/vitest";
import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { SimulationPanel } from "./SimulationPanel";
import { simulateIncident } from "./simulation";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("brings the panel into view for each new incident, not on every render", () => {
  const scroll = vi.fn();
  Element.prototype.scrollIntoView = scroll;
  const first = simulateIncident({ x: 700, y: 70 });
  const { rerender, container } = render(
    <SimulationPanel incident={first} onClear={() => {}} />,
  );
  expect(scroll).toHaveBeenCalledTimes(1);
  expect(scroll.mock.contexts[0]).toBe(container.querySelector(".sim-panel"));
  expect(scroll.mock.calls[0][0]).toMatchObject({ block: "nearest" });
  rerender(<SimulationPanel incident={{ ...first }} onClear={() => {}} />);
  expect(scroll).toHaveBeenCalledTimes(1);
  rerender(
    <SimulationPanel
      incident={simulateIncident({ x: 1000, y: 240 })}
      onClear={() => {}}
    />,
  );
  expect(scroll).toHaveBeenCalledTimes(2);
});
