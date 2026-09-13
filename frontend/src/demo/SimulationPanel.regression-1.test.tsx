// @vitest-environment jsdom
// Regression: ISSUE-003 — on a 1280×720 screen the alert state opened below the fold
// Found by /qa on 2026-09-13
// Report: .gstack/qa-reports/qa-report-127-0-0-1-5188-2026-09-13-simulation.md
import "@testing-library/jest-dom/vitest";
import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { Incident } from "./incidents";
import { SimulationPanel } from "./SimulationPanel";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const incident = (x: number, y: number): Incident => ({
  incident_id: "inc-1",
  source: "injected_test",
  strongest_sensor: 3,
  zone: "N3",
  position: { x, y },
  started_at: 1_789_300_000,
  status: "ANOMALY_PENDING",
  confirmed_at: null,
  alert_sent_at: null,
  resolved_at: null,
  transport: null,
  delivery_status: "not_dispatched",
  telegram_message_id: null,
  failure_reason: null,
});

it("brings the panel into view for each new incident, not on every render", () => {
  const scroll = vi.fn();
  Element.prototype.scrollIntoView = scroll;
  const props = { elapsedS: 3, persistenceS: 30, onResolve: () => {} };
  const { rerender, container } = render(
    <SimulationPanel incident={incident(700, 70)} {...props} />,
  );
  expect(scroll).toHaveBeenCalledTimes(1);
  expect(scroll.mock.contexts[0]).toBe(container.querySelector(".sim-panel"));
  expect(scroll.mock.calls[0][0]).toMatchObject({ block: "nearest" });
  rerender(
    <SimulationPanel incident={incident(700, 70)} {...props} elapsedS={4} />,
  );
  expect(scroll).toHaveBeenCalledTimes(1);
  rerender(<SimulationPanel incident={incident(1000, 240)} {...props} />);
  expect(scroll).toHaveBeenCalledTimes(2);
});
