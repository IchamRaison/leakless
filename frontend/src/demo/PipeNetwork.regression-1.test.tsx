// @vitest-environment jsdom
// Regression: ISSUE-001 — keyboard incident on pipes 5 and 6 landed exactly under the REC 03 / REC 02 nodes, hidden
// Found by /qa on 2026-09-13
// Report: .gstack/qa-reports/qa-report-127-0-0-1-5188-2026-09-13-simulation.md
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { PipeNetwork } from "./PipeNetwork";
import { PIPE_PATHS, projectToPipe } from "./simulation";

afterEach(cleanup);

it("places every keyboard incident on its pipe and clear of the recording markers", () => {
  const onIncident = vi.fn();
  const { container } = render(
    <PipeNetwork
      channels={["a", "b", "c"].map((id, i) => ({
        id,
        name: `REC 0${i + 1}`,
        level: null,
      }))}
      playing={false}
      onIncident={onIncident}
    />,
  );
  const recs = [...container.querySelectorAll(".listen-point .node")].map(
    (node) => ({
      x: Number(node.getAttribute("cx")),
      y: Number(node.getAttribute("cy")),
    }),
  );
  expect(recs).toHaveLength(3);
  PIPE_PATHS.forEach((_, pipe) => {
    fireEvent.keyDown(
      screen.getByRole("button", {
        name: `Simulate an incident on pipe ${pipe + 1}`,
      }),
      { key: "Enter" },
    );
    const point = onIncident.mock.calls[pipe][0];
    expect(projectToPipe(point).distance).toBeCloseTo(0, 6);
    for (const rec of recs)
      expect(
        Math.hypot(point.x - rec.x, point.y - rec.y),
      ).toBeGreaterThanOrEqual(70);
  });
});
