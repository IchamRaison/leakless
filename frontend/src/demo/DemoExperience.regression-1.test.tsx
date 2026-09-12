// @vitest-environment jsdom
// Regression: ISSUE-002 — deep links to #signal / #evidence / #inspect stayed at the top after the lazy view loaded
// Found by /qa on 2026-09-12
// Report: .gstack/qa-reports/qa-report-127-0-0-1-5178-2026-09-12.md
import { cleanup, render } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import DemoExperience from "./DemoExperience";

vi.mock("./SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  location.hash = "";
});

it.each(["signal", "evidence", "inspect"])(
  "scrolls to #%s once the view has rendered",
  (id) => {
    location.hash = `#${id}`;
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    render(<DemoExperience />);
    expect(scroll).toHaveBeenCalledTimes(1);
    expect(scroll.mock.contexts[0]).toHaveProperty("id", id);
  },
);

it("does not scroll for the page root or the monitor route", () => {
  const scroll = vi.fn();
  Element.prototype.scrollIntoView = scroll;
  for (const hash of ["", "#demo", "#monitor"]) {
    location.hash = hash;
    render(<DemoExperience />);
    cleanup();
  }
  expect(scroll).not.toHaveBeenCalled();
});
