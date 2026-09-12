// @vitest-environment jsdom
// Regression: ISSUE-005 — every in-page anchor (#signal, #evidence, #inspect) scrolled back to the top
// Found by /qa on 2026-09-12
// Report: .gstack/qa-reports/qa-report-127-0-0-1-5178-2026-09-12.md
import "@testing-library/jest-dom/vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { App } from "./App";

vi.mock("./demo/SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  location.hash = "";
});

const goTo = async (hash: string) => {
  await act(async () => {
    location.hash = hash;
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
};

it("keeps the browser's anchor jump for in-page links and resets scroll only when leaving the monitor", async () => {
  const scrollTo = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  render(<App />);
  await screen.findByRole("heading", { name: /Water damage/ });
  for (const anchor of ["#signal", "#evidence", "#inspect"]) await goTo(anchor);
  expect(scrollTo).not.toHaveBeenCalled();
  await goTo("#monitor");
  expect(
    await screen.findByText(/Replay, not a live feed\./),
  ).toBeInTheDocument();
  // Entering the monitor starts it at the top; leaving restores the demo position (0 here).
  expect(scrollTo).toHaveBeenCalledTimes(1);
  await goTo("#demo");
  expect(scrollTo).toHaveBeenLastCalledWith(0, 0);
  expect(scrollTo).toHaveBeenCalledTimes(2);
});
