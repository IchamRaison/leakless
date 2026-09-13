// @vitest-environment jsdom
// Regression: ISSUE-001 — leaving #monitor remounted the demo and lost the selected point, recording and upload
// Found by /qa on 2026-09-12
// Report: .gstack/qa-reports/qa-report-127-0-0-1-5178-2026-09-12.md
import "@testing-library/jest-dom/vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { App } from "./App";

vi.mock("./demo/SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  location.hash = "";
});

const goTo = async (hash: string) => {
  await act(async () => {
    location.hash = hash;
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
};

it("keeps the demo state and scroll position across a monitor round trip, and pauses its audio", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => new Promise(() => {})),
  );
  const scrollTo = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  const pause = vi
    .spyOn(HTMLMediaElement.prototype, "pause")
    .mockImplementation(() => {});
  render(<App />);
  fireEvent.click(
    await screen.findByRole(
      "button",
      { name: "Select measurement N3" },
      { timeout: 5000 },
    ),
  );
  expect(
    screen.getByRole("heading", { name: /Selected measurement point: N3/ }),
  ).toBeInTheDocument();
  const demoAudio = document.createElement("audio");
  document.querySelector(".demo-view")!.append(demoAudio);
  Object.defineProperty(window, "scrollY", { value: 640, configurable: true });

  await goTo("#monitor");
  expect(
    await screen.findByText(/Water signal · last 120 s/, {}, { timeout: 5000 }),
  ).toBeInTheDocument();
  expect(document.querySelector(".demo-view")).not.toBeVisible();
  expect(pause).toHaveBeenCalled();

  await goTo("#demo");
  expect(
    screen.getByRole("heading", { name: /Selected measurement point: N3/ }),
  ).toBeVisible();
  expect(scrollTo).toHaveBeenLastCalledWith(0, 640);
});
