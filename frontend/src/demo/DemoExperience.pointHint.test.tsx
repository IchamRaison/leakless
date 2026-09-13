// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import DemoExperience from "./DemoExperience";

vi.mock("./SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("takes the viewer from the locked recordings to the measurement point choice", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => new Promise(() => {})),
  );
  const scroll = vi.fn();
  Element.prototype.scrollIntoView = scroll;
  render(<DemoExperience />);
  fireEvent.click(
    screen.getByRole("button", { name: /select a measurement point first/ }),
  );
  expect(screen.getByRole("button", { name: "Measurement N1" })).toHaveFocus();
  expect(scroll).toHaveBeenCalled();
  // Still locked: choosing a point remains the only way to enable them.
  expect(screen.getByRole("button", { name: /No-leak/ })).toBeDisabled();
});
