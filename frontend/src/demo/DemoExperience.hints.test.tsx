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

it("explains why the demo recordings are disabled until a point is selected", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => new Promise(() => {})),
  );
  render(<DemoExperience />);
  expect(
    screen.getByText("select a measurement point first"),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /No-leak/ })).toBeDisabled();
  fireEvent.click(
    screen.getByRole("button", { name: "Select measurement N2" }),
  );
  expect(
    screen.queryByText("select a measurement point first"),
  ).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /No-leak/ })).toBeEnabled();
});
