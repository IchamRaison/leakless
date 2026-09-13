// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { App } from "./App";

vi.mock("./demo/SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
afterEach(() => {
  cleanup();
  location.hash = "";
});

it("renders LeakLess as the only experience, including for the former studio hash", async () => {
  location.hash = "#studio";
  render(<App />);
  expect(
    await screen.findByRole(
      "heading",
      { name: /Water damage/ },
      { timeout: 5000 },
    ),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("link", { name: /Load a recording/ }),
  ).toHaveAttribute("href", "#inspect");
  expect(
    screen.getByRole("heading", { name: "Inspect another recording." }),
  ).toBeInTheDocument();
  const text = document.body.textContent ?? "";
  expect(text).not.toMatch(/PIPE|Acoustic studio|Safoan/i);
  expect(text).not.toMatch(/Forme d|Fréquence|Amplitude numérique/);
});

it("opens the monitor replay on #monitor", async () => {
  location.hash = "#monitor";
  render(<App />);
  expect(
    await screen.findByText(/Water signal · last 120 s/, {}, { timeout: 5000 }),
  ).toBeInTheDocument();
});
