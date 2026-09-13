// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import MonitorReplay from "./MonitorReplay";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: vi.fn(
    async (_path: string, schema: { parse: (v: unknown) => unknown }) =>
      schema.parse({
        server_time: Date.now() / 1000,
        persistence_seconds: 30,
        transport: { name: "telegram", configured: false },
        active: null,
        history: [],
      }),
  ),
}));
afterEach(cleanup);

it("shows the live sensor monitor: chart, table and network, without alerts or scores", async () => {
  const { container } = render(<MonitorReplay />);
  expect(
    screen.getByRole("heading", { name: "Water signal · last 120 s" }),
  ).toBeInTheDocument();
  expect(await screen.findByText("NORMAL")).toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: "Continuous monitoring" }),
  ).toBeInTheDocument();
  expect(document.body.textContent).not.toMatch(
    /NO PER-RECORDING OUTPUT|Model output|synthetic|TELEGRAM_/,
  );
  expect(
    screen.getAllByRole("img", {
      name: /of the three sensors, last 120 seconds/,
    }),
  ).toHaveLength(2);
  const rows = container.querySelectorAll(".sensor-table tbody tr");
  expect(
    [...rows].map((row) => row.querySelector(".sensor-status")?.textContent),
  ).toEqual(["healthy", "healthy", "healthy"]);
  expect(
    screen.getByRole("group", { name: /Illustrative pipe network/ }),
  ).toBeInTheDocument();
  expect(
    screen.getByText(/ripple size = live RMS of each sensor/),
  ).toBeInTheDocument();
  expect(container.querySelector(".sim-panel")).toBeNull();
  expect(document.body.textContent).not.toMatch(
    /\d+(\.\d+)?\s*%|probability|Replay, not a live feed|Dataset label/i,
  );
});
