// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { loadExample } from "./loadExample";
import DemoExperience from "./DemoExperience";
import MonitorReplay from "./MonitorReplay";
import recordings from "./recordings.json";
import { tslm } from "./official";

vi.mock("./SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
vi.mock("../SignalView", () => ({ SignalView: () => <div>signal view</div> }));
vi.mock("./loadExample", () => ({ loadExample: vi.fn() }));
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

const visualization = {
  sample_id: "a".repeat(24),
  input_sha256: "a".repeat(64),
  visualization_version: "test",
  duration_seconds: 1,
  waveform: { times: [0, 0.5], min: [-0.1, -0.2], max: [0.1, 0.2] },
  spectrogram: {
    times: [0.25, 0.75],
    frequencies_hz: [100, 2000],
    power_db: [
      [-50, -50],
      [-50, -50],
    ],
    floor_db: -120,
    reference: "test",
  },
  parameters: {
    n_fft: 2,
    hop_samples: 1,
    window: "hann",
    channels: "mono",
    resampling: false,
    pooling: "none",
  },
  notice: "test",
};
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(loadExample).mockImplementation(async (record) => ({
    record,
    sample: {
      sample_id: "a".repeat(24),
      input_sha256: "a".repeat(64),
      duration_seconds: 1,
      sample_rate_hz: 8000,
      channels: 1,
      source: "test",
      execution_mode: "development_fixture",
      warnings: [],
      label_available: false,
    },
    visualization,
  }));
});
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  location.hash = "";
});

it("keeps a building point illustrative and sends the viewer to the live monitor", async () => {
  render(<DemoExperience />);
  fireEvent.click(
    screen.getByRole("button", { name: "Select measurement N1" }),
  );
  const panel = screen.getByRole("complementary", {
    name: "Illustrative measurement point",
  });
  expect(within(panel).getByText("N1")).toBeInTheDocument();
  expect(
    within(panel).getByText("Illustrative position · does not locate a leak."),
  ).toBeInTheDocument();
  // No recording, waveform or model output next to a building position.
  expect(loadExample).not.toHaveBeenCalled();
  expect(within(panel).queryByRole("img")).not.toBeInTheDocument();
  expect(within(panel).queryByText(/REC 0|TSLM/)).not.toBeInTheDocument();
  expect(panel.textContent).not.toContain(tslm.status);
  expect(panel.textContent).not.toMatch(/%|probability|detected|association/i);

  fireEvent.click(
    within(panel).getByRole("button", {
      name: `Listen to the ${recordings.records.length} dataset recordings`,
    }),
  );
  await vi.waitFor(() => expect(location.hash).toBe("#signal"));

  fireEvent.click(
    within(panel).getByRole("button", { name: /Open the live monitor/ }),
  );
  await vi.waitFor(() => expect(location.hash).toBe("#monitor"));
});

it("highlights the requested sensor from the hash and lets network nodes change it", async () => {
  location.hash = "#monitor/sensor-02";
  const { container } = render(<MonitorReplay />);
  const row = (n: number) =>
    container.querySelectorAll(".sensor-table tbody tr")[n - 1];
  expect(row(2)).toHaveClass("is-focused");
  // jsdom does not expose roles on SVG <g>; the browser QA checks the accessible button.
  const marker = document.querySelector('[aria-label="Highlight Sensor 3"]');
  expect(marker).toHaveAttribute("role", "button");
  fireEvent.click(marker!);
  expect(row(3)).toHaveClass("is-focused");
  expect(row(2)).not.toHaveClass("is-focused");
  expect(location.hash).toBe("#monitor/sensor-03");
  expect(await screen.findByText("NORMAL")).toBeInTheDocument();
});
