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

vi.mock("./SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
vi.mock("../SignalView", () => ({ SignalView: () => <div>signal view</div> }));
vi.mock("./loadExample", () => ({ loadExample: vi.fn() }));

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

it("lets a building point replay a chosen real recording without associating it", async () => {
  render(<DemoExperience />);
  fireEvent.click(
    screen.getByRole("button", { name: "Select measurement N1" }),
  );
  const panel = screen.getByRole("complementary", {
    name: "Illustrative measurement point",
  });
  expect(within(panel).getByText("N1")).toBeInTheDocument();
  expect(loadExample).not.toHaveBeenCalled();
  expect(within(panel).getByText("Choose a recording.")).toBeInTheDocument();

  fireEvent.click(within(panel).getByRole("button", { name: "REC 02" }));
  expect(vi.mocked(loadExample).mock.calls[0][0]).toBe(recordings.records[1]);
  expect(
    await within(panel).findByRole("img", { name: "Waveform of REC 02" }),
  ).toBeInTheDocument();
  expect(within(panel).getByText("Leak-associated")).toBeInTheDocument();
  expect(
    within(panel).getByText("TSLM · NOT EVALUATED YET"),
  ).toBeInTheDocument();
  expect(panel.textContent).not.toMatch(/%|probability|detected|located/i);

  fireEvent.click(
    within(panel).getByRole("button", { name: /Inspect this recording/ }),
  );
  await vi.waitFor(() => expect(location.hash).toBe("#monitor/rec-02"));
});

it("opens the monitor with the requested recording in front and lets markers change it", async () => {
  location.hash = "#monitor/rec-02";
  render(<MonitorReplay />);
  const leak = await screen.findByRole("article", {
    name: "Dataset label Leak-associated",
  });
  expect(leak).toHaveClass("is-focused");
  // jsdom does not expose roles on SVG <g>; the browser QA checks the accessible button.
  const marker = document.querySelector('[aria-label="Show REC 03 in front"]');
  expect(marker).toHaveAttribute("role", "button");
  fireEvent.click(marker!);
  expect(
    await screen.findByRole("article", {
      name: "Dataset label Environmental noise",
    }),
  ).toHaveClass("is-focused");
  expect(leak).not.toHaveClass("is-focused");
  expect(location.hash).toBe("#monitor/rec-03");
  expect(screen.getByText("None generated")).toBeInTheDocument();
  expect(document.querySelector(".monitor audio")).not.toHaveAttribute("src");
});
