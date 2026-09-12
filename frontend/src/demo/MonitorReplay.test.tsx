// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { loadExample } from "./loadExample";
import MonitorReplay from "./MonitorReplay";
import recordings from "./recordings.json";

vi.mock("./loadExample", () => ({ loadExample: vi.fn() }));
const visualization = (level: number) => ({
  sample_id: "a".repeat(24),
  input_sha256: "a".repeat(64),
  visualization_version: "test",
  duration_seconds: 1,
  waveform: { times: [0, 0.5], min: [-0.1, -0.2], max: [0.1, 0.2] },
  spectrogram: {
    times: [0.25, 0.75],
    frequencies_hz: [100, 2000, 3900],
    power_db: [
      [level, level],
      [level, level],
      [level, level],
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
});
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("replays measured values from the real recordings without alerts or scores", async () => {
  vi.mocked(loadExample).mockImplementation(async (record) => {
    if (record.id === "environmental-noise")
      throw new Error("Recording integrity check failed.");
    const index = recordings.records.indexOf(record);
    return {
      record,
      sample: {} as never,
      visualization: visualization(-50 - index * 5),
    };
  });
  render(<MonitorReplay />);
  expect(screen.getByRole("note")).toHaveTextContent(
    "Replay, not a live feed.",
  );
  const leak = await screen.findByRole("article", {
    name: "Dataset label Leak-associated",
  });
  expect(within(leak).getByText("Level").nextElementSibling).toHaveTextContent(
    "-55.0 dB",
  );
  expect(
    within(leak).getByText("TSLM · NOT EVALUATED YET"),
  ).toBeInTheDocument();
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "integrity check failed",
  );
  expect(screen.getByText("N/A · replay only")).toBeInTheDocument();
  expect(
    screen.getByRole("group", { name: /Illustrative pipe network/ }),
  ).toBeInTheDocument();
  expect(screen.getByText(/no leak is shown/)).toBeInTheDocument();
  expect(document.body.textContent).not.toMatch(/\d+(\.\d+)?\s*%|probability/i);
});
