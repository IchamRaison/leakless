// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "../api";
import DemoExperience from "./DemoExperience";
import recordings from "./recordings.json";
import { tslm } from "./official";

vi.mock("./SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
vi.mock("../SignalView", () => ({
  SignalView: ({ data }: { data: { sample_id: string } }) => (
    <div>Waveform {data.sample_id}</div>
  ),
}));
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: vi.fn(),
}));
const records = recordings.records;
const sample = (i: number) => ({
  sample_id: records[i].inputSha256.slice(0, 24),
  input_sha256: records[i].inputSha256,
  duration_seconds: 1,
  sample_rate_hz: 8000,
  channels: 1,
  source: "test",
  execution_mode: "development_fixture",
  warnings: [],
  label_available: false,
});
const visual = (i: number) => ({
  ...sample(i),
  visualization_version: "test",
  waveform: { times: [0], min: [0], max: [0] },
  spectrogram: {
    times: [0.5],
    frequencies_hz: [100],
    power_db: [[-60]],
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
let currentRecord = 0;
beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      currentRecord = records.findIndex((record) => record.url === url);
      return {
        ok: true,
        arrayBuffer: async () => new Uint8Array([currentRecord]).buffer,
      };
    }),
  );
  vi.stubGlobal("crypto", {
    subtle: {
      digest: vi.fn(async (_algorithm: string, data: ArrayBuffer) => {
        const hex = records[new Uint8Array(data)[0]].fileSha256;
        return new Uint8Array(
          hex.match(/../g)!.map((value) => parseInt(value, 16)),
        ).buffer;
      }),
    },
  });
  vi.mocked(api).mockImplementation(async (path) => {
    if (path === "/samples") return sample(currentRecord);
    const index = records.findIndex((record) =>
      path.includes(record.inputSha256.slice(0, 24)),
    );
    return visual(index);
  });
});
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

it("loads a real-example identity through the existing API and never requests a fake prediction", async () => {
  render(<DemoExperience />);
  expect(screen.getByRole("button", { name: /No-leak/ })).toBeDisabled();
  expect(api).not.toHaveBeenCalled();
  fireEvent.click(
    screen.getByRole("button", { name: "Select measurement N3" }),
  );
  expect(
    screen.getByRole("heading", {
      name: "Selected measurement point: N3 — illustrative building position",
    }),
  ).toBeInTheDocument();
  // A point alone never loads a recording.
  expect(api).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: /No-leak/ }));
  expect(
    await screen.findByText(`Waveform ${sample(0).sample_id}`),
  ).toBeInTheDocument();
  expect(
    screen.getAllByText("Probability leak")[0].nextElementSibling,
  ).toHaveTextContent(tslm.probability);
  expect(vi.mocked(api).mock.calls.some(([path]) => path === "/predict")).toBe(
    false,
  );
  fireEvent.click(screen.getByRole("button", { name: /Environmental noise/ }));
  expect(
    await screen.findByText(`Waveform ${sample(2).sample_id}`),
  ).toBeInTheDocument();
  expect(
    screen.queryByText(`Waveform ${sample(0).sample_id}`),
  ).not.toBeInTheDocument();
});

it("never ties a measurement point to a recording class", async () => {
  render(<DemoExperience />);
  fireEvent.click(
    screen.getByRole("button", { name: "Select measurement N1" }),
  );
  expect(api).not.toHaveBeenCalled();
  expect(
    screen.getByText("Illustrative position · does not locate a leak."),
  ).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /^REC 0/ })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: /Leak-associated/ }));
  await screen.findByText(`Waveform ${sample(1).sample_id}`);
  for (const point of ["N2", "N3", "N4"]) {
    fireEvent.click(
      screen.getByRole("button", { name: `Select measurement ${point}` }),
    );
    expect(
      screen.getByRole("button", { name: /Leak-associated/ }),
    ).toHaveAttribute("aria-pressed", "true");
  }
  expect(screen.getByText(/They do not locate leaks/)).toBeInTheDocument();
});

it("does not attach a late result to the new selected clip", async () => {
  let resolveLate: (value: unknown) => void = () => {};
  vi.mocked(api).mockImplementation(async (path) => {
    if (path === "/samples") return sample(currentRecord);
    if (path.includes(sample(1).sample_id))
      return new Promise((resolve) => {
        resolveLate = resolve;
      });
    const index = records.findIndex((record) =>
      path.includes(record.inputSha256.slice(0, 24)),
    );
    return visual(index);
  });
  render(<DemoExperience />);
  fireEvent.click(
    screen.getByRole("button", { name: "Select measurement N3" }),
  );
  fireEvent.click(screen.getByRole("button", { name: /No-leak/ }));
  await screen.findByText(`Waveform ${sample(0).sample_id}`);
  fireEvent.click(screen.getByRole("button", { name: /Leak-associated/ }));
  await waitFor(() => expect(api).toHaveBeenCalledTimes(4));
  fireEvent.click(screen.getByRole("button", { name: /Environmental noise/ }));
  await screen.findByText(`Waveform ${sample(2).sample_id}`);
  resolveLate(visual(1));
  await waitFor(() =>
    expect(
      screen.queryByText(`Waveform ${sample(1).sample_id}`),
    ).not.toBeInTheDocument(),
  );
});

it("rejects mismatched signal identities and supports an explicit retry", async () => {
  vi.mocked(api).mockResolvedValueOnce({ ...sample(0), input_sha256: "wrong" });
  render(<DemoExperience />);
  fireEvent.click(
    screen.getByRole("button", { name: "Select measurement N3" }),
  );
  fireEvent.click(screen.getByRole("button", { name: /No-leak/ }));
  const alerts = await screen.findAllByRole("alert");
  expect(alerts[0]).toHaveTextContent("does not match");
  expect(screen.queryByText(/Waveform /)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Retry recording" }));
  expect(
    await screen.findByText(`Waveform ${sample(0).sample_id}`),
  ).toBeInTheDocument();
});
