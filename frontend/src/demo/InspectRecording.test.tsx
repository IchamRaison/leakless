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
import { InspectRecording } from "./InspectRecording";
import { tslm } from "./official";

vi.mock("./SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
vi.mock("../SignalView", () => ({
  SignalView: ({ data }: { data: { sample_id: string } }) => (
    <div>Waveform {data.sample_id}</div>
  ),
}));

const sample = (id: string) => ({
  sample_id: id.repeat(24),
  input_sha256: id.repeat(64),
  duration_seconds: 1,
  sample_rate_hz: 16000,
  channels: 1,
  source: "Local upload · provenance not verified",
  execution_mode: "development_fixture",
  warnings: [],
  label_available: false,
});
const visual = (id: string) => ({
  sample_id: id.repeat(24),
  input_sha256: id.repeat(64),
  visualization_version: "display-stft-v1",
  duration_seconds: 1,
  waveform: { times: [0], min: [0], max: [0] },
  spectrogram: {
    times: [0.5],
    frequencies_hz: [100],
    power_db: [[-60]],
    floor_db: -120,
    reference: "test",
  },
  parameters: {
    n_fft: 1024,
    hop_samples: 256,
    window: "hann",
    channels: "mono",
    resampling: false,
    pooling: "test",
  },
  notice: "test",
});
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
const input = () => screen.getByLabelText("Upload a WAV recording");
const upload = (file: File) =>
  fireEvent.change(input(), { target: { files: [file] } });

let fetchMock: ReturnType<typeof vi.fn>;
let next = "a";
beforeEach(() => {
  next = "a";
  fetchMock = vi.fn(async (path: string, init?: RequestInit) => {
    if (init?.method === "DELETE") return new Response(null, { status: 204 });
    if (path === "/api/samples") return json(sample(next), 201);
    const id = path.split("/")[3];
    return json(visual(id[0]));
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("rejects wrong types and oversized files before transfer, and surfaces server errors", async () => {
  render(<InspectRecording paused />);
  upload(new File(["x"], "clip.mp3"));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Choose a .wav file.",
  );
  const large = new File(["x"], "secret_leak_name.wav");
  Object.defineProperty(large, "size", { value: 9 * 1024 * 1024 });
  upload(large);
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "File limited to 8 MiB.",
  );
  expect(fetchMock).not.toHaveBeenCalled();
  fetchMock.mockResolvedValueOnce(
    json(
      {
        error: {
          code: "invalid_audio",
          message: "Unreadable or corrupted WAV.",
        },
      },
      422,
    ),
  );
  upload(new File(["bad"], "leak.wav"));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Unreadable or corrupted WAV.",
  );
  expect(screen.queryByText(/leak\.wav/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Temporal Signal Map/)).not.toBeInTheDocument();
});

it("shows metadata, waveform, map and a pending model for a valid upload, without a prediction call", async () => {
  render(<InspectRecording paused />);
  upload(new File(["wav"], "hidden_leak_label.wav"));
  expect(
    await screen.findByText(`Waveform ${"a".repeat(24)}`),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: "Temporal Signal Map" }),
  ).toBeInTheDocument();
  expect(screen.getByText("Duration").nextElementSibling).toHaveTextContent(
    "1.00 s",
  );
  expect(screen.getByText("Sample rate").nextElementSibling).toHaveTextContent(
    "16 kHz",
  );
  expect(screen.getByText("Channels").nextElementSibling).toHaveTextContent(
    "Mono",
  );
  expect(
    screen.getByText("Probability leak").nextElementSibling,
  ).toHaveTextContent(tslm.probability);
  expect(screen.getByText("a".repeat(64))).toBeInTheDocument();
  const [, init] = fetchMock.mock.calls[0];
  expect(((init as RequestInit).body as FormData).get("file")).toHaveProperty(
    "name",
    "recording.wav",
  );
  expect(
    fetchMock.mock.calls.some(([path]) => String(path).includes("predict")),
  ).toBe(false);
  expect(document.body.textContent).not.toMatch(
    /\b0\.\d+\s*%|probability:\s*\d/i,
  );
});

it("releases the previous upload when replaced, and removes on request", async () => {
  render(<InspectRecording paused />);
  upload(new File(["one"], "one.wav"));
  await screen.findByText(`Waveform ${"a".repeat(24)}`);
  next = "b";
  upload(new File(["two"], "two.wav"));
  await screen.findByText(`Waveform ${"b".repeat(24)}`);
  await waitFor(() =>
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/samples/${"a".repeat(24)}`,
      expect.objectContaining({ method: "DELETE" }),
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: /Remove recording/ }));
  expect(
    await screen.findByText("Drop a short WAV recording here."),
  ).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledWith(
    `/api/samples/${"b".repeat(24)}`,
    expect.objectContaining({ method: "DELETE" }),
  );
});

it("rejects a visualization that does not match the uploaded signal", async () => {
  fetchMock.mockImplementation(async (path: string) =>
    path === "/api/samples" ? json(sample("a"), 201) : json(visual("c")),
  );
  render(<InspectRecording paused />);
  upload(new File(["wav"], "x.wav"));
  expect(await screen.findByRole("alert")).toHaveTextContent("do not match");
  expect(screen.queryByText(/Waveform/)).not.toBeInTheDocument();
});

it("reports an unreachable API", async () => {
  fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
  render(<InspectRecording paused />);
  upload(new File(["wav"], "x.wav"));
  expect(await screen.findByRole("alert")).toHaveTextContent("API unreachable");
});
