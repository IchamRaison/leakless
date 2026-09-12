// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const health = {
  status: "ok",
  schema_version: "0.1",
  device: "cpu",
  models: ["tslm", "baseline"].map((name) => ({
    name,
    available: false,
    version: null,
    reason: "Adaptateur et poids non livrés",
  })),
  capabilities: {
    upload: true,
    visualization: true,
    perturbation: false,
    replay: false,
  },
};
const sample = (id: string) => ({
  sample_id: id.repeat(24),
  input_sha256: id.repeat(64),
  duration_seconds: 1,
  sample_rate_hz: 16000,
  channels: 1,
  source: "Import local",
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
    times: [0],
    frequencies_hz: [0],
    power_db: [[-120]],
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
  notice: `Visual ${id}`,
});
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Studio application", () => {
  it("shows empty state and never offers an invented prediction", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (path: string) =>
        json(path.endsWith("/health") ? health : []),
      ),
    );
    render(<App />);
    expect(await screen.findByText("API locale connectée")).toBeInTheDocument();
    expect(
      screen.getByText("Tout commence par un signal."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Analyser le signal" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Révéler le label" }),
    ).toBeDisabled();
    expect(screen.getByText("Modèle indisponible")).toBeInTheDocument();
  });

  it("rejects oversized uploads before transfer and surfaces server errors", async () => {
    const fetchMock = vi.fn(async (path: string) =>
      json(path.endsWith("/health") ? health : []),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await screen.findByText("API locale connectée");
    const large = new File(["x"], "secret_name.wav", { type: "audio/wav" });
    Object.defineProperty(large, "size", { value: 9 * 1024 * 1024 });
    fireEvent.change(screen.getByLabelText("Importer un fichier WAV"), {
      target: { files: [large] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Fichier limité à 8 Mio.",
    );
    expect(fetchMock).toHaveBeenCalledTimes(2);
    fetchMock.mockResolvedValueOnce(
      json(
        {
          error: {
            code: "invalid_audio",
            message: "WAV illisible ou corrompu.",
          },
        },
        422,
      ),
    );
    fireEvent.change(screen.getByLabelText("Importer un fichier WAV"), {
      target: { files: [new File(["bad"], "leak.wav")] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "WAV illisible ou corrompu.",
    );
    expect(screen.queryByText("leak.wav")).not.toBeInTheDocument();
  });

  it("ignores a visualization arriving after the user selects a different clip", async () => {
    let resolveFirst: (value: Response) => void = () => {};
    vi.stubGlobal(
      "fetch",
      vi.fn((path: string) => {
        if (path.endsWith("/health")) return Promise.resolve(json(health));
        if (path.endsWith("/samples"))
          return Promise.resolve(json([sample("a"), sample("b")]));
        if (path.includes("a".repeat(24)))
          return new Promise<Response>((resolve) => {
            resolveFirst = resolve;
          });
        return Promise.resolve(json(visual("b")));
      }),
    );
    render(<App />);
    const picker = await screen.findByLabelText("Enregistrement");
    fireEvent.change(picker, { target: { value: "a".repeat(24) } });
    await screen.findByText("Calcul des visuels…");
    fireEvent.change(picker, { target: { value: "b".repeat(24) } });
    await screen.findByText(/Visual b/);
    resolveFirst(json(visual("a")));
    await waitFor(() =>
      expect(screen.queryByText(/Visual a/)).not.toBeInTheDocument(),
    );
    expect(screen.getByText("b".repeat(64))).toBeInTheDocument();
  });

  it("reports an unavailable API and can reconnect", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch"));
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "API inaccessible",
    );
    fetchMock.mockImplementation(async (path: string) =>
      json(path.endsWith("/health") ? health : []),
    );
    fireEvent.click(screen.getByRole("button", { name: /API non connectée/ }));
    expect(await screen.findByText("API locale connectée")).toBeInTheDocument();
  });
});
