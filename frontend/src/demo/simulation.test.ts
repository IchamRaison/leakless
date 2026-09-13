// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createElement } from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { loadExample } from "./loadExample";
import MonitorReplay from "./MonitorReplay";
import recordings from "./recordings.json";
import {
  HIT_STROKE,
  PIPE_PATHS,
  RESPONSE_LAMBDA,
  SIM_SENSORS,
  projectToPipe,
  simulateIncident,
} from "./simulation";
import {
  alertRecipient,
  assertDemoAlert,
  buildAlertMessage,
  whatsappLink,
} from "./whatsapp";

vi.mock("./loadExample", () => ({ loadExample: vi.fn() }));
vi.mock("./SceneCanvas", () => ({ SceneCanvas: () => null }));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  location.hash = "";
});

const FORBIDDEN =
  /%|probabilit|confidence|accuracy|leak detected|localization achieved|model detected/i;

describe("deterministic simulation", () => {
  it("projects a click onto the nearest pipe", () => {
    expect(projectToPipe({ x: 400, y: 262 })).toMatchObject({
      x: 400,
      y: 240,
      pipe: 0,
    });
    // Beyond the end of a pipe, the projection clamps to its end point.
    expect(projectToPipe({ x: 1120, y: 330 })).toMatchObject({
      x: 1080,
      y: 300,
      pipe: 1,
    });
  });

  it("finds the nearest simulated sensor and applies exp(-d / lambda)", () => {
    const incident = simulateIncident({ x: 505, y: 150 });
    expect(incident.position).toEqual({ x: 505, y: 160 });
    expect(incident.nearest.name).toBe("Sim Sensor 2");
    for (const { sensor, distance, response } of incident.responses) {
      expect(distance).toBeCloseTo(
        Math.hypot(sensor.x - 505, sensor.y - 160),
        9,
      );
      expect(response).toBeCloseTo(Math.exp(-distance / RESPONSE_LAMBDA), 12);
    }
    const [s1, s2, s3] = incident.responses;
    expect(s2.response).toBeGreaterThan(s1.response);
    expect(s1.response).toBeGreaterThan(s3.response);
  });

  it("gives the same result for the same click", () => {
    expect(simulateIncident({ x: 870, y: 120 })).toEqual(
      simulateIncident({ x: 870, y: 120 }),
    );
    expect(simulateIncident({ x: 870, y: 120 }).nearest.name).toBe(
      "Sim Sensor 3",
    );
  });

  it("keeps sensors on the drawn pipes and hit targets large on a phone", () => {
    for (const sensor of SIM_SENSORS)
      expect(projectToPipe(sensor).distance).toBe(0);
    // Measured in Chrome: the drawing is ~319 px wide on a 420 px screen; the hit stroke stays above 32 px.
    expect((HIT_STROKE * 319) / 1240).toBeGreaterThanOrEqual(32);
    expect(PIPE_PATHS).toHaveLength(6);
  });

  it("simulation code never imports the official TSLM result or a model", () => {
    for (const file of [
      "simulation.ts",
      "whatsapp.ts",
      "SimulationPanel.tsx",
    ]) {
      const source = readFileSync(
        join(process.cwd(), "src/demo", file),
        "utf8",
      );
      expect(source, file).not.toMatch(
        /from "\.\/(official|officialResult|evidence|ModelReadout)|from "\.\.\/api"|\/predict/,
      );
    }
  });
});

describe("WhatsApp alert", () => {
  it("builds a demo, simulated message and a wa.me link only", () => {
    const message = buildAlertMessage(SIM_SENSORS[1]);
    expect(message).toBe(
      "LeakLess demo alert\nSimulated anomaly detected near Sim Sensor 2.\nLocation: Building demo network · Zone N2.\nAction: inspection recommended.",
    );
    expect(message).toMatch(/demo/i);
    expect(message).toMatch(/simulated/i);
    expect(message).not.toMatch(FORBIDDEN);
    expect(whatsappLink(message, "")).toBe(
      `https://wa.me/?text=${encodeURIComponent(message)}`,
    );
    expect(whatsappLink(message, "41790000000")).toMatch(
      /^https:\/\/wa\.me\/41790000000\?text=/,
    );
  });

  it("refuses an alert that is not marked demo or simulated", () => {
    expect(() => assertDemoAlert("Leak alert near sensor 2")).toThrow(
      /demo or simulated/,
    );
    expect(() => whatsappLink("Inspection needed", "")).toThrow();
  });

  it("keeps only digits of the optional recipient", () => {
    expect(alertRecipient("+41 79 000 00 00")).toBe("41790000000");
    expect(alertRecipient("")).toBe("");
    expect(alertRecipient(undefined)).toBe("");
  });
});

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

describe("monitor simulation flow", () => {
  it("adds a labelled simulation on a pipe click and clears it without touching the replay", async () => {
    vi.mocked(loadExample).mockImplementation(async (record) => ({
      record,
      sample: {} as never,
      visualization: visualization(
        -50 - recordings.records.indexOf(record) * 5,
      ),
    }));
    const { container } = render(createElement(MonitorReplay));
    await screen.findByRole("article", {
      name: "Dataset label Leak-associated",
    });
    fireEvent.click(screen.getByRole("button", { name: "Pause replay" }));
    const recMarkers = () =>
      [...container.querySelectorAll(".listen-point")].map(
        (node) => node.outerHTML,
      );
    const cards = () =>
      [...container.querySelectorAll(".monitor-card")].map(
        (node) => node.textContent,
      );
    const beforeMarkers = recMarkers();
    const beforeCards = cards();
    expect(container.querySelector(".sim-panel")).toBeNull();

    const hit = screen.getByRole("button", {
      name: "Simulate an incident on pipe 6",
    });
    expect(hit).toHaveAttribute("stroke-width", String(HIT_STROKE));
    fireEvent.keyDown(hit, { key: "Enter" });

    expect(screen.getAllByText("SIMULATION MODE")).toHaveLength(2);
    expect(
      screen.getByRole("heading", {
        name: "SIMULATED INCIDENT — synthetic event and illustrative sensor responses. TSLM not involved.",
      }),
    ).toBeInTheDocument();
    expect(container.querySelectorAll(".sim-sensor")).toHaveLength(3);
    expect(container.querySelector(".sim-incident")).not.toBeNull();
    expect(
      screen.getByText("Inspection recommended near Sim Sensor 2"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Demo rule: response strength is simulated from distance. Not a model output.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(
        "Illustrative sensor positions — not dataset channels.",
      ),
    ).not.toHaveLength(0);
    const preview = screen.getByRole("figure", { name: "Alert preview" });
    expect(
      within(preview).getByText(/LeakLess demo alert/),
    ).toBeInTheDocument();
    const send = screen.getByRole("link", { name: /Send WhatsApp alert/ });
    expect(send.getAttribute("href")).toMatch(
      /^https:\/\/wa\.me\/\d*\?text=LeakLess%20demo%20alert/,
    );
    expect(send).toHaveAttribute("target", "_blank");
    const panelText = container.querySelector(".sim-panel")?.textContent ?? "";
    expect(panelText).not.toMatch(FORBIDDEN);
    expect(document.body.textContent).not.toMatch(
      /\d\s*%|probability|confidence/i,
    );

    // Measured REC markers and cards are untouched by the simulation.
    expect(recMarkers()).toEqual(beforeMarkers);
    expect(cards()).toEqual(beforeCards);

    fireEvent.click(screen.getByRole("button", { name: "Clear simulation" }));
    expect(container.querySelector(".sim-panel")).toBeNull();
    expect(container.querySelector(".sim-layer")).toBeNull();
    expect(screen.queryByText("SIMULATION MODE")).toBeNull();
    expect(recMarkers()).toEqual(beforeMarkers);
    expect(cards()).toEqual(beforeCards);
    expect(
      screen.getByRole("button", { name: "Resume replay" }),
    ).toBeInTheDocument();
  });

  it("projects a pointer click in drawing units", async () => {
    vi.mocked(loadExample).mockReturnValue(new Promise(() => {}));
    const { container } = render(createElement(MonitorReplay));
    const svg = container.querySelector(".pipe-network svg") as SVGSVGElement;
    vi.spyOn(svg, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 1240,
      height: 280,
      right: 1240,
      bottom: 280,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
    // Screen (400, 230) is drawing (400, 260): 20 units below the first pipe.
    fireEvent.click(
      screen.getByRole("button", { name: "Simulate an incident on pipe 1" }),
      { clientX: 400, clientY: 230 },
    );
    const marker = container.querySelector(".sim-incident");
    expect(marker).toHaveAttribute("cx", "400");
    expect(marker).toHaveAttribute("cy", "240");
    expect(
      screen.getByText("Inspection recommended near Sim Sensor 1"),
    ).toBeInTheDocument();
  });
});

describe("navigation", () => {
  it("opens Evidence at its heading when leaving the monitor", async () => {
    vi.mocked(loadExample).mockReturnValue(new Promise(() => {}));
    const scrollTo = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
    const intoView = vi.fn();
    Element.prototype.scrollIntoView = intoView;
    render(createElement(App));
    await screen.findByRole("heading", { name: /Water damage/ });
    const go = (hash: string) =>
      act(async () => {
        location.hash = hash;
        window.dispatchEvent(new HashChangeEvent("hashchange"));
      });
    await go("#monitor");
    await screen.findByText(/Replay, not a live feed\./);
    scrollTo.mockClear();
    intoView.mockClear();
    await go("#evidence");
    expect(intoView).toHaveBeenCalled();
    expect(intoView.mock.contexts.at(-1)).toBe(
      document.getElementById("evidence"),
    );
    expect(scrollTo).not.toHaveBeenCalled();
  });
});
