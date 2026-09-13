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
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { loadExample } from "./loadExample";
import MonitorReplay from "./MonitorReplay";
import {
  HIT_STROKE,
  PIPE_PATHS,
  RESPONSE_LAMBDA,
  SENSORS,
  projectToPipe,
  simulateIncident,
} from "./simulation";
import { api } from "../api";
import type { Incident, IncidentSnapshot } from "./incidents";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  api: vi.fn(),
}));
vi.mock("./loadExample", () => ({ loadExample: vi.fn() }));
vi.mock("./SceneCanvas", () => ({ SceneCanvas: () => null }));
afterEach(() => {
  vi.useRealTimers();
  cleanup();
  vi.restoreAllMocks();
  location.hash = "";
});

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

  it("finds the nearest sensor node and applies exp(-d / lambda)", () => {
    const incident = simulateIncident({ x: 505, y: 150 });
    expect(incident.position).toEqual({ x: 505, y: 160 });
    expect(incident.nearest.name).toBe("Sensor 2");
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
    expect(simulateIncident({ x: 870, y: 120 }).nearest.name).toBe("Sensor 3");
  });

  it("keeps sensors on the drawn pipes and hit targets large on a phone", () => {
    for (const sensor of SENSORS)
      expect(projectToPipe(sensor).distance).toBe(0);
    // Measured in Chrome: the drawing is ~319 px wide on a 420 px screen; the hit stroke stays above 32 px.
    expect((HIT_STROKE * 319) / 1240).toBeGreaterThanOrEqual(32);
    expect(PIPE_PATHS).toHaveLength(6);
  });

  it("simulation code never imports the official TSLM result or a model", () => {
    for (const file of ["simulation.ts", "SimulationPanel.tsx"]) {
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

/** In-memory stand-in for the backend /incidents API: the test decides every server transition. */
function fakeServer(persistenceSeconds = 30) {
  const server = {
    active: null as Incident | null,
    history: [] as Incident[],
    now: () => Date.now() / 1000,
    retries: 0,
  };
  const snapshot = (): IncidentSnapshot => ({
    server_time: server.now(),
    persistence_seconds: persistenceSeconds,
    transport: { name: "telegram", configured: true },
    active: server.active,
    history: server.history,
  });
  vi.mocked(api).mockImplementation(async (path, schema, options) => {
    if (path === "/incidents" && options?.method === "POST" && !server.active) {
      const body = JSON.parse(String(options.body));
      server.active = {
        incident_id: "inc-test0001",
        source: "injected_test",
        strongest_sensor: body.strongest_sensor,
        zone: `N${body.strongest_sensor}`,
        position: body.position,
        started_at: server.now(),
        status: "ANOMALY_PENDING",
        confirmed_at: null,
        alert_sent_at: null,
        resolved_at: null,
        transport: null,
        delivery_status: "not_dispatched",
        telegram_message_id: null,
        failure_reason: null,
      };
    }
    if (path.endsWith("/retry") && server.active) {
      server.retries += 1;
    }
    if (path.endsWith("/resolve") && server.active) {
      server.history = [
        { ...server.active, status: "RESOLVED", resolved_at: server.now() },
        ...server.history,
      ];
      server.active = null;
    }
    return schema.parse(snapshot());
  });
  return server;
}

const REMOVED =
  /SIMULATION MODE|SIM [123]\b|Sim Sensor|LeakLess demo alert|Demo rule|SIMULATED INCIDENT|TSLM not involved|Test scenario|response distribution|no leak is shown|WhatsApp|leak probability|confidence|accuracy|TSLM detected|TELEGRAM_|api\.telegram|synthetic|NO PER-RECORDING OUTPUT|Model output|not configured|\bDEMO\b|SIMULATION/i;

const tick = (ms: number) =>
  act(async () => {
    vi.advanceTimersByTime(ms);
  });

describe("monitor incident flow (backend-owned)", () => {
  it("observes for 30 s, recommends inspection only after server confirmation, shows SENT only after Telegram, keeps history", async () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "Date"] });
    vi.setSystemTime(new Date("2026-09-13T10:57:50"));
    const server = fakeServer();
    const { container } = render(createElement(MonitorReplay));
    await tick(0);
    const alerts = () =>
      container.querySelector(".monitor-status .alert-state");
    const status = () => container.querySelector(".alert-status");
    const stage = () => container.querySelector("#sim-title")?.textContent;
    expect(alerts()).toHaveTextContent("NORMAL");
    expect(alerts()).toHaveClass("is-normal");
    expect(
      screen.getByText("3 sensors online · 2 Hz · last 120 s"),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Scenario injection")).toHaveLength(1);
    expect(container.querySelectorAll(".listen-point")).toHaveLength(3);
    expect(container.querySelector(".live-threshold-label")).toHaveTextContent(
      "Alert threshold · 1.6× baseline",
    );

    fireEvent.click(screen.getByRole("button", { name: "Inject incident" }));
    await tick(0);
    // t = 0: observation only. No recommendation before the persistence condition.
    expect(stage()).toBe("Anomaly under observation");
    expect(
      screen.getByText("Strongest response: Sensor 03"),
    ).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/Inspection recommended/);
    expect(status()).toHaveTextContent("Armed · waiting 30 s");
    expect(alerts()).toHaveClass("is-observing");
    expect(container.querySelectorAll(".response-ring")).toHaveLength(3);

    await tick(12_000);
    expect(alerts()).toHaveTextContent(
      "ANOMALY UNDER OBSERVATION · 12 s / 30 s",
    );
    expect(
      container.querySelector(".persistence .demo-kicker"),
    ).toHaveTextContent("Anomaly under observation · 12 s / 30 s");
    expect(document.body.textContent).not.toMatch(
      /Inspection recommended|Sent/,
    );

    // Server confirms persistence and starts the Telegram request.
    await tick(18_000);
    const started = server.active!.started_at;
    server.active = {
      ...server.active!,
      status: "ALERT_DISPATCHING",
      confirmed_at: started + 30,
      transport: "telegram",
      delivery_status: "dispatching",
    };
    await tick(1_000);
    expect(stage()).toBe("Persistent anomaly confirmed");
    expect(screen.getByText("Inspection recommended")).toBeInTheDocument();
    expect(status()).toHaveTextContent("Dispatching");
    expect(alerts()).toHaveClass("is-triggered");
    expect(document.body.textContent).not.toMatch(/\bSent\b/);

    // Telegram confirmed delivery.
    server.active = {
      ...server.active!,
      status: "ALERT_SENT",
      delivery_status: "sent",
      alert_sent_at: started + 31,
      telegram_message_id: "9001",
    };
    await tick(1_000);
    expect(status()).toHaveTextContent("Sent · 10:58:21");
    expect(alerts()).toHaveTextContent("ALERT SENT");
    expect(screen.getByText("Telegram → On-call plumber")).toBeInTheDocument();
    expect(screen.getByText("Not implemented")).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(REMOVED);

    // Resolve: nothing erased, a resolution marker and the incident history remain.
    await tick(220_000);
    fireEvent.click(screen.getByRole("button", { name: "Resolve incident" }));
    await tick(0);
    expect(container.querySelector(".sim-panel")).toBeNull();
    expect(alerts()).toHaveTextContent("RESOLVED");
    const history = screen.getByRole("region", { name: "Last incident" });
    expect(history).toHaveTextContent(
      /10:57:50 anomaly started.*10:58:20 persistence threshold reached.*10:58:21 alert sent · telegram.*11:02:02 resolved/,
    );
    expect(history).toHaveTextContent("inc-test0001");
    expect(history).toHaveTextContent("#9001");
    expect(history).toHaveTextContent("telegram · sent");
    await tick(1_000);
    expect(container.querySelector(".live-resolved-line")).not.toBeNull();
    expect(container.querySelector(".live-event rect")).not.toBeNull();
    expect(document.body.textContent).not.toMatch(REMOVED);
  });

  it("injects from a pipe click at the projected position", async () => {
    fakeServer();
    const { container } = render(createElement(MonitorReplay));
    await screen.findByText("NORMAL");
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
      screen.getByRole("button", { name: "Inject incident on pipe 1" }),
      { clientX: 400, clientY: 230 },
    );
    expect(
      await screen.findByText(
        "Strongest response: Sensor 03".replace("03", "01"),
      ),
    ).toBeInTheDocument();
    const marker = container.querySelector(".sim-incident");
    expect(marker).toHaveAttribute("cx", "400");
    expect(marker).toHaveAttribute("cy", "240");
  });

  it("frontend source holds no Telegram secret or direct Telegram call", () => {
    for (const file of [
      "incidents.ts",
      "MonitorReplay.tsx",
      "SimulationPanel.tsx",
    ]) {
      const source = readFileSync(
        join(process.cwd(), "src/demo", file),
        "utf8",
      );
      expect(source, file).not.toMatch(/TELEGRAM_|api\.telegram\.org|bot\d+:/);
    }
  });
});

describe("code review regressions", () => {
  it("shows the persistence duration supplied by the server (single source of truth)", async () => {
    fakeServer(12);
    const { container } = render(createElement(MonitorReplay));
    await screen.findByText("NORMAL");
    fireEvent.click(screen.getByRole("button", { name: "Inject incident" }));
    expect(await screen.findByText("Armed · waiting 12 s")).toBeInTheDocument();
    expect(
      container.querySelector(".monitor-status .alert-state"),
    ).toHaveTextContent(/^ANOMALY UNDER OBSERVATION · \d+ s \/ 12 s$/);
    expect(document.body.textContent).not.toMatch(/\b30 s\b/);
  });

  it("never lets a slow poll overwrite a newer resolve result", async () => {
    const server = fakeServer();
    const { container } = render(createElement(MonitorReplay));
    await screen.findByText("NORMAL");
    fireEvent.click(screen.getByRole("button", { name: "Inject incident" }));
    await screen.findByText("Resolve incident");
    // Next poll hangs and carries the pre-resolve state.
    const staleActive = server.active;
    let releaseStalePoll: () => void = () => {};
    const realImpl = vi.mocked(api).getMockImplementation()!;
    vi.mocked(api).mockImplementation((path, schema, options) => {
      if (path === "/incidents/current")
        return new Promise((resolve) => {
          releaseStalePoll = () =>
            resolve(
              schema.parse({
                server_time: Date.now() / 1000,
                persistence_seconds: 30,
                transport: { name: "telegram", configured: true },
                active: staleActive,
                history: [],
              }),
            );
        });
      return realImpl(path, schema, options);
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1100));
    });
    fireEvent.click(screen.getByRole("button", { name: "Resolve incident" }));
    await screen.findByText("RESOLVED");
    await act(async () => {
      releaseStalePoll();
    });
    expect(container.querySelector(".sim-panel")).toBeNull();
    expect(screen.queryByText("Resolve incident")).toBeNull();
    expect(screen.getByText("RESOLVED")).toBeInTheDocument();
  });

  it("offers an explicit retry when the notification failed, and resolve frees the slot", async () => {
    const server = fakeServer();
    render(createElement(MonitorReplay));
    await screen.findByText("NORMAL");
    fireEvent.click(screen.getByRole("button", { name: "Inject incident" }));
    await screen.findByText("Resolve incident");
    expect(screen.queryByText("Retry notification")).toBeNull();
    server.active = {
      ...server.active!,
      status: "ALERT_FAILED",
      confirmed_at: server.active!.started_at + 30,
      transport: "telegram",
      delivery_status: "failed",
      failure_reason: "Telegram rejected the message: chat not found.",
    };
    expect(
      await screen.findByText(
        "Notification failed · Telegram delivery unavailable",
      ),
    ).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/chat not found|TELEGRAM_/);
    fireEvent.click(screen.getByRole("button", { name: "Retry notification" }));
    await vi.waitFor(() => expect(server.retries).toBe(1));
    await vi.waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Resolve incident" }),
      ).not.toBeDisabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: "Resolve incident" }));
    await screen.findByText("RESOLVED");
    expect(
      screen.getByRole("button", { name: "Inject incident" }),
    ).not.toBeDisabled();
  });
});

describe("navigation", () => {
  it("opens Evidence at its heading when leaving the monitor", async () => {
    vi.mocked(loadExample).mockReturnValue(new Promise(() => {}));
    fakeServer();
    const scrollTo = vi.spyOn(window, "scrollTo").mockImplementation(() => {});
    const intoView = vi.fn();
    Element.prototype.scrollIntoView = intoView;
    render(createElement(App));
    await screen.findByRole(
      "heading",
      { name: /Water damage/ },
      { timeout: 5000 },
    );
    const go = (hash: string) =>
      act(async () => {
        location.hash = hash;
        window.dispatchEvent(new HashChangeEvent("hashchange"));
      });
    await go("#monitor");
    await screen.findByText(/Water signal · last 120 s/, {}, { timeout: 5000 });
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
