// @vitest-environment jsdom
// Dry run d'affichage avec la fixture SYNTHÉTIQUE injectée par mock : ce chemin
// n'existe que dans les tests.
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import DemoExperience from "./DemoExperience";
import { loadExample } from "./loadExample";
import MonitorReplay from "./MonitorReplay";

vi.mock("./loadExample", () => ({ loadExample: vi.fn() }));
vi.mock("./SceneCanvas", () => ({
  SceneCanvas: () => <div>3D test host</div>,
}));
vi.mock("./official", async () => {
  const { readOfficialResult, tslmCopy } = await import("./officialResult");
  const dir = "../../tests/fixtures/synthetic-official-tslm";
  const files = {
    metrics: (await import(`${dir}/metrics.json`)).default,
    comparison: (await import(`${dir}/comparison.json`)).default,
    receipt: (await import(`${dir}/receipt.json`)).default,
  };
  const officialTslm = readOfficialResult(files, { allowSynthetic: true });
  return { officialTslm, tslm: tslmCopy(officialTslm) };
});
afterEach(cleanup);

it("populates C4, keeps the control order and shows no per-recording probability", () => {
  render(<DemoExperience />);
  const cards = document.querySelectorAll(".control-card");
  expect(
    [...cards].map(
      (c) => c.querySelector(".control-id")?.firstChild?.textContent,
    ),
  ).toEqual(["C0", "C1", "C2", "C2b", "C3", "C4"]);
  const c4 = within(cards[5] as HTMLElement);
  expect(c4.getByText("0.111")).toBeInTheDocument();
  expect(c4.getByText("0.222")).toBeInTheDocument();
  expect(
    within(cards[1] as HTMLElement).getByText("0.902"),
  ).toBeInTheDocument();
  expect(screen.queryByText(/NOT EVALUATED YET/)).not.toBeInTheDocument();
  expect(screen.getByText(/No demonstrated advantage\./)).toBeInTheDocument();
  expect(
    screen.getByText(/Temporal stress on the TSLM, not retrained/),
  ).toBeInTheDocument();
  expect(
    screen.getAllByText("Probability leak")[0].nextElementSibling,
  ).toHaveTextContent("NOT SHOWN");
  const text = (document.body.textContent ?? "").replace(
    "Probability leak",
    "",
  );
  expect(text).not.toMatch(/%|chance|probability of|likely/i);
});

it("monitor and point panel show no per-recording model output", async () => {
  vi.mocked(loadExample).mockReturnValue(new Promise(() => {}));
  render(<MonitorReplay />);
  expect(
    await screen.findByRole("heading", { name: "Continuous monitoring" }),
  ).toBeInTheDocument();
  // The operator monitor carries no model score or benchmark figure at all.
  expect(document.body.textContent).not.toMatch(
    /%|chance|0\.111|0\.222|TSLM|probability/i,
  );
});
