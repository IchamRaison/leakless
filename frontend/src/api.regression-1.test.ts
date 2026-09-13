// Regression: ISSUE-003 — with the API stopped, the dev proxy's 502 surfaced as "Server error (502)." with no next step
// Found by /qa on 2026-09-12
// Report: .gstack/qa-reports/qa-report-127-0-0-1-5178-2026-09-12.md
import { afterEach, expect, it, vi } from "vitest";
import { z } from "zod";
import { api } from "./api";

afterEach(() => vi.unstubAllGlobals());
const respond = (status: number, body: string) =>
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(body, { status })),
  );

it.each([502, 503, 504])(
  "explains an unreachable API on a bodiless %s from the proxy",
  async (status) => {
    respond(status, "");
    await expect(api("/health", z.unknown())).rejects.toThrow(
      "API unreachable. Check that the local server is running, then retry.",
    );
  },
);

it("keeps the API's own message and other status codes unchanged", async () => {
  respond(503, JSON.stringify({ error: { message: "Model unavailable." } }));
  await expect(api("/predict", z.unknown())).rejects.toThrow(
    "Model unavailable.",
  );
  respond(500, "");
  await expect(api("/health", z.unknown())).rejects.toThrow(
    "Server error (500).",
  );
});
