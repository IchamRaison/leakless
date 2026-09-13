import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import comparison from "../../tests/fixtures/synthetic-official-tslm/comparison.json";
import controlsOnly from "../../tests/fixtures/synthetic-official-tslm/controls-only-metrics.json";
import metrics from "../../tests/fixtures/synthetic-official-tslm/metrics.json";
import receipt from "../../tests/fixtures/synthetic-official-tslm/receipt.json";
import {
  readOfficialResult,
  tslmCopy,
  type OfficialFiles,
} from "./officialResult";

// Fixture SYNTHÉTIQUE : chiffres TSLM inventés, jamais atteignables en production.
const fixture = () =>
  structuredClone({ metrics, comparison, receipt }) as {
    metrics: any;
    comparison: any;
    receipt: any;
  };
const read = (files: OfficialFiles) =>
  readOfficialResult(files, { allowSynthetic: true });
const T = "SYNTHETIC-FIXTURE-tslm";

it("absent result keeps the exact current NOT EVALUATED YET copy and no C4 score", () => {
  const state = readOfficialResult({});
  expect(state.status).toBe("NOT_EVALUATED");
  // Texte figé par le DEMO FREEZE : toute différence ici change le site actuel.
  expect(tslmCopy(state)).toEqual({
    evaluated: false,
    c4: {
      detail: "Awaiting an evaluated checkpoint.",
      clipAuc: null,
      clusterAuc: null,
    },
    verdict: "No TSLM superiority demonstrated.",
    stress: null,
    status: "TSLM · NOT EVALUATED YET",
    probability: "NOT EVALUATED YET",
    modelOutput: "Awaiting evaluated checkpoint",
    definition:
      "TSLM = Time-Series Language Model. Training and evaluation are still in progress; no final held-out result is shown here.",
    decision:
      "No inspection recommendation is generated while TSLM is not evaluated yet.",
    mapKey: "model output · not evaluated yet",
    monitorBanner: "No detection model output: TSLM not evaluated yet.",
  });
});

it("dry run: a valid frozen report populates C4 from its own fields", () => {
  const state = read(fixture());
  expect(state.status).toBe("OFFICIAL");
  if (state.status !== "OFFICIAL") return;
  expect(state.result.runId).toBe(T);
  expect(state.result.test.nClusters).toBe(41);
  expect(state.result.test.nClustersNonLeak).toBe(11);
  expect(state.result.stress.map((s) => s.transform)).toEqual([
    "T1",
    "T2",
    "T3",
  ]);
  const copy = tslmCopy(state);
  expect(copy.c4.clipAuc).toBe(0.111);
  expect(copy.c4.clusterAuc).toBe(0.222);
  expect(copy.verdict).toBe(
    "TSLM vs C1, paired on the same clusters: clip AUC inconclusive; cluster AUC inconclusive. No demonstrated advantage.",
  );
});

it("the production reader refuses the synthetic fixture", () => {
  expect(readOfficialResult(fixture()).status).toBe("NOT_EVALUATED");
});

it("the real controls-only frozen report (no TSLM run) stays NOT EVALUATED", () => {
  const files = fixture();
  files.metrics = structuredClone(controlsOnly);
  files.receipt.tslm_run_id = "c2b";
  expect(read(files)).toMatchObject({ status: "NOT_EVALUATED" });
});

const malformed: [string, (f: ReturnType<typeof fixture>) => void][] = [
  ["receipt missing", (f) => delete f.receipt],
  ["metrics missing", (f) => delete f.metrics],
  ["comparison missing", (f) => delete f.comparison],
  ["receipt not official", (f) => (f.receipt.status = "DRAFT")],
  ["contract not passed", (f) => (f.receipt.contract_check = "FAIL")],
  [
    "other protocol commit",
    (f) => (f.receipt.protocol_commit = "0".repeat(40)),
  ],
  ["extra receipt field", (f) => (f.receipt.clip_roc_auc = 0.9)],
  ["run id mismatch", (f) => (f.receipt.tslm_run_id = "tslm-v2")],
  ["TSLM run absent", (f) => delete f.metrics.runs[T]],
  [
    "AUC out of range",
    (f) => (f.metrics.runs[T].folds.test.clip_level.roc_auc = 1.5),
  ],
  [
    "AUC not a number",
    (f) => (f.metrics.runs[T].folds.test.clip_level.roc_auc = "0.9"),
  ],
  [
    "control changed",
    (f) => (f.metrics.runs.c1.folds.test.clip_level.roc_auc = 0.95),
  ],
  ["ladder changed", (f) => f.metrics.control_ladder.pop()],
  ["pair missing", (f) => delete f.comparison[`${T}_vs_c1`]],
  [
    "unknown lecture",
    (f) => (f.comparison[`${T}_vs_c1`].clip_roc_auc.lecture = "significant"),
  ],
  [
    "stress comparison missing",
    (f) => delete f.metrics.stress_comparisons[`${T}_vs_${T}-T2`],
  ],
  ["not JSON objects", (f) => (f.metrics = "garbage")],
];
it.each(malformed)(
  "malformed (%s) falls back to NOT EVALUATED YET",
  (_, edit) => {
    const files = fixture();
    edit(files);
    const state = read(files);
    expect(state.status).toBe("NOT_EVALUATED");
    expect(tslmCopy(state).c4.clipAuc).toBeNull();
  },
);

const setC1 = (
  f: ReturnType<typeof fixture>,
  clip: string,
  cluster: string,
) => {
  f.comparison[`${T}_vs_c1`].clip_roc_auc.lecture = clip;
  f.comparison[`${T}_vs_c1`].cluster_roc_auc.lecture = cluster;
};

it("case A (compatible with improvement on both) never says superior", () => {
  const files = fixture();
  setC1(files, "compatible with improvement", "compatible with improvement");
  const state = read(files);
  expect(state).toMatchObject({ result: { outcome: "improvement" } });
  const { verdict } = tslmCopy(state);
  expect(verdict).toContain("compatible with improvement");
  expect(verdict).toContain("No statistical significance claimed.");
  expect(verdict).not.toMatch(/superior|significant(?!ce claimed)|better/i);
});

it("case B: one improvement and one inconclusive stays inconclusive", () => {
  const files = fixture();
  setC1(files, "compatible with improvement", "inconclusive");
  expect(read(files)).toMatchObject({ result: { outcome: "inconclusive" } });
});

it("case C (degradation vs C1) is shown honestly", () => {
  const files = fixture();
  setC1(files, "compatible with improvement", "compatible with degradation");
  const state = read(files);
  expect(state).toMatchObject({ result: { outcome: "degradation" } });
  expect(tslmCopy(state).verdict).toContain(
    "cluster AUC compatible with degradation. C1 remains the bar to beat.",
  );
});

it("stress lectures use the frozen inverted mapping, threshold-free only", () => {
  const files = fixture();
  files.metrics.stress_comparisons[`${T}_vs_${T}-T2`].clip_roc_auc.lecture =
    "compatible with improvement";
  const state = read(files);
  expect(tslmCopy(state).stress).toBe(
    "Temporal stress on the TSLM, not retrained: T1 clip AUC inconclusive; T2 clip AUC compatible with degradation under stress; T3 clip AUC inconclusive. This measures sensitivity to temporal organisation, not its physical relevance.",
  );
});

it("official copy never turns a score into a per-recording probability", () => {
  const copy = tslmCopy(read(fixture()));
  const text = JSON.stringify(copy);
  expect(text).not.toMatch(/%|chance|likely|probability of/i);
  expect(copy.probability).toBe("NOT SHOWN");
});

it("fixture receipt hashes match its files, as the real receipt must", () => {
  const dir = new URL(
    "../../tests/fixtures/synthetic-official-tslm/",
    import.meta.url,
  );
  const sha = (name: string) =>
    createHash("sha256")
      .update(readFileSync(new URL(name, dir)))
      .digest("hex");
  expect(sha("metrics.json")).toBe(receipt.metrics_sha256);
  expect(sha("comparison.json")).toBe(receipt.comparison_sha256);
});

it("control stress runs (c2b-T1..T3) in the real report are kept out of the TSLM stress line", () => {
  const files = fixture();
  const real = structuredClone(controlsOnly) as any;
  for (const rid of ["c2b-T1", "c2b-T2", "c2b-T3"]) {
    files.metrics.runs[rid] = real.runs[rid];
    files.metrics.stress_run_bases[rid] = real.stress_run_bases[rid];
    files.metrics.stress_comparisons[`c2b_vs_${rid}`] =
      real.stress_comparisons[`c2b_vs_${rid}`];
  }
  const state = read(files);
  expect(state.status).toBe("OFFICIAL");
  if (state.status !== "OFFICIAL") return;
  expect(state.result.stress.map((s) => s.runId)).toEqual([
    `${T}-T1`,
    `${T}-T2`,
    `${T}-T3`,
  ]);
});

it("a TSLM report without stress runs is still official, with no stress line", () => {
  const files = fixture();
  files.metrics.stress_run_bases = {};
  files.metrics.stress_comparisons = {};
  const state = read(files);
  expect(state.status).toBe("OFFICIAL");
  expect(tslmCopy(state).stress).toBeNull();
});

const hardening: [string, (f: ReturnType<typeof fixture>) => void][] = [
  [
    "a control copied under an upper-case id",
    (f) => {
      f.metrics.runs.C1 = structuredClone(f.metrics.runs.c1);
      f.metrics.runs.C1.run_id = "C1";
      f.metrics.tslm_run_id = "C1";
      f.receipt.tslm_run_id = "C1";
    },
  ],
  [
    "free text in a stress transform label",
    (f) => {
      f.metrics.stress_run_bases[`${T}-T1`][1] = "T1 (97% leak probability)";
    },
  ],
  [
    "free text in a run id",
    (f) => {
      f.metrics.tslm_run_id = "87% chance of leak";
      f.receipt.tslm_run_id = "87% chance of leak";
    },
  ],
  ["another split", (f) => (f.metrics.split.sha256 = "0".repeat(64))],
  [
    "a report commit other than the frozen protocol commit",
    (f) => (f.receipt.report_commit = "5a29d6eb".padEnd(40, "0")),
  ],
  [
    "a stress run on another population",
    (f) => (f.metrics.runs[`${T}-T2`].folds.test.n_clusters = 3),
  ],
];
it.each(hardening)("refuses %s", (_, edit) => {
  const files = fixture();
  edit(files);
  expect(read(files).status).toBe("NOT_EVALUATED");
});

it("TSLM metric values equal to a control are not an identity signal", () => {
  const files = fixture();
  const t = files.metrics.runs[T].folds.test;
  const c = files.metrics.runs.c1.folds.test;
  t.clip_level = structuredClone(c.clip_level);
  t.cluster_level = structuredClone(c.cluster_level);
  expect(read(files).status).toBe("OFFICIAL");
});
