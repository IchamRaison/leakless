// Lecture du résultat TSLM OFFICIEL produit par le générateur gelé
// (protocol-freeze-v1, scripts/eval/build_final_report.py). Aucun calcul
// d'évaluation ici : on sélectionne des champs existants, on vérifie leur
// cohérence, et tout écart renvoie l'état NOT EVALUATED YET.
import { z } from "zod";
import { evidence } from "./evidence";

export const PROTOCOL_TAG = "protocol-freeze-v1";
export const PROTOCOL_COMMIT = "3e4e73ab6e1944b42b0223e9bd9b3d500f49bc65";
const CONTROL_ORDER = ["c0", "c1", "c2", "c2b", "c3"] as const;
const SYNTHETIC_MARKER = "SYNTHETIC";
// Identité du split gelé, lue dans le rapport contrôles seuls du tag.
const FROZEN_SPLIT = {
  filename: "split_v2.csv",
  sha256: "7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896",
} as const;
// Un identifiant de run est affiché : aucun texte libre ne passe.
const runId = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/);

// Formulations imposées par harness/metrics.py::verdict.
const lecture = z.enum([
  "compatible with improvement",
  "compatible with degradation",
  "inconclusive",
]);
export type Lecture = z.infer<typeof lecture>;
// build_final_report.py::STRESS_VERDICT : les comparaisons de stress portent
// T0 − stress, la lecture est donc inversée pour se lire « sous stress ».
const STRESS_VERDICT = {
  "compatible with improvement": "compatible with degradation under stress",
  "compatible with degradation": "compatible with improvement under stress",
  inconclusive: "inconclusive",
} as const;

const unit = z.number().finite().min(0).max(1);
const delta = z.object({
  delta_observe: z.number().finite(),
  ci95_low: z.number().finite(),
  ci95_high: z.number().finite(),
  lecture,
});
const interval = z.object({ ci95_low: unit, ci95_high: unit });
const fold = z.object({
  n_clips: z.number().int().positive(),
  n_clusters: z.number().int().positive(),
  n_clusters_leak: z.number().int().nonnegative(),
  n_clusters_non_leak: z.number().int().nonnegative(),
  clip_level: z.object({
    roc_auc: unit,
    pr_auc: unit,
    macro_f1: unit,
    brier: unit,
  }),
  cluster_level: z.object({ roc_auc: unit, macro_f1: unit }),
  bootstrap_ci95: z.object({
    clip_roc_auc: interval,
    cluster_roc_auc: interval,
  }),
});
const run = z.object({
  run_id: runId,
  model_name: z.string().min(1),
  checkpoint: z.string().nullable(),
  training_commit: z.string().nullable(),
  folds: z.object({ test: fold }),
});
const pairwise = z.object({
  a: z.string(),
  b: z.string(),
  fold: z.literal("test"),
  n_clusters: z.number().int().positive(),
  clip_roc_auc: delta,
  cluster_roc_auc: delta,
});
const metricsSchema = z.object({
  split: z.object({ filename: z.string(), sha256: z.string() }),
  control_ladder: z.array(z.string()),
  runs: z.record(z.string(), run),
  stress_run_bases: z.record(
    runId,
    z.tuple([runId, z.enum(["T1", "T2", "T3"])]),
  ),
  stress_comparisons: z.record(z.string(), pairwise),
  tslm_run_id: runId.nullable(),
});
type Pairwise = z.infer<typeof pairwise>;
const comparisonSchema = z.record(z.string(), pairwise);
// Reçu d'intégration écrit au moment de la copie, d'après le verdict de CC2.
// Il ne porte aucun score : seulement l'identité et les empreintes des fichiers.
export const receiptSchema = z
  .object({
    status: z.literal("OFFICIAL"),
    contract_check: z.literal("PASS"),
    provenance_check: z.literal("PASS"),
    final_report: z.literal("GENERATED"),
    protocol_tag: z.literal(PROTOCOL_TAG),
    protocol_commit: z.literal(PROTOCOL_COMMIT),
    report_commit: z
      .string()
      .regex(/^[0-9a-f]{40}$/)
      .refine((commit) => !/^0+$/.test(commit)),
    tslm_run_id: runId,
    metrics_sha256: z.string().regex(/^[0-9a-f]{64}$/),
    comparison_sha256: z.string().regex(/^[0-9a-f]{64}$/),
  })
  .strict();

export type PairedDelta = {
  delta: number;
  low: number;
  high: number;
  lecture: Lecture;
};
export type OfficialTslmResult = {
  protocolTag: string;
  protocolCommit: string;
  reportCommit: string;
  splitSha256: string;
  runId: string;
  modelName: string;
  trainingCommit: string | null;
  test: {
    nClips: number;
    nClusters: number;
    nClustersLeak: number;
    nClustersNonLeak: number;
    clipAuc: number;
    clipPrAuc: number;
    clipMacroF1: number;
    clipBrier: number;
    clusterAuc: number;
    clusterMacroF1: number;
    clipAucCi: [number, number];
    clusterAucCi: [number, number];
  };
  /** TSLM − contrôle, apparié sur les mêmes tirages de clusters (test). */
  versus: Record<
    (typeof CONTROL_ORDER)[number],
    { clipAuc: PairedDelta; clusterAuc: PairedDelta }
  >;
  /** Lecture de stress déjà orientée « sous stress », seuil exclu. */
  stress: {
    transform: string;
    runId: string;
    clipAuc: number;
    clusterAuc: number;
    clipAucVerdict: string;
    clusterAucVerdict: string;
  }[];
  /** Cas A/B/C contre C1, le contrôle de référence. */
  outcome: "improvement" | "inconclusive" | "degradation";
};
export type OfficialState =
  | { status: "NOT_EVALUATED"; reason: string }
  | { status: "OFFICIAL"; result: OfficialTslmResult };

export type OfficialFiles = {
  metrics?: unknown;
  comparison?: unknown;
  receipt?: unknown;
};

const notEvaluated = (reason: string): OfficialState => ({
  status: "NOT_EVALUATED",
  reason,
});
const paired = (d: z.infer<typeof delta>): PairedDelta => ({
  delta: d.delta_observe,
  low: d.ci95_low,
  high: d.ci95_high,
  lecture: d.lecture,
});
const round3 = (x: number) => Math.round(x * 1000) / 1000;

/** Pure : fichiers gelés en entrée, résultat officiel ou NOT_EVALUATED en sortie. */
export function readOfficialResult(
  files: OfficialFiles,
  { allowSynthetic = false }: { allowSynthetic?: boolean } = {},
): OfficialState {
  if (
    files.metrics === undefined &&
    files.comparison === undefined &&
    files.receipt === undefined
  )
    return notEvaluated("No official TSLM result delivered.");
  const receiptParsed = receiptSchema.safeParse(files.receipt);
  if (!receiptParsed.success)
    return notEvaluated("Receipt missing or invalid.");
  const metricsParsed = metricsSchema.safeParse(files.metrics);
  if (!metricsParsed.success) return notEvaluated("metrics.json invalid.");
  const comparisonParsed = comparisonSchema.safeParse(files.comparison);
  if (!comparisonParsed.success)
    return notEvaluated("comparison.json invalid.");
  const receipt = receiptParsed.data;
  const metrics = metricsParsed.data;
  const comparison = comparisonParsed.data;

  if (
    !allowSynthetic &&
    JSON.stringify([files.receipt, files.metrics, files.comparison]).includes(
      SYNTHETIC_MARKER,
    )
  )
    return notEvaluated("Synthetic fixture refused.");

  const tslmId = metrics.tslm_run_id;
  if (!tslmId) return notEvaluated("Report has no TSLM run.");
  if (receipt.tslm_run_id !== tslmId)
    return notEvaluated("Receipt and report name different TSLM runs.");
  if (JSON.stringify(metrics.control_ladder) !== JSON.stringify(CONTROL_ORDER))
    return notEvaluated("Control ladder differs from the frozen ladder.");
  if (
    metrics.split.filename !== FROZEN_SPLIT.filename ||
    metrics.split.sha256 !== FROZEN_SPLIT.sha256
  )
    return notEvaluated("Report is not on the frozen split.");
  if ((CONTROL_ORDER as readonly string[]).includes(tslmId.toLowerCase()))
    return notEvaluated("TSLM run id collides with a control.");
  const tslm = metrics.runs[tslmId];
  if (!tslm) return notEvaluated("TSLM run missing from metrics.json.");
  const t = tslm.folds.test;

  // Les contrôles affichés viennent de evidence.ts : le rapport doit porter
  // exactement les mêmes chiffres, sinon ce n'est pas le même split ni la même échelle.
  for (const [i, id] of CONTROL_ORDER.entries()) {
    const shown = evidence.controls[i];
    const c = metrics.runs[id]?.folds.test;
    if (
      !c ||
      round3(c.clip_level.roc_auc) !== shown.clipAuc ||
      round3(c.cluster_level.roc_auc) !== shown.clusterAuc
    )
      return notEvaluated(`Control ${id} differs from the displayed evidence.`);
    if (c.n_clips !== t.n_clips || c.n_clusters !== t.n_clusters)
      return notEvaluated(`TSLM and ${id} test populations differ.`);
    if (
      JSON.stringify([c.clip_level, c.cluster_level]) ===
      JSON.stringify([t.clip_level, t.cluster_level])
    )
      return notEvaluated(`TSLM scores are a copy of control ${id}.`);
  }
  if (
    t.n_clusters !== evidence.testClusters ||
    t.n_clusters_non_leak !== evidence.testNonLeakClusters
  )
    return notEvaluated(
      "Test cluster counts differ from the displayed evidence.",
    );

  const versus = {} as OfficialTslmResult["versus"];
  for (const id of CONTROL_ORDER) {
    const c: Pairwise | undefined = comparison[`${tslmId}_vs_${id}`];
    if (!c || c.a !== tslmId || c.b !== id)
      return notEvaluated(`Paired comparison ${tslmId} vs ${id} missing.`);
    versus[id] = {
      clipAuc: paired(c.clip_roc_auc),
      clusterAuc: paired(c.cluster_roc_auc),
    };
  }

  const stress: OfficialTslmResult["stress"] = [];
  for (const [rid, [base, transform]] of Object.entries(
    metrics.stress_run_bases,
  ).sort(([a], [b]) => a.localeCompare(b))) {
    if (base !== tslmId) continue;
    const s = metrics.runs[rid];
    const c: Pairwise | undefined =
      metrics.stress_comparisons[`${tslmId}_vs_${rid}`];
    if (!s || !c || c.a !== tslmId || c.b !== rid)
      return notEvaluated(`Stress run ${rid} incomplete.`);
    if (
      s.folds.test.n_clips !== t.n_clips ||
      s.folds.test.n_clusters !== t.n_clusters
    )
      return notEvaluated(`Stress run ${rid} test population differs.`);
    stress.push({
      transform,
      runId: rid,
      clipAuc: s.folds.test.clip_level.roc_auc,
      clusterAuc: s.folds.test.cluster_level.roc_auc,
      clipAucVerdict: STRESS_VERDICT[c.clip_roc_auc.lecture],
      clusterAucVerdict: STRESS_VERDICT[c.cluster_roc_auc.lecture],
    });
  }

  const c1 = [versus.c1.clipAuc.lecture, versus.c1.clusterAuc.lecture];
  const outcome = c1.includes("compatible with degradation")
    ? "degradation"
    : c1.every((l) => l === "compatible with improvement")
      ? "improvement"
      : "inconclusive";

  return {
    status: "OFFICIAL",
    result: {
      protocolTag: receipt.protocol_tag,
      protocolCommit: receipt.protocol_commit,
      reportCommit: receipt.report_commit,
      splitSha256: metrics.split.sha256,
      runId: tslmId,
      modelName: tslm.model_name,
      trainingCommit: tslm.training_commit,
      test: {
        nClips: t.n_clips,
        nClusters: t.n_clusters,
        nClustersLeak: t.n_clusters_leak,
        nClustersNonLeak: t.n_clusters_non_leak,
        clipAuc: t.clip_level.roc_auc,
        clipPrAuc: t.clip_level.pr_auc,
        clipMacroF1: t.clip_level.macro_f1,
        clipBrier: t.clip_level.brier,
        clusterAuc: t.cluster_level.roc_auc,
        clusterMacroF1: t.cluster_level.macro_f1,
        clipAucCi: [
          t.bootstrap_ci95.clip_roc_auc.ci95_low,
          t.bootstrap_ci95.clip_roc_auc.ci95_high,
        ],
        clusterAucCi: [
          t.bootstrap_ci95.cluster_roc_auc.ci95_low,
          t.bootstrap_ci95.cluster_roc_auc.ci95_high,
        ],
      },
      versus,
      stress,
      outcome,
    },
  };
}

/** Tout le texte public lié au TSLM. Hors résultat officiel : texte actuel, inchangé. */
export function tslmCopy(state: OfficialState) {
  if (state.status !== "OFFICIAL")
    return {
      evaluated: false,
      c4: {
        detail: "Awaiting an evaluated checkpoint.",
        clipAuc: null as number | null,
        clusterAuc: null as number | null,
      },
      verdict: "No TSLM superiority demonstrated.",
      stress: null as string | null,
      status: "TSLM · NOT EVALUATED YET",
      probability: "NOT EVALUATED YET",
      modelOutput: "Awaiting evaluated checkpoint",
      definition:
        "TSLM = Time-Series Language Model. Training and evaluation are still in progress; no final held-out result is shown here.",
      decision:
        "No inspection recommendation is generated while TSLM is not evaluated yet.",
      mapKey: "model output · not evaluated yet",
      monitorBanner: "No detection model output: TSLM not evaluated yet.",
    };
  const r = state.result;
  const c1 = r.versus.c1;
  const suffix = {
    improvement: "No statistical significance claimed.",
    inconclusive: "No demonstrated advantage.",
    degradation: "C1 remains the bar to beat.",
  }[r.outcome];
  return {
    evaluated: true,
    c4: {
      detail: `Evaluated offline on the same frozen split · run ${r.runId}.`,
      clipAuc: round3(r.test.clipAuc),
      clusterAuc: round3(r.test.clusterAuc),
    },
    verdict: `TSLM vs C1, paired on the same clusters: clip AUC ${c1.clipAuc.lecture}; cluster AUC ${c1.clusterAuc.lecture}. ${suffix}`,
    stress: r.stress.length
      ? `Temporal stress on the TSLM, not retrained: ${r.stress
          .map((s) => `${s.transform} clip AUC ${s.clipAucVerdict}`)
          .join(
            "; ",
          )}. This measures sensitivity to temporal organisation, not its physical relevance.`
      : null,
    status: "TSLM · NO PER-RECORDING OUTPUT",
    probability: "NOT SHOWN",
    modelOutput: "Held-out benchmark only · see evidence",
    definition: `TSLM = Time-Series Language Model. Evaluated offline on the frozen held-out split (${r.protocolTag}); its scores are benchmark results, not outputs for this recording.`,
    decision:
      "No inspection recommendation is generated. TSLM scores are held-out benchmark results, not per-recording outputs.",
    mapKey: "model output · not shown per recording",
    monitorBanner:
      "No detection model output: TSLM is benchmarked offline only.",
  };
}
