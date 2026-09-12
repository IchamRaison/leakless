// Presentation copy of the authorized results. No model probabilities live here.
// Update this single module from the scientific report; never compute evaluation in the UI.
export const evidence = {
  source:
    "https://github.com/IchamRaison/ehl-hackathon-zurich/tree/nevil/temporal-evidence",
  records: 1000,
  clusters: 185,
  testClusters: 41,
  testNonLeakClusters: 11,
  controls: [
    {
      id: "C0",
      name: "Loudness",
      detail: "Raw-signal RMS. Absolute level only.",
      clipAuc: 0.878,
      clusterAuc: 0.839,
    },
    {
      id: "C1",
      name: "Envelope",
      detail: "Normalised amplitude shape. No frequency features.",
      clipAuc: 0.902,
      clusterAuc: 0.927,
    },
    {
      id: "C2",
      name: "Static spectrum",
      detail: "Aggregated spectral descriptors. No phase or ordering.",
      clipAuc: 0.71,
      clusterAuc: 0.779,
    },
    {
      id: "C2b",
      name: "Shallow temporal",
      detail: "A simple control of temporal information.",
      clipAuc: 0.847,
      clusterAuc: 0.915,
    },
    {
      id: "C3",
      name: "Mixed static baseline",
      detail: "Envelope + spectrum + zero-crossing. Non-temporal.",
      clipAuc: 0.824,
      clusterAuc: 0.9,
    },
    {
      id: "C4",
      name: "TSLM",
      detail: "Awaiting an evaluated checkpoint.",
      clipAuc: null,
      clusterAuc: null,
    },
  ],
  demonstrated: [
    "Experimental acoustic dataset processed through TimeNet/TimeF",
    "Leakage-aware frozen split",
    "C0–C3 and C2b controls evaluated",
    "Shallow temporal control responds to temporal perturbations",
  ],
  notDemonstrated: [
    "Field leak detection",
    "Physical leak localization",
    "Customer deployment",
    "Early warning lead time",
    "Causal physical interpretation",
    "TSLM superiority",
  ],
} as const;
