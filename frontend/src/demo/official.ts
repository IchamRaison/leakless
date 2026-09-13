// Point d'entrée de production. Seuls les trois fichiers exacts de ./official/
// sont lus : les fixtures synthétiques des tests n'y sont jamais atteignables,
// et un autre JSON déposé à côté n'entre pas dans le bundle. Les empreintes sont
// vérifiées par le plugin officialResultGuard (vite.config.ts) en dev comme en build.
import { readOfficialResult, tslmCopy } from "./officialResult";

const files = import.meta.glob(
  [
    "./official/metrics.json",
    "./official/comparison.json",
    "./official/receipt.json",
  ],
  { eager: true, import: "default" },
);

export const officialTslm = readOfficialResult({
  metrics: files["./official/metrics.json"],
  comparison: files["./official/comparison.json"],
  receipt: files["./official/receipt.json"],
});
export const tslm = tslmCopy(officialTslm);
