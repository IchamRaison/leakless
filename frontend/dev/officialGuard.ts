// Garde du dossier de dépôt du résultat officiel, exécutée par Vite au démarrage
// du serveur de dev et avant chaque build : un dépôt incohérent arrête tout,
// au lieu d'afficher en silence NOT EVALUATED YET ou un score non vérifié.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { Plugin } from "vite";

const EXPECTED = ["comparison.json", "metrics.json", "receipt.json"];

/** Liste des problèmes du dossier ; vide = dossier vide ou dépôt intègre. */
export function checkOfficialDrop(dir: string): string[] {
  const present = readdirSync(dir)
    .filter((name) => name.endsWith(".json"))
    .sort();
  if (present.length === 0) return [];
  if (JSON.stringify(present) !== JSON.stringify(EXPECTED))
    return [
      `expected exactly ${EXPECTED.join(", ")}; found ${present.join(", ")}`,
    ];
  const raw = (name: string) => readFileSync(join(dir, name));
  const problems: string[] = [];
  let receipt: { metrics_sha256?: unknown; comparison_sha256?: unknown };
  try {
    receipt = JSON.parse(raw("receipt.json").toString());
  } catch {
    return ["receipt.json is not valid JSON"];
  }
  for (const [name, key] of [
    ["metrics.json", "metrics_sha256"],
    ["comparison.json", "comparison_sha256"],
  ] as const) {
    const sha = createHash("sha256").update(raw(name)).digest("hex");
    if (sha !== receipt[key])
      problems.push(`${name} sha256 does not match receipt`);
  }
  for (const name of present)
    if (raw(name).toString().includes("SYNTHETIC"))
      problems.push(`${name} contains a synthetic fixture marker`);
  return problems;
}

export function officialResultGuard(dir: string): Plugin {
  return {
    name: "official-result-guard",
    buildStart() {
      const problems = checkOfficialDrop(dir);
      if (problems.length)
        this.error(`Official TSLM result refused: ${problems.join("; ")}`);
    },
  };
}
