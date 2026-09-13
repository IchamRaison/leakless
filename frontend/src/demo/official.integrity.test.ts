// Garde du dossier de dépôt réel : vide => NOT EVALUATED ; rempli => il doit
// être officiel, intègre et non synthétique, sinon la suite échoue bruyamment.
import {
  mkdtempSync,
  readdirSync,
  readFileSync,
  writeFileSync,
  copyFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";
import { checkOfficialDrop } from "../../dev/officialGuard";
import { officialTslm } from "./official";

const dir = fileURLToPath(new URL("./official/", import.meta.url));
const fixture = fileURLToPath(
  new URL("../../tests/fixtures/synthetic-official-tslm/", import.meta.url),
);
const present = readdirSync(dir).filter((name) => name.endsWith(".json"));

it("the real drop zone passes the build guard", () => {
  expect(checkOfficialDrop(dir)).toEqual([]);
});

it("an empty drop zone renders NOT EVALUATED YET; a filled one must be OFFICIAL", () => {
  expect(officialTslm.status).toBe(
    present.length ? "OFFICIAL" : "NOT_EVALUATED",
  );
});

const drop = (edit?: (d: string) => void) => {
  const d = mkdtempSync(join(tmpdir(), "official-"));
  for (const name of ["metrics.json", "comparison.json", "receipt.json"])
    copyFileSync(join(fixture, name), join(d, name));
  edit?.(d);
  return checkOfficialDrop(d);
};

it("the guard refuses the synthetic fixture copied into a drop zone", () => {
  expect(drop()).toEqual(
    expect.arrayContaining([expect.stringMatching(/synthetic fixture marker/)]),
  );
});

it("the guard refuses a tampered file and an extra JSON", () => {
  expect(drop((d) => writeFileSync(join(d, "metrics.json"), "{}"))).toEqual(
    expect.arrayContaining(["metrics.json sha256 does not match receipt"]),
  );
  expect(
    drop((d) => writeFileSync(join(d, "metrics.v2.json"), "{}"))[0],
  ).toMatch(/expected exactly/);
  expect(checkOfficialDrop(mkdtempSync(join(tmpdir(), "official-")))).toEqual(
    [],
  );
});

it("no production module imports the synthetic fixture", () => {
  const src = new URL("../", import.meta.url);
  const walk = (url: URL): string[] =>
    readdirSync(url, { withFileTypes: true }).flatMap((entry) =>
      entry.isDirectory()
        ? walk(new URL(`${entry.name}/`, url))
        : /\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name)
          ? [fileURLToPath(new URL(entry.name, url))]
          : [],
    );
  for (const file of walk(src))
    expect(readFileSync(file, "utf8"), file).not.toMatch(
      /tests\/fixtures|SYNTHETIC-FIXTURE/,
    );
});
