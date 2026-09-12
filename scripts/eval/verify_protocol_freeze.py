#!/usr/bin/env python3
"""Vérifie que le protocole d'évaluation gelé n'a pas bougé.

Le gel (`protocol/FREEZE_v1.json`) fixe, par SHA-256, chaque fichier qui définit
l'expérience : split, contrôles C0-C3, moteur d'évaluation, bootstrap, stress
T0-T3, contrat de run, règles de provenance, rapport final. Il fixe aussi les
prédictions des contrôles publiés, hors dépôt.

Un fichier gardé ne peut changer que par un amendement déclaré dans
`protocol/AMENDMENTS.json`, de catégorie `implementation-bugfix`, qui cite la
preuve du défaut et dit si des résultats TSLM avaient été vus. Les amendements
s'enchaînent : chacun part de l'empreinte laissée par le précédent. Un fichier
modifié sans amendement, ou dont l'empreinte ne correspond à aucun maillon de la
chaîne, fait échouer la vérification.

Ce script ne juge pas les intentions : il rend chaque changement explicite,
daté et relisible dans l'historique Git.

Usage :
  python3 scripts/eval/verify_protocol_freeze.py                  # fichiers du dépôt
  python3 scripts/eval/verify_protocol_freeze.py --runs-dir <runs> # + contrôles publiés
  python3 scripts/eval/verify_protocol_freeze.py --git             # le gel décrit bien son commit
  python3 scripts/eval/verify_protocol_freeze.py --self-test       # le vérificateur détecte une altération

Code de sortie 0 si le protocole est intact, 1 sinon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "protocol" / "FREEZE_v1.json"
AMENDMENTS = ROOT / "protocol" / "AMENDMENTS.json"
ALLOWED_CATEGORIES = {"implementation-bugfix"}
AMENDMENT_FIELDS = ("id", "date", "category", "paths", "demonstrated_by", "justification",
                    "tslm_results_inspected_before")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_chain(path: str, frozen: str, amendments: list[dict]) -> tuple[list[str], list[str]]:
    """Empreintes successives autorisées pour `path`, et erreurs de chaînage."""
    chain, errors = [frozen], []
    for a in amendments:
        step = a.get("paths", {}).get(path)
        if step is None:
            continue
        if step.get("from") != chain[-1]:
            errors.append(f"{path} : l'amendement {a.get('id')} part de "
                          f"{str(step.get('from'))[:12]}, attendu {chain[-1][:12]}")
        chain.append(step.get("to"))
    return chain, errors


def check_amendments(amendments: list[dict], guarded: dict) -> list[str]:
    errors = []
    for a in amendments:
        missing = [k for k in AMENDMENT_FIELDS if k not in a or a[k] in (None, "", {}, [])
                   and k != "tslm_results_inspected_before"]
        if missing:
            errors.append(f"amendement {a.get('id')} : champs manquants {missing}")
        if a.get("category") not in ALLOWED_CATEGORIES:
            errors.append(f"amendement {a.get('id')} : catégorie {a.get('category')!r} interdite "
                          f"(autorisée : {sorted(ALLOWED_CATEGORIES)})")
        if not isinstance(a.get("tslm_results_inspected_before"), bool):
            errors.append(f"amendement {a.get('id')} : tslm_results_inspected_before doit être "
                          f"déclaré true ou false")
        for p in a.get("paths", {}):
            if p not in guarded:
                errors.append(f"amendement {a.get('id')} : {p} n'est pas un fichier gardé")
    return errors


def verify_files(root: Path, freeze: dict, amendments: list[dict]) -> list[str]:
    guarded = freeze["guarded_files"]
    errors = check_amendments(amendments, guarded)
    for path, frozen in sorted(guarded.items()):
        f = root / path
        if not f.is_file():
            errors.append(f"{path} : fichier gardé absent")
            continue
        chain, chain_errors = expected_chain(path, frozen, amendments)
        errors += chain_errors
        actual = sha256_file(f)
        if actual != chain[-1]:
            where = "modifié sans amendement" if actual not in chain else \
                "revenu à un état antérieur de la chaîne d'amendements"
            errors.append(f"{path} : {where} (attendu {chain[-1][:12]}, trouvé {actual[:12]})")
    return errors


def verify_runs(runs_dir: Path, freeze: dict) -> list[str]:
    errors = []
    for rid, spec in sorted(freeze["frozen_control_runs"].items()):
        d = runs_dir / rid
        if not (d / "predictions.csv").is_file():
            errors.append(f"run {rid} : absent de {runs_dir}")
            continue
        actual = sha256_file(d / "predictions.csv")
        if actual != spec["predictions_sha256"]:
            errors.append(f"run {rid} : predictions.csv modifié "
                          f"(attendu {spec['predictions_sha256'][:12]}, trouvé {actual[:12]})")
        meta = json.loads((d / "metadata.json").read_text())
        for k, v in spec["identity"].items():
            if meta.get(k) != v:
                errors.append(f"run {rid} : {k} = {meta.get(k)!r}, gelé à {v!r}")
    return errors


def verify_git(root: Path, freeze: dict) -> list[str]:
    errors, commit = [], freeze["frozen_commit"]
    for path, frozen in sorted(freeze["guarded_files"].items()):
        if path in freeze.get("added_by_freeze_commit", []):
            continue
        r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=root, capture_output=True)
        if r.returncode != 0:
            errors.append(f"{path} : absent de {commit[:12]}")
        elif hashlib.sha256(r.stdout).hexdigest() != frozen:
            errors.append(f"{path} : le gel ne décrit pas {commit[:12]}")
    return errors


def self_test() -> list[str]:
    """Le vérificateur doit échouer sur une altération, et passer sur un amendement conforme."""
    freeze = json.loads(FREEZE.read_text())
    errors = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for p in freeze["guarded_files"]:
            (tmp / p).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / p, tmp / p)
        if verify_files(tmp, freeze, []):
            errors.append("copie intacte refusée")
        victim = "scripts/eval/harness/metrics.py"
        before = sha256_file(tmp / victim)
        (tmp / victim).write_bytes((tmp / victim).read_bytes() + b"\n# ajustement\n")
        after = sha256_file(tmp / victim)
        if not verify_files(tmp, freeze, []):
            errors.append("altération non détectée")
        good = [{"id": "A1", "date": "2026-09-12", "category": "implementation-bugfix",
                 "paths": {victim: {"from": before, "to": after}},
                 "demonstrated_by": "test", "justification": "test",
                 "tslm_results_inspected_before": False}]
        if verify_files(tmp, freeze, good):
            errors.append("amendement conforme refusé")
        bad_cat = [{**good[0], "category": "tslm-performance"}]
        if not verify_files(tmp, freeze, bad_cat):
            errors.append("catégorie d'amendement interdite acceptée")
        broken = [{**good[0], "paths": {victim: {"from": "0" * 64, "to": after}}}]
        if not verify_files(tmp, freeze, broken):
            errors.append("chaîne d'amendements rompue acceptée")
        undeclared = [{k: v for k, v in good[0].items() if k != "tslm_results_inspected_before"}]
        if not verify_files(tmp, freeze, undeclared):
            errors.append("amendement sans déclaration tslm_results_inspected_before accepté")
        (tmp / victim).unlink()
        if not verify_files(tmp, freeze, good):
            errors.append("fichier gardé supprimé non détecté")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", help="vérifie aussi les runs de contrôle publiés")
    ap.add_argument("--git", action="store_true", help="vérifie que le gel décrit son commit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        errors = self_test()
        print("self-test : " + ("ÉCHEC\n  " + "\n  ".join(errors) if errors else "OK"))
        sys.exit(1 if errors else 0)

    freeze = json.loads(FREEZE.read_text())
    amendments = json.loads(AMENDMENTS.read_text())["amendments"]
    errors = verify_files(ROOT, freeze, amendments)
    if args.git:
        errors += verify_git(ROOT, freeze)
    if args.runs_dir:
        errors += verify_runs(Path(args.runs_dir), freeze)

    print(f"gel {freeze['freeze_id']} — commit gelé {freeze['frozen_commit'][:12]} — "
          f"{len(freeze['guarded_files'])} fichiers gardés, {len(amendments)} amendement(s)"
          + (f", {len(freeze['frozen_control_runs'])} runs de contrôle" if args.runs_dir else ""))
    if errors:
        print("\n❌ PROTOCOLE MODIFIÉ\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("✅ protocole intact")


if __name__ == "__main__":
    main()
