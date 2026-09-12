"""Validation du contrat de run : `metadata.json` + `predictions.csv`.

Tout ce qui veut être évalué — les contrôles C0 à C3 comme le TSLM de Hicham —
passe par ce validateur. Il n'y a pas de chemin de contournement.

Chaque contrôle publie sa **couverture** en plus de son verdict. Un contrôle qui
renvoie « 0 problème » sans dire sur combien de clips il a porté ne prouve rien :
c'est exactement l'erreur qui a invalidé `split_v1` (docs/SPLIT_V2_AUDIT.md §3).

Ce que le validateur refuse, et qui sont autant de sentinelles testées :

  - une colonne de métadonnée dans `predictions.csv` (pression, débit, device,
    chemin, nom de fichier, md5) ;
  - une colonne `fold` ou `label` dans `predictions.csv` : les folds et les
    étiquettes viennent du manifeste, jamais du modèle ;
  - un clip_id inconnu, en double, ou manquant pour le fold évalué ;
  - un SHA de split qui ne correspond pas au gelé ;
  - une probabilité hors [0, 1] ou non numérique ;
  - `split_v1` sous n'importe quelle forme.
"""

from __future__ import annotations

import csv
import json
import math
from numbers import Real
import re
from dataclasses import dataclass, field
from pathlib import Path

from .split_loader import FROZEN_SPLIT_NAME, FROZEN_SPLIT_SHA256, Split, sha256_of

# Colonnes interdites dans predictions.csv. Deux familles :
#  - métadonnées d'acquisition : elles ne doivent jamais circuler avec un modèle ;
#  - fold / label : ils appartiennent au manifeste, un modèle ne peut pas les redéfinir.
FORBIDDEN_PREDICTION_COLUMNS = frozenset({
    "pressure", "pressure_mpa", "flow", "flow_ms", "device", "path", "filename",
    "material", "region", "noise_category", "md5", "window", "rep",
    "fold", "label", "label_3c", "y", "target", "group_id",
})

REQUIRED_METADATA_FIELDS = (
    "run_id", "model_name", "checkpoint", "training_commit", "split_filename",
    "split_sha256", "timestamp", "threshold_rule", "test_labels_not_used_for_tuning",
)


class ContractError(RuntimeError):
    """Le run ne respecte pas le contrat. L'évaluation s'arrête."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    coverage: str
    detail: str = ""


@dataclass
class PredictionRun:
    run_id: str
    model_name: str
    metadata: dict
    probabilities: dict[str, float]
    source: Path
    checks: list[CheckResult] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"run « {self.run_id} » ({self.model_name}) — {self.source}"]
        for c in self.checks:
            lines.append(f"  {'OK  ' if c.passed else 'FAIL'} {c.name}"
                         f"   [couverture {c.coverage}]"
                         + (f"\n       {c.detail}" if c.detail and not c.passed else ""))
        return "\n".join(lines)


def _fail(checks: list[CheckResult], name: str, coverage: str, detail: str) -> None:
    checks.append(CheckResult(name, False, coverage, detail))
    raise ContractError(f"{name} — {detail}")


def _external_manifest(split: Split, name: str | None, digest: str | None) -> bool:
    """Opt-in explicite : identité ET octets du manifeste externe déjà chargé.

    Le lecteur externe construit les Clip/ Split et vérifie ses cibles séparées.
    Cette fonction ne remplace pas ce lecteur ni le split_loader historique.
    """
    if name is None and digest is None:
        return False
    if (not isinstance(name, str) or not name or name.strip() != name
            or name in (".", "..") or any(char in name for char in ("/", "\\", "\x00", ":"))
            or re.search(r"split_v\d", name, flags=re.IGNORECASE)
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or digest == FROZEN_SPLIT_SHA256):
        raise ContractError("Nom distinct et SHA-256 complet du manifeste externe requis ensemble")
    path = Path(split.manifest_path)
    if path.name != name or split.sha256 != digest:
        raise ContractError("Identité externe différente du manifeste chargé")
    if not path.is_file() or sha256_of(path) != digest:
        raise ContractError("Octets du manifeste externe différents de l'empreinte attendue")
    ids = [clip.clip_id for clip in split.clips]
    if not ids or len(ids) != len(set(ids)):
        raise ContractError("Scope externe vide ou contenant des clip_id dupliqués")
    return True


def load_run(run_dir: str | Path, split: Split, *, folds: tuple[str, ...] = ("val", "test"),
             require_frozen_sha: bool = True, external_manifest_name: str | None = None,
             external_manifest_sha256: str | None = None) -> PredictionRun:
    """Charge et valide un dossier de run.

    Args:
        run_dir: dossier contenant `metadata.json` et `predictions.csv`.
        split: le split gelé déjà chargé et vérifié.
        folds: folds dont chaque clip doit être prédit exactement une fois.
        require_frozen_sha: vérifie que le run déclare le SHA gelé.
        external_manifest_name / external_manifest_sha256: opt-in fourni ensemble.
            Vérifie l'identité du vrai manifeste externe au lieu de l'identité V1 ;
            impose deux colonnes exactes et une couverture exacte des folds choisis.
            Exemple : folds=("external",), nom="external_aghashahi_v1.json".

    Raises:
        ContractError: au premier contrôle en échec, avec sa couverture.
    """
    run_dir = Path(run_dir)
    is_external = _external_manifest(split, external_manifest_name, external_manifest_sha256)
    expected_name = external_manifest_name if is_external else FROZEN_SPLIT_NAME
    expected_sha = external_manifest_sha256 if is_external else FROZEN_SPLIT_SHA256
    checks: list[CheckResult] = []
    meta_path, pred_path = run_dir / "metadata.json", run_dir / "predictions.csv"

    for p in (meta_path, pred_path):
        if not p.exists():
            _fail(checks, "fichiers du run présents", "2 fichiers requis",
                  f"{p.name} manquant dans {run_dir}")
    checks.append(CheckResult("fichiers du run présents", True, "2/2 fichiers"))

    with open(meta_path) as fh:
        meta = json.load(fh)

    missing = [f for f in REQUIRED_METADATA_FIELDS if f not in meta]
    if missing:
        _fail(checks, "champs de metadata.json", f"{len(REQUIRED_METADATA_FIELDS)} requis",
              f"champs manquants : {missing}")
    checks.append(CheckResult("champs de metadata.json", True,
                              f"{len(REQUIRED_METADATA_FIELDS)}/{len(REQUIRED_METADATA_FIELDS)} présents"))

    if meta["split_filename"] != expected_name:
        _fail(checks, "identité du split déclaré", "1 champ",
              f"le run déclare « {meta['split_filename']} ». Seul {expected_name} "
              f"est valide ; split_v1 est INVALIDE.")
    checks.append(CheckResult("identité du split déclaré", True, "1/1 champ"))

    if (require_frozen_sha or is_external) and meta["split_sha256"] != expected_sha:
        _fail(checks, "SHA256 du split déclaré", "1 champ",
              f"déclaré {meta['split_sha256'][:16]}…, gelé {expected_sha[:16]}…")
    if meta["split_sha256"] != split.sha256:
        _fail(checks, "SHA256 du split déclaré", "1 champ",
              f"le run déclare {meta['split_sha256'][:16]}… mais le manifeste chargé "
              f"vaut {split.sha256[:16]}…")
    checks.append(CheckResult("SHA256 du split déclaré", True, "1/1 champ"))
    if is_external:
        checks.append(CheckResult("octets du manifeste externe épinglés", True, "1/1 manifeste",
                                  f"{external_manifest_name} : {external_manifest_sha256}"))

    if meta["test_labels_not_used_for_tuning"] is not True:
        _fail(checks, "déclaration d'absence de réglage sur le test", "1 champ",
              "le run ne déclare pas que les étiquettes de test n'ont pas servi au réglage")
    checks.append(CheckResult("déclaration d'absence de réglage sur le test", True, "1/1 champ"))

    with open(pred_path) as fh:
        reader = csv.DictReader(fh)
        ordered_columns = reader.fieldnames or []
        columns = set(reader.fieldnames or [])
        rows = list(reader)

    if is_external and (ordered_columns != ["clip_id", "probability_leak"]
                        or any(set(row) != {"clip_id", "probability_leak"} for row in rows)):
        _fail(checks, "colonnes externes exactes", f"{len(rows)} lignes ; {len(ordered_columns)} colonnes",
              "exactement clip_id,probability_leak, sans colonne ajoutée/dupliquée ni valeur surnuméraire")

    leaked = sorted(columns & FORBIDDEN_PREDICTION_COLUMNS)
    if leaked:
        _fail(checks, "aucune colonne interdite dans predictions.csv",
              f"{len(columns)} colonnes inspectées",
              f"colonnes interdites présentes : {leaked}. Les folds et les étiquettes "
              f"viennent du manifeste ; les métadonnées d'acquisition ne circulent pas.")
    checks.append(CheckResult("aucune colonne interdite dans predictions.csv", True,
                              f"{len(columns)} colonnes inspectées"))

    if not {"clip_id", "probability_leak"} <= columns:
        _fail(checks, "colonnes requises dans predictions.csv", f"{len(columns)} colonnes",
              f"il faut clip_id et probability_leak, trouvé {sorted(columns)}")
    checks.append(CheckResult("colonnes requises dans predictions.csv", True,
                              "2/2 colonnes requises"))

    known = split.by_id()
    probs: dict[str, float] = {}
    for r in rows:
        cid = r["clip_id"]
        if cid in probs:
            _fail(checks, "aucun clip_id en double", f"{len(rows)} lignes",
                  f"clip_id prédit deux fois : {cid}")
        if cid not in known:
            _fail(checks, "aucun clip_id inconnu", f"{len(rows)} lignes",
                  f"clip_id absent du manifeste : {cid}")
        try:
            p = float(r["probability_leak"])
        except (TypeError, ValueError):
            _fail(checks, "probabilités numériques", f"{len(rows)} lignes",
                  f"probabilité non numérique pour {cid} : {r['probability_leak']!r}")
        if math.isnan(p) or not (0.0 <= p <= 1.0):
            _fail(checks, "probabilités dans [0, 1]", f"{len(rows)} lignes",
                  f"probabilité hors bornes pour {cid} : {p}")
        probs[cid] = p
    checks.append(CheckResult("aucun clip_id en double", True, f"{len(rows)} lignes"))
    checks.append(CheckResult("aucun clip_id inconnu", True, f"{len(rows)} lignes"))
    checks.append(CheckResult("probabilités dans [0, 1]", True, f"{len(rows)} lignes"))

    required = {c.clip_id for c in split.clips if c.fold in folds}
    if is_external and (not required or set(probs) != required):
        _fail(checks, f"couverture externe exacte des folds {folds}", f"{len(required)} clips attendus",
              f"scope vide ou couverture différente : {len(required - set(probs))} manquants, "
              f"{len(set(probs) - required)} hors scope")
    missing_ids = sorted(required - set(probs))
    if missing_ids:
        _fail(checks, f"couverture des folds {folds}", f"{len(required)} clips attendus",
              f"{len(missing_ids)} clip(s) sans prédiction, ex. {missing_ids[:3]}")
    checks.append(CheckResult(f"couverture des folds {folds}", True,
                              f"{len(required)}/{len(required)} clips prédits"))

    return PredictionRun(run_id=meta["run_id"], model_name=meta["model_name"], metadata=meta,
                         probabilities=probs, source=run_dir, checks=checks)


def write_run(run_dir: str | Path, *, run_id: str, model_name: str, checkpoint: str,
              training_commit: str, split: Split, threshold_rule: str,
              probabilities: dict[str, float], extra: dict | None = None,
              external_manifest_name: str | None = None,
              external_manifest_sha256: str | None = None) -> Path:
    """Écrit un run ; comportement historique inchangé sans opt-in externe.

    En externe, le Split représente exactement le scope à exporter : aucun ID
    supplémentaire/manquant, fichier de labels séparé, dossier neuf et scores
    flottants conservés à 17 chiffres significatifs.
    Extra peut compléter la provenance, pas écraser les champs requis.
    """
    import datetime

    run_dir = Path(run_dir)
    is_external = _external_manifest(split, external_manifest_name, external_manifest_sha256)
    if is_external:
        if run_dir.exists() or run_dir.is_symlink():
            raise ContractError("Sortie externe déjà présente : aucun écrasement, même d'un dossier vide")
        if set(probabilities) != {clip.clip_id for clip in split.clips}:
            raise ContractError("Couverture externe différente du scope chargé")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise ContractError("Probabilités externes finies dans [0,1] requises, sans valeur de remplacement")
        if set(extra or {}) & set(REQUIRED_METADATA_FIELDS):
            raise ContractError("Extra ne peut pas remplacer la provenance externe requise")
    run_dir.mkdir(parents=True, exist_ok=not is_external)
    meta = {
        "run_id": run_id,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "training_commit": training_commit,
        "config_hash": None,
        "split_filename": external_manifest_name if is_external else FROZEN_SPLIT_NAME,
        "split_sha256": split.sha256,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "threshold_rule": threshold_rule,
        "test_labels_not_used_for_tuning": True,
        **(extra or {}),
    }
    with open(run_dir / "metadata.json", "x" if is_external else "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(run_dir / "predictions.csv", "x" if is_external else "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "probability_leak"])
        for cid in sorted(probabilities):
            value = f"{float(probabilities[cid]):.17g}" if is_external else f"{probabilities[cid]:.10f}"
            w.writerow([cid, value])
    return run_dir
