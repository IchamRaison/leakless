# Intégration du résultat TSLM officiel dans la démo

Source d'autorité : tag `protocol-freeze-v1` (`3e4e73ab6e1944b42b0223e9bd9b3d500f49bc65`),
générateur `scripts/eval/build_final_report.py`. La démo ne calcule rien : elle lit des champs
qui existent dans `metrics.json` et `comparison.json`, vérifie leur cohérence, et retombe sur
`TSLM · NOT EVALUATED YET` au moindre écart.

## 1. Sorties du rapport gelé utilisées

`artifacts/final_evaluation/` contient `metrics.json`, `comparison.json`, `FINAL_EVALUATION.md`.

| champ lu                 | fichier             | chemin exact                                                                                                                                           |
| ------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| run TSLM de référence    | metrics             | `tslm_run_id` (null = aucun TSLM, reste NOT EVALUATED)                                                                                                 |
| échelle de contrôles     | metrics             | `control_ladder` == `["c0","c1","c2","c2b","c3"]`                                                                                                      |
| identité                 | metrics             | `runs[<tslm>].model_name`, `.training_commit` (checkpoint lu, jamais affiché : chemin local)                                                           |
| effectifs test           | metrics             | `runs[<tslm>].folds.test.n_clips`, `n_clusters`, `n_clusters_leak`, `n_clusters_non_leak`                                                              |
| scores clip              | metrics             | `…folds.test.clip_level.roc_auc`, `pr_auc`, `macro_f1`, `brier`                                                                                        |
| scores cluster           | metrics             | `…folds.test.cluster_level.roc_auc`, `macro_f1`                                                                                                        |
| IC95 bootstrap cluster   | metrics             | `…folds.test.bootstrap_ci95.clip_roc_auc`, `.cluster_roc_auc`                                                                                          |
| TSLM − contrôle, apparié | comparison          | `"<tslm>_vs_<c>".clip_roc_auc` / `.cluster_roc_auc` : `delta_observe`, `ci95_low`, `ci95_high`, `lecture`                                              |
| stress T1/T2/T3          | metrics             | `stress_run_bases[<rid>] = [<tslm>, "Tn"]`, `runs[<rid>].folds.test.*.roc_auc`, `stress_comparisons["<tslm>_vs_<rid>"].{clip,cluster}_roc_auc.lecture` |
| commit du générateur     | FINAL_EVALUATION.md | ligne `Commit de génération du rapport` (seul endroit où il figure)                                                                                    |

Non utilisés volontairement : F1 et Brier sous stress (seuil recalculé, exclus par le protocole),
`stress_prediction_shift` (pas de lecture publique), toute probabilité par clip.

Il n'existe **pas** de champ `status`, `clip_f1` ou `brier` au niveau racine : le schéma démo
ci-dessous les nomme d'après leur chemin réel.

## 2. Contrat d'entrée de la démo

Zone de dépôt : `frontend/src/demo/official/`. Vide = NOT EVALUATED YET. Trois fichiers exactement :

- `metrics.json`, `comparison.json` : copies octet pour octet du rapport gelé ;
- `receipt.json` : reçu d'intégration, **aucun score**, schéma strict (champ en trop = refus) :

```json
{
  "status": "OFFICIAL",
  "contract_check": "PASS",
  "provenance_check": "PASS",
  "final_report": "GENERATED",
  "protocol_tag": "protocol-freeze-v1",
  "protocol_commit": "3e4e73ab6e1944b42b0223e9bd9b3d500f49bc65",
  "report_commit": "3e4e73ab6e1944b42b0223e9bd9b3d500f49bc65",
  "tslm_run_id": "<== metrics.tslm_run_id>",
  "metrics_sha256": "<sha256 du fichier copié>",
  "comparison_sha256": "<sha256 du fichier copié>"
}
```

Lecteur : `frontend/src/demo/officialResult.ts` (`readOfficialResult`, pur). Objet produit :
`OfficialTslmResult` = identité (`runId`, `modelName`, `trainingCommit`, `reportCommit`,
`splitSha256`), `test` (effectifs, clip AUC/PR-AUC/macro-F1/Brier, cluster AUC/macro-F1, IC95),
`versus.c0…c3` (Δ apparié + lecture), `stress[]` (transformation, AUC, lecture sous stress),
`outcome` contre C1.

Refus → NOT EVALUATED YET si : un fichier manque ; reçu non strict ou autre commit de protocole ;
`tslm_run_id` null ou différent du reçu ; échelle différente ; un AUC hors [0,1] ou non numérique ;
un contrôle arrondi à 3 décimales ≠ `evidence.ts` (autre split ou autre échelle) ; populations test
différentes ; 41/11 clusters non retrouvés ; une comparaison appariée manquante ; une lecture hors
des trois formulations gelées ; un run de stress incomplet ; toute chaîne `SYNTHETIC`.

Garde de dépôt : `dev/officialGuard.ts`, branchée dans `vite.config.ts`. Elle arrête `vite`
(dev) et `vite build` si le dossier ne contient pas exactement les trois fichiers, si une empreinte
ne correspond pas au reçu, ou si un fichier porte le marqueur synthétique. `npm run build` lance en
plus `official.integrity.test.ts`. Seuls les trois chemins exacts sont importés dans le bundle.

Refus supplémentaires du lecteur : identifiant de run hors `[A-Za-z0-9._-]` (aucun texte libre
affiché), identifiant égal à un contrôle quelle que soit la casse, split autre que `split_v2.csv` / `7a8716a3…`, transformation de stress hors
T1/T2/T3, run de stress sur une autre population, `report_commit` différent du commit gelé `3e4e73ab…` (aucun amendement accepté pour cette soumission).

## 3. Formulations publiques (anglais)

Toutes dans `tslmCopy()` ; l'état absent reproduit le texte actuel au caractère près (test figé).

- Référence : **C1** (limite n° 2 du rapport : « le barreau à dépasser est C1 »). Ordre C0 → C4 inchangé.
- Cas A, les deux lectures TSLM − C1 `compatible with improvement` : lectures exactes +
  « No statistical significance claimed. » Jamais « superior » ; « TSLM superiority » reste dans
  _not demonstrated_.
- Cas B, sinon et sans dégradation : lectures exactes + « No demonstrated advantage. »
- Cas C, une lecture `compatible with degradation` : lectures exactes + « C1 remains the bar to beat. »
- Stress : lecture inversée selon `STRESS_VERDICT`, AUC clip seulement, suivie de « This measures
  sensitivity to temporal organisation, not its physical relevance. »
- Par enregistrement : `TSLM · NO PER-RECORDING OUTPUT`, `Probability leak: NOT SHOWN`. Aucun
  pourcentage, aucune probabilité par clip, aucune recommandation d'inspection.

## 4. Correspondance fichiers → champs

| fichier                          | élément                                                                                                                                | absent                                      | officiel                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| `frontend/src/demo/evidence.ts`  | C4 `clipAuc/clusterAuc/detail`                                                                                                         | null, « Awaiting an evaluated checkpoint. » | inchangé : surchargé à l'affichage par `tslm.c4`                  |
| `DemoExperience.tsx` §05         | carte C4, légende sous l'échelle, ligne stress                                                                                         | NOT EVALUATED YET                           | `test.clip/cluster AUC` (3 déc.), verdict C1, stress              |
| `DemoExperience.tsx` §04         | `ModelReadout`, `.tslm-definition`, note de décision, légende carte                                                                    | texte actuel                                | `tslm.status/probability/modelOutput/definition/decision/mapKey`  |
| `ModelReadout.tsx`               | Model / Probability leak / Model output                                                                                                | texte actuel                                | aucune probabilité                                                |
| `PointPanel.tsx`                 | ligne TSLM du panneau plein écran                                                                                                      | texte actuel                                | `tslm.status`                                                     |
| `MonitorReplay.tsx`              | bandeau, résumé « Model output », pied de carte                                                                                        | texte actuel                                | `tslm.monitorBanner`, `tslm.status`                               |
| `InspectRecording.tsx` §06       | note de décision                                                                                                                       | texte actuel                                | `tslm.decision`                                                   |
| `src/pipe/api/main.py`           | `/health` tslm `available:false`, `/predict` 503, `/evaluation` 404                                                                    | inchangé                                    | **inchangé** : le benchmark n'est pas une inférence live          |
| `docs/pitch/leakless-pitch.html` | slide 4 ligne `["C4","TSLM",null,null,"tslm"]` (l. 336) et rendu l. 372 ; slide 2 l. 234 ; slide 5 l. 252 et l. 295 ; notes l. 324-327 | NOT EVALUATED YET                           | valeurs imprimées par le script d'ingestion, à reporter à la main |
| `docs/pitch/README.md`           | fiche de préparation                                                                                                                   | idem                                        | idem                                                              |

## 5. Procédure mécanique (après CONTRACT PASS / PROVENANCE PASS / FINAL REPORT GENERATED)

```bash
python3 frontend/scripts/ingest-official-tslm.py <…/artifacts/final_evaluation> --contract PASS --provenance PASS
# refuse : commit de génération différent de 3e4e73ab… (aucun amendement accepté),
# dossier déjà rempli (sauf --replace) ; copie atomique via un dossier temporaire
cd frontend && npm test && npm run build
grep -rq SYNTHETIC-FIXTURE dist && echo "REFUS" || echo "dist propre"
```

Puis deck : remplacer la ligne C4 et les phrases « will be scored… » par les valeurs imprimées
(C4 AUC, lectures TSLM vs C1), selon le cas A/B/C ci-dessus. Aucun autre changement.

## 6. Adaptateur modèle de Safoan (non intégré)

`origin/feat/demo-temporal-building` a été avancé à `5beb344` par la fusion `bba67ff`
(« integrer le modele V2 fiable dans la demo ») : 320 fichiers, dont `src/pipe/api/model_service.py`,
`/predict` live, `ModelReadout` avec bouton « Run TSLM V2 », score brut et décision
LEAK-ASSOCIATED/NON-LEAK. Branche **non tirée** ici.

- Il attend `PIPE_TSLM_CHECKPOINT`, `PIPE_TSLM_DECISION`, `PIPE_TSLM_VALIDATION_EVIDENCE` et
  `pipe.tslm.coherent.CoherentPredictor` : artefacts et code **V2 uniquement**.
- Il ne lit ni `metrics.json` ni `comparison.json` : il ne peut pas consommer le résultat officiel.
- Il afficherait un score par enregistrement non calibré à côté d'une carte C4 évaluée sur un autre
  objet (run gelé V1 ou autre) : mélange inférence live / preuve de benchmark.
- Il réécrit la définition TSLM et les notes de décision : conflit direct avec le texte ci-dessus.

Garder la démo sur la voie benchmark seule ; toute reprise de cet adaptateur exige que le
checkpoint servi soit celui de `runs[<tslm>].checkpoint` et une validation scientifique des
probabilités par enregistrement.

## 7. Fixture synthétique

`frontend/tests/fixtures/synthetic-official-tslm/` : chiffres TSLM inventés (0.111, 0.222)
marqués `SYNTHETIC`, greffés sur le vrai rapport contrôles seuls du tag gelé. Importée seulement par
les tests ; le lecteur de production la refuse ; un test vérifie qu'aucun module de production ne
l'importe.
