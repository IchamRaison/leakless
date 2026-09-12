# Évaluation qualité V1 - exécution

Icham

## État courant

Nouvel objectif actif : exécuter les six étapes de [[Protocole évaluation#Plan qualité V1 — proposé, non exécuté]]. Les deux fichiers d'objectif ont été lus intégralement. Préparation en cours, **aucune métrique finale TSLM calculée à ce jalon**. Branche isolée `feat/icham-quality-eval`, worktree `/home/animus/ehl-hackathon-zurich-quality-eval`, base `b4c8059` ; les chantiers ML et simulateur existants sont préservés.

## Règles et preuves avant résultats

- `docs/evidence/quality-v1/preregistration.json` fige protocole, population, comparateurs, seuil/agrégation/bootstrap, audit des 402 textes et empreintes avant évaluation réelle.
- Les huit SHA locaux des fichiers T0–T3 restent identiques à leur publication. Split `7a8716a3…`, audit mapping `1a3bd3c1…`. Bundle H100 vérifié : `checksums.json` SHA `b95569c5…`, poids temporels `183a1b1c…` inchangés.
- Runtime distant existant : Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.8.0+cu128. GPU observé à 0 Mio / 0 % au départ, aucune nouvelle machine/dépendance GPU.
- Recherche Entire retrouve les commits du harness dont `6dfdf63`, pas de transcription consultée. Lecture du code réel : seuil pleine précision disponible dans le bloc `folds.val.threshold` ; l'affichage à six décimales ne doit pas piloter l'audit texte.

## Correction prospective de précision moyenne

Avant ouverture des résultats TSLM, un test jouet démontre une dépendance indue de `pr_auc` à l'ordre des lignes en cas de scores égaux : `[0.5,0.5]` donne 1.0 avec labels `[1,0]` et 0.5 avec `[0,1]`, alors que la précision moyenne standard vaut 0.5 dans les deux cas. Correction minimale prévue/isolée : agréger les scores égaux en un même seuil avant intégration. ROC-AUC, seuil, macro-F1 et bootstrap restent inchangés. Aucun choix motivé par une performance réelle ; provenance différente du fichier amont explicitement tracée. Les phrases de conclusion codées en dur du rapport seront également neutralisées.

## Comparateurs disponibles

Contrôles C0/C1/C2/C2b/C3 : reproduction du pipeline existant, grille et descripteurs gelés ; pas de nouvelle recherche de modèle. TSLM T0/T1/T2/T3 déjà exportés, pas à régénérer.

RF Vincent : branche `feat/vincent-baseline-v1` à `fc837f7b05c4ab7dd8a9eabd420daf65b49db9d4`, 402 exports et compatibilité v2 documentés mais CSV/JSON absents des dépôts/worktrees accessibles. Artefacts annoncés sur son Mac sous `/Users/patrickjane/Projets/ehl-baseline-artifacts/verification-c1-c2/rf-v1-001/nevil/`, existence distante non vérifiée. Demande non bloquante faite à Icham pour les deux fichiers ; inclure seulement un export original vérifié, ne pas réentraîner sa RF ni contourner un contrôle de provenance.

## Prochaine action

Terminer les scripts de réutilisation du harness et d'audit texte, vérifier leurs tests synthétiques, publier la préinscription et le code avant calcul, puis contrôles/T0/comparaisons/stress CPU → audit texte H100 sur 208 validation puis 194 test → rapport de verdict et revue indépendante. Aucun réentraînement du TSLM ni changement de prompt/scoring autorisé par un résultat test.
