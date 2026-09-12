# V1 TSLM — scoring, entraînement et livraison au harness

Icham

## Statut

Code de campagne et d'export : `1199789f471acf79f7e82604e2a15acd45e0b2f7`.
Scoring introduit à `4456c3168f22b17d81bbb0bc6c2a91e34e14bb30`.
Les 19 tests CPU passent dans le runtime H100 reconstruit ; le score réel a été
vérifié sur quatre clips **train seulement**. Preuves : [evidence/tslm-v1](evidence/tslm-v1/).
La campagne de 600 étapes est terminée ; époque 4 retenue parmi les trois candidats
sur validation, bundle autonome rechargé dans un processus neuf avec score et texte
identiques. **T0, T1, T2 et T3 sont exécutés et conformes ; aucune métrique finale calculée ici.**
Exporteur renforcé : `5a29d6eb9ceca39fdce829eb22dbbfd67b25589b` ; le modèle,
preprocessing, scoring et stress sont inchangés depuis le code d'entraînement.

Modèle `pipe-qwen3.5-4b-v1-1199789f-e4`, bundle sur la H100 :
`/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle`.
SHA-256 de `checksums.json` : `b95569c50f5e1bbf3533bddc92530b5f285eb25dec999f83999f60ffd912d1ae`.
Les paramètres temporels retenus ont changé et le SHA intégral de Qwen gelé est
identique avant/après ; preuves dans `training-report.json` et `reload.json`.

Les quatre dossiers contiennent chacun seulement `metadata.json` et `predictions.csv`,
402 IDs val/test et deux colonnes exactes, contrôleur conforme :

| Run | Prédictions | Métadonnées |
| --- | --- | --- |
| T0 original | [CSV](../artifacts/tslm_runs/tslm-v1/predictions.csv) | [JSON](../artifacts/tslm_runs/tslm-v1/metadata.json) |
| T1 inversion | [CSV](../artifacts/tslm_runs/tslm-v1-T1/predictions.csv) | [JSON](../artifacts/tslm_runs/tslm-v1-T1/metadata.json) |
| T2 blocs 250 | [CSV](../artifacts/tslm_runs/tslm-v1-T2/predictions.csv) | [JSON](../artifacts/tslm_runs/tslm-v1-T2/metadata.json) |
| T3 phase | [CSV](../artifacts/tslm_runs/tslm-v1-T3/predictions.csv) | [JSON](../artifacts/tslm_runs/tslm-v1-T3/metadata.json) |

T0 publié à `186c45a` **avant** lancement des stress ; T1–T3 publiés à `7c04c9a`.
Le bundle/reload, la méthode
de score, les trois configurations comparées et le code de modèle sont identiques
dans les quatre runs ; seules les transformations/identités/horodatages et
prédictions changent. Le SHA du bundle est toujours identique après les stress.
Les logs de conformité se trouvent dans [evidence/tslm-v1](evidence/tslm-v1/).

Les scores restent bruts et non calibrés ; aucun réglage n'a été effectué après
l'inspection du contrôleur. La qualité finale et la sensibilité temporelle sont
à établir par Nevil, pas déduites ici. GPU revenu à 0 Mio après tous les processus ;
instance laissée allumée, aucun service public déployé.

La V0 et son bundle sont conservés. L'environnement verrouillé et les sources
TimeNet/Qwen restent ceux de [TSLM_V0.md](TSLM_V0.md). Aucun entraînement du
décodeur Qwen, aucune génération de faux pourcentage, aucune métrique finale ici.

## Score défini avant la sélection

Les deux continuations sont `leak;` (tokens `[273,570,26]`) et `no_leak;`
(`[2083,11414,570,26]`). Même signal et prompt, log-probabilités conditionnelles
sommées jusqu'au point-virgule, softmax stable sur les deux sommes. Pas de moyenne
par longueur, EOS ou explication dans le score. La frontière de tokens est vérifiée
contre les réponses complètes de l'entraînement.

`probability_leak` est une probabilité relative à ces deux continuations, brute et
non calibrée ; elle n'est pas une probabilité physique de fuite étalonnée terrain.
Les métadonnées/labels/IDs ne sont jamais transmis au modèle. Les contrôles train
retrouvent exactement le même score depuis WAV, signal flottant et séries, ainsi
qu'au rechargement des entrées et avec un softmax manuel.

```python
from pipe.tslm.predict import Predictor

predictor = Predictor("CHEMIN_DU_BUNDLE")
probability_leak = predictor.score(wav_bytes)
prediction = predictor.predict(wav_bytes)  # contrat Safoan inchangé, texte séparé
```

`score_waveform(array, 8000)` accepte le brut flottant transformé avant le même
preprocessing, sans réencoder de WAV. `score_series(array)` attend `[4,64]` déjà
prétraité. Une seule inférence concurrente, erreurs explicites, pas de score de
secours. `predict()` conserve `score_type=none` : sa classe générée n'est pas le
score exporté au harness et peut différer de l'argmax des continuations scorées.

## Campagne préenregistrée

[v1.json](../configs/tslm/v1.json) : encodeur/projecteur neufs, Qwen 3.5-4B gelé,
598 clips train, batch 8 (dernier lot de 6), huit époques / 600 étapes. Seuls
candidats : fins d'époques 2, 4 et 8, sur une trajectoire commune.

Sélection sur **validation seulement** : ROC-AUC des médianes par groupe ; égalité
→ checkpoint le plus précoce. 208 clips, 42 groupes heuristiques dont 8 non-fuite,
pas des sessions indépendantes prouvées. Huit diagnostics textuels fixés avant
apprentissage, exclus de la sélection. Aucun cache ou WAV test ouvert par la
campagne. La préinscription et les essais réels sont conservés ; aucune LoRA,
nouvelle représentation, calibration ou recherche supplémentaire implicite.

Le dossier de campagne contient préinscription, logs, trois petits checkpoints
temporels/optimiseurs et diagnostics de développement ; un seul `bundle/` autonome
embarque Qwen et les poids retenus, avec licences, lockfile et empreintes.

## Commandes de reproduction

Depuis un checkout du code identifié, avec le Python de l'environnement verrouillé
et les packages `src` / `scripts/timenet` sur son chemin d'import. Chaque sortie
doit être **neuve**. Remplacer les chemins et `REVISION_CODE` par ceux du transfert
réel ; une archive distante ne contient pas nécessairement `.git`.

```bash
python -m unittest discover -s tests/tslm -v
python -m pipe.tslm.campaign \
  --config configs/tslm/v1.json --manifest manifests \
  --prepared PREPARED --data-root WAVROOT --base QWEN_BASE \
  --output CAMPAGNE_NEUVE --code-revision REVISION_CODE \
  --environment-lock requirements-ml.lock

# Nouveau processus après la fin du training, avant toute inférence test.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/tslm/check_v1_reload.py \
  --checkpoint CAMPAGNE_NEUVE/bundle --output RELOAD_NEUF.json

python scripts/tslm/export_run.py \
  --checkpoint CAMPAGNE_NEUVE/bundle --reload-report RELOAD_NEUF.json \
  --data-root WAVROOT --manifests manifests --output RUN_T0_NEUF \
  --run-id tslm-v1 --transform T0 --code-revision REVISION_CODE
```

T1/T2/T3 : même checkpoint et rapport reload, nouveau dossier/run_id, transformation
correspondante. Aucun réentraînement. Livrer T0 avant d'attendre ces trois runs.

## Contrat et provenance Nevil

Reprise sélective byte-identique de six fichiers à
`6dfdf63580bc15ce6a1cf0817b9da3569553aca1` : contrat, schéma, contrôleur, chargeur
de split, contrat Python et transformations. Seul `harness/__init__.py` est un
emballage minimal pour ne pas importer le moteur final absent de notre branche.

`clip_rng` utilise le correctif SHA-256 officiel `b23601a`. Le test CPU vérifie la
graine publiée et l'identité de T2/T3 sous deux processus à hash Python différent.
T2 = 250 échantillons (31,25 ms) sans reste ; certains blocs peuvent rester fixes.
Les transforms s'appliquent au brut, puis au preprocessing T0, sans requantification.

L'export exige un reload validé du même bundle, sa méthode de score figée et la
déclaration de non-réglage sur test. Il réutilise `check_run.py --template` dans un
dossier neuf, puis `--run --inspect` ; il n'appelle aucun évaluateur final.
Chaque dossier contient **seulement** `metadata.json` et `predictions.csv`, exactement
`clip_id,probability_leak` pour les 402 IDs val/test, sans seuillage ni arrondi
destructeur. Les empreintes du code, du bundle, du scoring et la version NumPy
figurent dans la provenance, ainsi que le nombre réel de candidats comparés.

L'inspection du contrôleur mélange val/test : aucune adaptation n'en découle.
Un avertissement de sortie binaire n'est pas un échec du contrôleur ; il ne faut
jamais fabriquer une dispersion de scores pour le faire disparaître.
Nevil est seul responsable des seuils et de toutes les métriques finales.

Les performances de classification et les stress ne prouvent pas un taux de
fausses alertes par jour ni un délai de détection en surveillance continue.
