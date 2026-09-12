# Porte de parité canonique V2

`scripts/tslm/check_v2_parity.py` vérifie la correction **sans entraîner, calculer
de métrique qualité, choisir un seuil, ni lire un WAV/cache test**. Il ne remplace
pas le diagnostic historique `diagnose_parity.py`, qui reste inchangé.

## Entrées et portée

Le bundle doit déclarer `rms-f32-hann256-hop128-band4-log1p-v2`. Utiliser une copie
corrigée séparée avec les poids V1 inchangés ; ne jamais modifier le bundle V1.
Les répertoires de sortie doivent être neufs.

- `--prepared` : préparation canonique V2, caches `train.npz` et `val.npz`.
- `--historical-prepared` : préparation V1 historique, ces mêmes deux caches.
- `--timef-version` : version TimeF historique qui a alimenté l'entraînement V1.
- `--data-root`, `--manifests` : WAV originaux et manifeste gelé vérifié.
- `--checkpoint` : bundle autonome corrigé, dont les checksums sont vérifiés par `Predictor`.

Le gate couvre les 208 validations et le seul train témoin `c0128b879694e`.
Il vérifie les MD5 des WAV puis l'égalité exacte, y compris le type, des tenseurs
produits par le dispatcher WAV canonique, l'appel canonique explicite, TimeF et
les deux caches. TimeF est filtré avant lecture des signaux, sans annotations.
Les caches train/val complets sont chargés et empreintés ; seul le train témoin
est comparé à son WAV et scoré. Le gate ne prétend donc pas vérifier par WAV les
597 autres entrées train : leur préparation canonique reste un contrôle séparé.

Sur ces 209 clips il appelle réellement `Predictor.score`, `score_waveform` et
`score_series`, puis le scorer du même modèle avec les vrais batches 1, 2 et 4,
en ordre normal et inversé. Le dernier batch partiel n'est pas complété par
duplication. Les identifiants ne sont jamais transmis au modèle.

La borne **absolue de 10⁻⁶, sans tolérance relative**, porte sur l'écart maximal
entre tous ces chemins, tailles et ordres, puis entre les deux processus. Un
échec de batch n'est pas converti en succès batch=1 ; il bloque et reste visible.
Les états encodeur/projecteur/LLM sont empreintés avant et après les inférences.

## Deux lancements distincts

Depuis le runtime ML du dépôt, remplacer les chemins en majuscules :

```bash
python scripts/tslm/check_v2_parity.py \
  --checkpoint BUNDLE_CORRIGE --prepared PREPARED_V2 \
  --historical-prepared PREPARED_V1 --timef-version TIMEF_V1 \
  --data-root WAV_ROOT --manifests manifests --output GATE_PREMIER

python scripts/tslm/check_v2_parity.py \
  --checkpoint BUNDLE_CORRIGE --prepared PREPARED_V2 \
  --historical-prepared PREPARED_V1 --timef-version TIMEF_V1 \
  --data-root WAV_ROOT --manifests manifests --output GATE_RELOAD \
  --reference GATE_PREMIER/report.json
```

Le premier lancement peut terminer avec code 0, mais reste
`status="awaiting_fresh_process"`, `all_checks_pass=false`. Il n'autorise aucune
étape dépendante. Le deuxième relance toute la matrice, exige le même hôte et un
PID distinct, les mêmes poids, sources, versions runtime, seeds et entrées.
Seul le second peut fournir `all_checks_pass=true`.

`--threshold` est optionnel et doit recevoir uniquement un seuil **déjà figé**.
Les franchissements et scores à moins de 10⁻⁶ sont rapportés séparément : même
un écart numériquement accepté peut changer une décision proche du seuil. Aucun
seuil n'est sélectionné ou utilisé pour modifier les scores. Sans cette option,
aucune décision seuillée n'est inventée.

## Contrat du reçu

`report.json` porte `schema="pipe-parity-v2"`, `all_checks_pass` et les cinq
booléens `checks.canonical_inputs_exact`, `bundle_unchanged`,
`adapter_scores_within_atol`, `batch_scores_within_atol`,
`fresh_process_verified`. Un consommateur doit exiger **tous** à `true`, pas
seulement un code retour 0.

`provenance` contient les versions canonique/historique, `score_atol=1e-6`,
`seed=20260912`, les 209 `clip_ids`, le SHA du manifeste, `cache_sha256.train/val`
(canoniques), `historical_cache_sha256.train/val`, `checkpoint_checksums_sha256`,
`temporal_sha256`, `scoring_spec`, les `state_sha256`, `source_sha256` (notamment
`model.py`, `predict.py`, `preprocessing.py`, `connector.py`), `runtime` et
`process`. Le consommateur compare les empreintes à ses propres entrées.
`reference_sha256` lie le second reçu au premier. `records` lie WAV et tenseurs ;
`scores` conserve les valeurs brutes des neuf chemins pour chaque clip.

Une exception, une entrée divergente ou une matrice incomplète laisse un reçu
`failed` non réutilisable. Les divergences numériques sont conservées dans le
rapport ; aucun échantillon n'est omis pour obtenir un passage.
