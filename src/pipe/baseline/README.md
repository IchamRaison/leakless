# Baseline PIPE de Vincent

V0 CPU : chargement contrôlé, Random Forest, validation, sauvegarde joblib, prédiction et exports. Les données et features réelles de Nevil ne sont pas encore disponibles dans ce checkout. Une exécution synthétique vérifie le code ; elle ne mesure aucune performance acoustique PIPE.

Périmètre : `src/pipe/baseline`, `configs/baseline`, `tests/baseline`, branche `feat/vincent-baseline`. Le DSP, les splits et le scoring final restent à Nevil ; le schéma partagé à Safoan, le bootstrap Python commun à Icham. Aucun `pyproject.toml` ni lockfile commun modifié.

## Installer et vérifier

Depuis la racine du dépôt, Python 3.12 :

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r configs/baseline/requirements-cpu.txt
env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/baseline -q
```

Versions CPU épinglées depuis l'environnement réellement installé. Validation initiale sur macOS arm64 ; Linux et l'environnement partagé restent à vérifier. Les dépendances sont proposées à Icham pour intégration, sans modifier son environnement.

## Exécuter la V0 synthétique

Choisir un nouveau dossier à chaque run : aucun run existant n'est écrasé.

```sh
env PYTHONPATH=src .venv/bin/python -m pipe.baseline.fixture --output /tmp/pipe-fixture-v0
env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pipe.baseline.train --config /tmp/pipe-fixture-v0/config.json
```

Le générateur crée uniquement des tableaux synthétiques, sans audio ni DSP : 24 lignes train, 12 validation et un manifeste comprenant aussi 4 IDs test sans leurs signaux/labels. Ces nombres servent aux tests mécaniques, pas à recommander un ratio à Nevil. Les résultats, métadonnées et prédictions portent `execution_mode=development_fixture`. Un modèle entraîné sur fixture garde ce marquage même si son entrée est ensuite déclarée réelle.

## Entrées à convenir avec Nevil

L'adaptateur JSON est **provisoire**, distinct des contrats TimeNet et du manifeste global de Nevil. Il ne crée jamais de split. Il consomme une projection de ses affectations, puis vérifie la disjonction des groupes, test compris. Nevil doit confirmer ou remplacer cet adaptateur et l'ordre exact des features avant G1.

`split.json` : objet `schema_version=baseline-split-v0`, `assignments` liste de `{sample_id, event_group_id, split}`. Splits admis : train, validation, test, quarantine. Aucun label ou signal dans ce fichier. Le hash demandé dans la configuration est le SHA256 des **octets exacts de tout ce fichier**, test/quarantaine compris, UTF-8. Changer espaces, ordre ou contenu change le hash. Nevil transmet ce fichier et son hash ; aucun recalcul automatique pour accepter un split modifié.

`features.json` : `schema_version=baseline-features-v0`, `execution_mode` (`live` ou `development_fixture`), `dataset_version`, `feature_version`, `preprocessing_version`, `feature_names` ordonnés, `samples`. Chaque sample contient exactement `{sample_id, input_sha256, features, label}`. `label` est l'entier 0 (no_leak) ou 1 (leak). `features` est une liste numérique dans l'ordre déclaré. `input_sha256` identifie le signal source convenu avec Nevil, pas son nom. Dans la fixture seulement, il identifie les octets des features synthétiques.

Tous les samples train/validation du manifeste doivent être présents une seule fois ; aucun sample test/quarantine n'est accepté dans le fichier de développement. Les données réelles restent chez Nevil jusqu'à livraison autorisée. Les labels/IDs restent séparés de X. Les deux classes sont exigées dans train ; classe manquante en validation signalée. Un signal déclaré identique dans train et validation est refusé.

La configuration demande en plus les features attendues et leurs versions. Les chemins sont relatifs au fichier de configuration, jamais au dossier courant. `configs/baseline/default.json` décrit les fichiers réels attendus et échoue explicitement tant qu'ils manquent. Les paramètres RF sont simples, seed 42 et un seul worker ; aucun réglage automatique ni utilisation du test. La classe majoritaire vient du train et est évaluée sur validation, avec prédiction déterministe de la classe la plus fréquente (0 en cas d'égalité).

NaN/Inf, overflow float32 et écrasement à zéro par sous-dépassement sont rejetés par choix de contrat jusqu'à accord de Nevil, même si sklearn sait gérer certains NaN. Aucune normalisation, imputation, sélection de features ou transformation DSP n'est apprise ici. Si imputation validée plus tard, son fit doit rester limité à train dans une Pipeline.

## Artefacts et inférence

Chaque run produit `model.joblib`, `metadata.json`, `predictions.jsonl`, `predictions.csv`, `metrics.json` et `run.json`. Ce dernier n'est écrit qu'après livraison complète et contient les hashes de tous les autres fichiers. En cas d'erreur d'inférence, le run échoue sans livrer une liste tronquée de prédictions comme résultat complet. Les métriques sont limitées à validation et `benchmark_eligible=false` en V0, même pour des données déclarées réelles ; Nevil valide le scoring avant toute publication.

Métadonnées : versions Python/sklearn/NumPy/SciPy/joblib, config/seed, versions features/DSP/dataset, hashes données/split/config/code, commit, état local modifié, effectifs et modèle. `model_version` identifie le run ; SHA256 identifie son contenu. Les artefacts vont hors Git, par exemple sous `/tmp` pour les fixtures. Ne jamais ajouter les WAV/poids au commit ; convenir avec Icham des exclusions runtime avant de travailler dans `data/` ou `artifacts/`.

`predict_baseline(artefact, exemple)` reçoit les agrégats déjà préparés par Nevil, sans refaire le DSP. L'exemple contient exactement sample_id, input_sha256, features, feature_names, feature_version, preprocessing_version, execution_mode. La fonction retourne tous les champs Prediction v0.1, sous forme de dictionnaire à faire valider par Safoan. Aucun schéma partagé n'est redéfini. `class_scores` suit les vraies `classes_` ; scores bruts, aucune confiance calibrée ni explication inventée. Aucun label, groupe ou nom source dans la sortie. Cette fonction calcule une nouvelle prédiction et refuse le mode replay en entrée : le rejeu consiste à afficher un résultat archivé, tâche de l’application de Safoan, sans rappeler le modèle. Une perturbation synthétique ne constitue pas un replay.

Chargement programmatique : `charger_modele(chemin, sha256_attendu, artefact_de_confiance=True)`, puis `predict_baseline`. Joblib peut exécuter du code : ce paramètre est réservé à un artefact dont l'origine est connue. Un checksum vérifie l'intégrité, jamais l'identité du producteur. Les versions sklearn/NumPy/SciPy/joblib différentes sont refusées avec le nom de la dépendance, y compris un avertissement sklearn pendant le chargement. Un correctif Python différent au sein du même majeur.mineur est toléré ; cette règle est testée par substitution de métadonnées, sans prétendre avoir testé un second interpréteur. La reproductibilité est vérifiée dans le même environnement, pas garantie entre machines/versions.

La CLI de rechargement, également utilisée par les tests dans un processus neuf :

```sh
env PYTHONPATH=src .venv/bin/python -m pipe.baseline.predict --help
```

Elle attend le modèle, son hash de référence, un fichier JSON d'inférence sans label, un chemin de sortie et la déclaration explicite d'origine de confiance. La sortie CLI est créée exclusivement et ne remplace jamais un fichier existant. Les tests vérifient l'égalité exacte des classes et scores après rechargement sur toute la validation synthétique.

## Limites du contrôle

Les noms de colonnes et les versions détectent un contrat incohérent, pas une colonne frauduleusement renommée ou une permutation non déclarée des valeurs. Les hashes déclarés ne prouvent pas l'indépendance physique des événements ; Nevil doit auditer doublons, quasi-doublons et groupes depuis les WAV. Le pilote n'a pas accès aux signaux de test, donc il ne peut pas y rechercher des doublons acoustiques. Un score parfait synthétique ou réel ne lève pas ces limites.

Prochaine étape : Nevil livre une petite fixture réelle et confirme ordre/version des agrégats, manifeste complet et hash. Vincent adapte uniquement le chargement au contrat validé, puis entraîne V1 sur développement. Safoan valide la réponse Prediction ; Nevil conduit le test final après gel.

## Sources vérifiées et décisions

- Fit limité au train et prétraitement appris dans une Pipeline s'il devient nécessaire : [pièges et fuite de données, scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
- `classes_`, conversion float32, seed et paramètres : [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
- Confiance dans joblib et incompatibilité des versions : [persistance des modèles](https://scikit-learn.org/stable/model_persistence.html).
- Convention lignes réelles, colonnes prédites : [matrice de confusion](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).
- Séparation par groupes : [validation croisée de données groupées](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data). Cette V0 consomme les splits existants, sans utiliser de splitter.

Recherche et revue Codex/Claude tracées dans le bus local. Les choix ne prétendent pas maximiser le score avant d'avoir les données : priorité à une comparaison reproductible conforme à la fiche d'Icham.

## Compléments exploratoires demandés après V0

Nevil peut valider une livraison **sans entraînement**, avant de la transmettre :

```sh
env PYTHONPATH=src .venv/bin/python -m pipe.baseline.valider --config /tmp/pipe-fixture-v0/config.json
```

Options `--data` et `--split` pour remplacer les chemins sans modifier la configuration. Le hash attendu reste contrôlé. Sortie JSON lisible : effectifs et groupes par split, classes train/validation, colonnes attendues/reçues. Les labels test restent inconnus. Code 0 pour contrat valide, 1 pour données/configuration invalides ; erreur de syntaxe CLI traitée par argparse. Les erreurs de valeurs/forme d'une ligne donnent son sample_id. Ce validateur ne réalise pas l'audit G0 de Nevil.

Les analyses sont désactivées par défaut et s'activent explicitement sur un nouveau run :

```sh
env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pipe.baseline.train --config /tmp/pipe-fixture-v0/config.json --analyses configs/baseline/analyses.json
```

Utiliser un dossier `output_dir` neuf si le run simple existe déjà. Aucun score de développement, même réel, n'est promu automatiquement en benchmark.

- **Abstention** : seuil sur le maximum des scores bruts, choisi sur validation pour maximiser le macro-F1 parmi les réponses retenues sous contrainte de couverture minimale. Cet objectif exploratoire reprend une métrique du protocole mais ne fixe pas les coûts métier ; Nevil doit le valider et figer le point de fonctionnement avant le test. Grille/configuration explicites ; en cas d'égalité, couverture supérieure puis seuil inférieur. Le modèle principal est toujours fit sur train uniquement. Seuil sauvegardé dans l'artefact et appliqué à toute nouvelle inférence, sans champ permettant au client de le remplacer. La validation sert aussi à sélectionner ce seuil : les chiffres sont descriptifs et optimistes, `independent_evaluation=false` le signale. Le rappel global compte les abstentions sur fuite comme des non-détections.
- **Exports sélectifs** : toutes les lignes sont conservées. Une abstention porte prediction=null, abstained=true et un motif ; les scores restent raw. Avec abstention, `metrics.json.baseline=null` évite d'afficher une métrique pleine couverture trompeuse. `selective` contient la matrice 2×3, couverture, nombre d'abstentions, exactitude retenue, métriques des réponses retenues et rappel calculé sur **toutes** les fuites. Le CSV inclut aussi abstained/abstention_reason. Sans l'option d'analyse, le run V0 pleine couverture reste disponible ; ne pas comparer des scores conditionnels à des scores pleine couverture sans indiquer leurs dénominateurs.
- **CV groupée** : GroupKFold sur le développement uniquement, modèles clonés indépendants, sans modifier le modèle principal ou le split final. Moyenne et dispersion par pli, pas un intervalle de confiance. Si un pli perd une classe ou s'il manque des groupes, statut explicite et aucune moyenne des seuls plis réussis.
- **Bootstrap de groupes** : tirage avec remise de groupes entiers de validation, modèle figé et sans abstention. Intervalle percentile à 95 %, graine/repetitions enregistrées. Trop peu de groupes ou perte de classe dans des répétitions : intervalle=null, motif et compte explicites. Le minimum de groupes configurable est une règle prudente, pas une preuve d'indépendance. Cet intervalle ne mesure pas l'incertitude liée au réentraînement et n'évalue pas le seuil sélectionné.
- **Amplitude** : la forêt sur clip_log_rms utilise uniquement cette feature fournie par Nevil et les lignes train ; score sur validation. Si elle manque, statut unavailable sans substitution. Ce comparateur ne prouve pas une causalité et ne remplace pas une vraie perturbation de gain dans le DSP.

Les rapports se trouvent dans `metrics.json.analyses`, explicitement `development_exploration`, avec le hash des options dans les métadonnées. L'API Prediction reste inchangée ; les rapports complémentaires attendent la validation de Nevil avant intégration à son scoring.

## Proposition de features pour Nevil

```sh
env PYTHONPATH=src .venv/bin/python -m pipe.baseline.proposer_features --bands 4
```

Le résultat est une **proposition**, pas un choix de DSP. L'exemple enregistré dans `configs/baseline/proposition-features.json` utilise quatre bandes uniquement pour illustrer la liste complète. Nevil décide du vrai nombre de bandes et des paramètres après audit ; relancer avec ce nombre. Ordre : par bande croissante, mean, std, q25, q50, q75 sur l'axe temporel, puis clip_log_rms calculé sur le signal avant normalisation individuelle. Échelle PCM, réduction mono et epsilon à documenter par Nevil. Aucun calcul acoustique ajouté dans ce module. La configuration réelle par défaut reste à renseigner depuis le contrat accepté, jamais depuis cet exemple sans validation.

Sources des compléments : [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [séparation des données pour ajuster un seuil](https://scikit-learn.org/stable/modules/classification_threshold.html), [tirage avec remise dans NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html). La politique d'abstention et le bootstrap groupé ici sont notre implémentation explicite de développement, pas un mécanisme de calibration sklearn ni une validation scientifique automatique.
