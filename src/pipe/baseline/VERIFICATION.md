# Vérification de la V0 baseline

Date : 12 septembre 2026. Travail local sur `feat/vincent-baseline`.

## Résultat mesuré

Commande exécutée depuis la racine :

```sh
env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/baseline -q --junitxml=/private/tmp/pipe-vincent-v0-tests.xml
```

Résultat : **72 tests réussis en 9,66 secondes**, aucun échec. Python 3.12.14, macOS arm64 ; dépendances exactes dans `configs/baseline/requirements-cpu.txt`. `pip check` n'a trouvé aucune incompatibilité de dépendances.

Les tests couvrent les formats, types, valeurs float32 non représentables, groupes communs jusque dans le manifeste test, labels/IDs interdits, classes absentes, samples manquants/répétés, mapping `classes_`, fit train seul, reproductibilité, corruption/version d'artefact, contrat de prédiction, exhaustivité et égalité des exports. Toute la validation synthétique est prédite de nouveau dans des processus indépendants ; les classes et scores sont comparés exactement.

Une boucle distincte fixture → entraînement → artefacts a été exécutée. Ce sont uniquement des données synthétiques, sans WAV, sans TimeNet ni DSP. Aucun score de fixture n'est présenté comme résultat PIPE.

## Revue croisée

Recherche primaire par Codex et Claude, puis lecture mutuelle des raisonnements et revue du code par Claude. Échanges réels consignés dans le bus `agent-bus` sur le Mac de Vincent (messages de conception #271 à #276, revue #278/#280/#281, réponses et corrections #282/#284). Les avis ne sont pas traités comme des preuves sans contrôle du code ou test.

Claude a exécuté la suite initiale : 57 tests passaient aussi de son côté. Ses tests et observations ont conduit à :

- Calculer les métriques depuis les prédictions réellement exportées et vérifier le recalcul, y compris si le chemin d'inférence diverge volontairement du modèle brut en test.
- Tester la concordance exacte CSV/JSONL et l'absence de cibles dans les sorties imbriquées.
- Nommer la dépendance incompatible au chargement, tolérer seulement la différence de correctif Python, convertir l'avertissement sklearn en erreur claire.
- Refuser le sous-dépassement float32, nommer les mauvais chemins de configuration et protéger les fichiers de sortie existants.

Désaccords résolus : un échange de valeurs non déclaré ne peut pas être détecté grâce aux seuls noms de colonnes ; nous contrôlons l'ordre déclaré et documentons la limite. Le manifeste complet permet de contrôler les groupes test sans lire leurs labels. Les versions SciPy/joblib restent épinglées comme NumPy/sklearn, conformément à la recommandation de même environnement de la documentation de persistance. Le calcul d'une nouvelle prédiction refuse `replay` : afficher un résultat archivé appartient à Safoan. La configuration n'est pas libre : ses clés sont strictement contrôlées.

## Compléments exploratoires du 12 septembre

Après la V0 : validateur de livraison sans entraînement, abstention optionnelle, CV groupée, bootstrap de groupes, proposition de features et ablation conditionnelle sur `clip_log_rms`. Commande identique avec rapport `/private/tmp/pipe-vincent-analyses-tests.xml` : **88 tests réussis en 10,78 secondes** côté Codex. Les tests supplémentaires contrôlent les groupes réellement utilisés, les dénominateurs avec abstention, la sélection macro-F1 face à une exactitude trompeuse et le rechargement de la politique sauvegardée.

Claude a lu les modules et exécuté la version intermédiaire : **86 tests réussis en 10,66 secondes**, bus #309. Son objection sur l'objectif de sélection a conduit à remplacer l'exactitude par le macro-F1 parmi les réponses retenues, sous couverture minimale. Cet objectif reste à valider par Nevil ; le rappel global conserve toutes les fuites au dénominateur. Relecture de cette dernière correction demandée dans #311.

Désaccord conservé et expliqué dans #310 : ne pas agréger seulement les plis ayant les deux classes ni filtrer les tirages bootstrap dégénérés pour obtenir un intervalle. Les détails et motifs restent visibles ; l'absence d'agrégat décrit la limite des données. Les analyses sont optionnelles, réservées au développement, et la sélection et la mesure sur la même validation sont explicitement signalées comme optimistes. Aucun nouveau résultat acoustique n'est revendiqué.

## Limites restantes

- Données acoustiques, qualité du split et absence de quasi-doublons : Nevil.
- Accord sur le format de transport provisoire, les features et leurs versions : Nevil et consommateurs.
- Schéma Prediction consommé par la vraie application : Safoan.
- Environnement partagé et installation Linux : Icham et équipe.
- Entraînement réel V1, artefact réel V2, évaluation finale V3 : en attente des données et du protocole validés.

Les tests prouvent la mécanique et les refus attendus sur fixtures, pas une validation métier ou une conformité automatique de tout le projet au challenge. Voir le README du module pour les commandes et la livraison attendue de Nevil.
