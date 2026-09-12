# Agent Vincent - Baseline

Icham

## Mission

Tu travailles pour Vincent sur PIPE. Ton chantier technique est volontairement cadré : entraîner un Random Forest acoustique avec scikit-learn et livrer une comparaison fiable. La baseline est indispensable au challenge et peut battre le TSLM ; ce résultat doit rester visible.

## Lecture et périmètre

Lire [[Plan directeur agents]], [[Architecture]], [[Contrats techniques]], [[Protocole évaluation]], [[Coordination et passation agents]], [[Passation]]. Branche feat/vincent-baseline. Propriétaire src/pipe/baseline, configs/baseline, tests/baseline. Nevil possède le DSP/split et l'évaluation finale. Icham possède le TSLM, Safoan l'UI.

Tu n'as pas à comprendre ni entraîner OpenTSLM. Tu ne fais pas de nouveau téléchargement concurrent ou split aléatoire de ton côté. Tu ne deviens pas automatiquement responsable de tout le pitch.

## Entrées à demander à Nevil

Un tableau X float avec feature_names/feature_version, y séparé, sample_id et event_group_id pour audit, split figé et son hash. Une petite fixture de développement et son mode de chargement. Toute colonne qui identifie classe, fichier, appareil ou session doit être exclue de X dans le MVP. Demander clarification plutôt que deviner l'ordre des colonnes.

## V0 — Préparer le script

1. Créer un module d'entraînement exécutable avec configuration : chemin des données, split, seed, paramètres Random Forest, destination d'artefacts.
2. Définir validation des entrées : dimensions cohérentes, nombre de features attendu, labels binaires valides, sample_id uniques au niveau ligne, classes présentes et pas de NaN non gérés.
3. Si des valeurs manquantes existent, en discuter avec Nevil ; imputation éventuellement fit sur train dans une Pipeline. Pas de fit_transform sur train+validation+test.
4. Une fixture synthétique est autorisée uniquement pour tester le code et doit porter ce statut. Ne jamais en exporter les scores comme benchmark PIPE.

Sortie V0 : tests de parsing/config/shape, script qui échoue clairement si les vraies données manquent.

## V1 — Entraîner sur données de développement

1. Charger le split fourni, ne pas le recréer. Vérifier disjonction des groupes par un assert et enregistrer le split hash reçu.
2. Entraîner RandomForestClassifier avec random_state fixé. Commencer avec une configuration simple, n_jobs raisonnable et class_weight choisi sur validation si besoin. Pas de recherche massive ni utilisation GPU requise.
3. Calculer les prédictions validation et demander à Nevil de confirmer le scoring. Ne pas appeler classes_=[0,1] sans le vérifier : mapper predict_proba par la vraie propriété classes_.
4. Si ajustement : petite grille documentée sur validation/grouped CV développement, pas de sélection sur test. Enregistrer les configurations essayées, même si aucune amélioration.
5. Garder un comparateur majoritaire de sanity check, sans le présenter comme seule baseline sérieuse.

Sortie V1 : run reproductible avec seed, config, provenance et scores validation réels.

## V2 — Sauvegarde et interface

1. Sauvegarder modèle/Pipeline avec joblib dans artifacts/baseline ; enregistrer version sklearn, features ordonnées, preprocessing version, split hash, config, commit et checksum.
2. Recharger dans un processus neuf et comparer les prédictions sur la fixture réelle de validation. Ne charger que des artefacts joblib de confiance : la désérialisation peut exécuter du code.
3. Implémenter predict_baseline compatible avec [[Contrats techniques]]. Elle retourne leak/no_leak, class_scores, score_type raw par défaut et pas de raisonnement inventé. Si mauvaise version de features, refuser plutôt que produire une réponse silencieusement erronée.
4. Fournir cette fonction et un exemple réel à Safoan. Ne pas mettre les étiquettes dans l'objet Prediction.

## V3 — Exports et remise à Nevil

Exporter sample_id, input hash/version si disponible, prediction, scores et modèle en JSONL/CSV. Les labels de validation peuvent figurer dans un fichier d'analyse distinct, jamais transmis comme entrée d'inférence.

Produire matrice de confusion, précision/rappel fuite et macro-F1 via scikit-learn selon protocole validé. Inclure effectifs et avertissement si une classe manque. Les formats et le score final publiés restent sous contrôle de Nevil pour comparer exactement les mêmes exemples.

Transmettre modèle figé à Nevil pour le test final. Si Nevil rapporte une erreur d'implémentation, corriger avec nouvelle version ; si le score est simplement faible, ne pas retuner sur le test. Aider Safoan à afficher les vrais résultats baseline.

## Tests minimaux détaillés

- Dimensions X/y/IDs et ordre feature_names.
- Rejet d'une feature label/ID interdite.
- Pas d'intersection des groupes dans le split reçu.
- Training uniquement sur indices train.
- class_scores associés aux bonnes classes même si ordre inattendu.
- Rechargement modèle = mêmes prédictions à tolérance annoncée.
- Mauvaise feature_version = erreur explicite.
- Export lisible et un résultat par sample_id, pas d'omission d'échec.
- Jeu minuscule où la matrice de confusion attendue est vérifiée en test.

## Definition of done

- [ ] Script + config et commandes réellement exécutées.
- [ ] Baseline apprise sur données réelles autorisées.
- [ ] Validation enregistrée, test final non utilisé pour réglage.
- [ ] Modèle reload testé et artefact identifié par checksum.
- [ ] predict_baseline consommable par Safoan.
- [ ] Exports et tests remis à Nevil.
- [ ] README court : installation, entraînement, prédiction, fichiers attendus et limites.

## Blocages et aide

Données absentes : préparer squelette/test mécanique, demander fixture à Nevil. Features incohérentes : ne pas réécrire le DSP, partager sample_id et erreur. Score parfait suspect : vérifier immédiatement fuite de label, doublons/groupes et ordre d'index avec Nevil. Pas de succès proclamé uniquement parce que fit ne plante pas.

## Passation

[[Journal Vincent]] : commit, commandes, features/split versions, paramètres, scores validation, chemin/checksum modèle, état predict_baseline, tests et prochaine action. Garder clairement la séparation « implémenté », « entraîné », « validé » et « test final évalué ».
