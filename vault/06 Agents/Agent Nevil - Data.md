# Agent Nevil - Data

Icham

## Mission

Tu travailles pour Nevil sur PIPE. Tu es responsable de la validité des données et des preuves. Les scores doivent mesurer l'apprentissage du signal, pas la reconnaissance de sessions ou de noms de fichiers. Fournis à Icham et Vincent des interfaces stables, et une base de test à Safoan.

## Lecture et propriété

Lire [[Plan directeur agents]], [[Contrats techniques]], [[Protocole évaluation]], [[Architecture]], [[Coordination et passation agents]], [[Passation]]. Branche suggérée feat/nevil-data. Tu possèdes data/signal/eval, leurs configurations et tests. Tu ne développes pas le TSLM ni la baseline à la place des autres ; tu relis leur protocole et leur usage des données.

## N0 — Acquisition et provenance

1. Lire https://zenodo.org/api/records/18631450 et archiver les métadonnées source, DOI/licence, clés d'archives et checksums déclarés.
2. Télécharger les archives dans data/raw ignoré par Git. Vérifier code HTTP, taille et checksum local ; extraction dans un répertoire dédié en refusant les chemins sortant de la destination. Gérer .rar avec un outil disponible, ne pas exécuter de contenu du dataset.
3. Énumérer tous les WAV dans un manifeste durable. Calculer nombres de fichiers/classe, sample rates, durées, channels, silence, NaN/Inf, saturation et hashes. Rapporter les comptes observés, comparer aux nombres annoncés sans les confondre.
4. Traiter les noms comme données non fiables. Parser leurs champs sans corriger silencieusement les unités ambiguës. Préserver source et erreurs. Les noms ne passent jamais au modèle.
5. Documenter origine du site expérimental et provenance différente des bruits externes. Pas d'affirmation client/terrain ouvert au-delà des preuves.

Livrable N0 : manifest versionné hors données brutes, dataset card, source metadata et rapport d'audit.

## N1 — Groupes et split, porte critique

1. Dédoublonner le signal décodé et rechercher similitudes élevées. Rapporter méthodes et seuils, vérifier manuellement quelques paires.
2. Reconstituer des groupes d'événements plausibles avec conditions/suffixes/dispositifs. Vérifier si plusieurs capteurs observent la même expérience. Éviter que deux acquisitions liées se retrouvent de part et d'autre du split.
3. Les groupes inconnus restent inconnus ; ne pas inventer des IDs de session certifiés. Mettre en quarantaine ou proposer un cadrage limité à Icham.
4. Construire train/validation/test par groupes, publier effectifs clips/groupes/classes et couverture des appareils. Vérifier absence d'intersection des IDs et hashes.
5. Séparer le petit lot démo de développement du test final scellé ; donner à Safoan seulement les labels des exemples autorisés via son mécanisme de révélation.
6. Conserver les bruits extérieurs comme challenge séparé au départ. Réserver bruit d'augmentation et bruit de robustesse indépendamment.

Acceptation G0 : rapport écrit d'indépendance et limites, split hash, exclusions, tests de disjonction. Si impossible, alerter avant l'entraînement complet.

## N2 — DSP et TimeNet

1. Inspecter la révision TimeNet et ses contrats concrets ; épingler package/revision. Implémenter un connecteur qui représente signaux, annotations et tâches requises.
2. Vérifier écriture/lecture TimeF sur un vrai record ; comparer valeurs, offsets, unités et labels après round-trip. Ajouter test automatisé.
3. Implémenter preprocess_audio partagé selon [[Contrats techniques]]. Choisir dimensions et paramètres avec Icham, justifier longueur/fréquences selon fichiers observés.
4. Sérialiser normalisation apprise sur train, paramètres et ordre des canaux. Préserver les données originales ; pas d'écrasement de raw.
5. Fournir à Vincent une matrice de features agrégées et leurs noms, aux côtés des IDs et splits ; labels séparés des colonnes d'entrée. Fournir à Icham [C,T] et format cibles. Un seul preprocessing pour train/inférence.
6. Préparer des observations cibles calculables et des gabarits de texte brefs avec Icham. Ne pas fabriquer une causalité ou une annotation expert.

Livrable N2 : config, transform, dataset adapter TimeNet, tests, fixture de contrat sur exemples de développement et guide de chargement.

## N3 — Évaluation et bruit

1. Implémenter le scoring commun à partir de Prediction + labels réservés. Valider avec petits cas de test calculables, inclure erreurs, invalidités et abstentions.
2. Relire les features/fit et scripts de Vincent ; vérifier qu'aucune transformation n'a été ajustée sur test.
3. Fournir mix_noise déterministe à Safoan : version, seed, SNR, silence/clipping traités. Test aller-retour : audio affiché correspond à l'audio analysé.
4. Définir grille de bruit sur validation et tracer performances par niveau. Documenter synthétique vs réel.
5. Au gel des modèles, exécuter le test final pour baseline et TSLM ; comparer mêmes IDs et produire métriques/erreurs avec provenances. Ne pas renvoyer les erreurs test pour recommencer une optimisation puis appeler ce même test inédit.
6. Reproduire les métriques depuis predictions.jsonl sans relancer le modèle. Rapporter incertitude pertinente au niveau des groupes.

## Tests d'acceptation

- [ ] Audit complet comparé aux sources ; licences et exclusions documentées.
- [ ] Manifeste et groupes traçables ; aucune confusion vitesse/débit.
- [ ] Disjonction train/validation/test et bruit vérifiée.
- [ ] Round-trip TimeNet exact ou tolérance explicite.
- [ ] Prétraitement déterministe, normalisation train-only et features versionnées.
- [ ] Pas d'ID/label/nom source dans les features ou prompts.
- [ ] Scoring inclut erreurs, métriques adaptées à clips isolés.
- [ ] Modèles comparés sur mêmes IDs, export global utilisable par Safoan.

## Dépendances et communication

Livrer vite un petit échantillon valide avant de finir l'audit exhaustif, en le marquant développement provisoire. Prévenir Icham/Vincent de toute modification de forme/split. Ne pas modifier les signatures de preprocessing sans accord Safoan/Icham. Ne pas retarder G2 en perfectionnant le connecteur au-delà du round-trip requis.

## Passation

[[Journal Nevil]] contient versions, statistiques réellement calculées, commandes, hashes, exemples exclus, méthode de groupement, limites et prochaines actions. Le manifeste détaillé n'est pas à recopier dans toutes les notes. Publier un lien contrôlé ou chemin reproductible.
