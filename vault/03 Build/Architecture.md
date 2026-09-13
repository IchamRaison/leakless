# Architecture

Icham

## Correction actuelle — C1 + OpenTSLM-SP + Qwen

[[C1 OpenTSLM Qwen - exécution]] : C1 par fenêtre → historique64×10 → encodeur/projecteur OpenTSLM-SP → Qwen3.5-4B avec LoRA rang8, choix de description contrôlée. Compteurs/horodatages déterministes, proposition Nevil après31s, travailleur modèle asynchrone et tâches SQLite. Aucun réseau récurrent LSTM de classification ajouté ici. Livré `2318dee`, service `77c174f` sur8020,39tests/recetteHTTP43s réussis ; brut16/25scénarios artificiels, garde-fou contre désaccord factuel. Aucun bénéfice TSLM ou terrain démontré. L'ancien8019 reste une livraison distincte ci-dessous.

## Livraison actuelle — C1 temporel isolé, 13 septembre

[[C1 temporel - exécution]] : WAV1s/8kHz → neuf descripteurs C1 + score continu → suivi causal SQLite (événement et santé séparés) → aperçu déterministe, **aucun envoi WhatsApp**. API `/temporal` sur première H100/loopback8019, via tunnel SSH ; Monitor à raccorder séparément. Code/service `5553e14`, preuves `8baecab`, `docs/TEMPORAL_ENDPOINT.md`. Repli195 inchangé.

LSTM10→32→1 entraîné sur Aghashahi autorisé, mais gain réservé insuffisant : endpoint expérimental30secondes uniquement, aucun branchement sur le moteur d'alertes. Pas de LLM de notification ajouté faute de besoin au-delà du gabarit. Les architectures TSLM ci-dessous restent des branches expérimentales distinctes ; cette livraison ne démontre ni fiabilité terrain ni valeur d'un TSLM.

Statut : chaîne ML V0 implémentée et testée avec TimeNet/OpenTSLM-SP/Qwen 3.5-4B ; application/baseline finale restent des chantiers séparés. [[V0 ML - exécution]] décrit ce qui est réellement vérifié ; le reste de cette note reste l'architecture cible.

## Nouvelle cible proposée — surveillance continue

Le scénario utilisateur est désormais appareil → flux horodaté → fenêtres causales → score TSLM → suivi d'événement → alerte et examen humain. [[Plan surveillance continue]] détaille l'extension ; aucune de ces nouvelles couches n'est implémentée par le cadrage.

Réutiliser TimeNet pour préparation/traçabilité et le preprocessing V0 pour les fenêtres. Séparer score régulier et description sur événement ; mesurer la capacité à suivre le flux, sans supposer que la génération V0 tient le temps réel. Distinguer santé du flux, prédiction par fenêtre et état d'alerte ; les données manquantes ne valent pas « pas de fuite ». Première cible proposée : un canal sur serveur existant, source rejouée déclarée ou vrai flux compatible. L'embarqué reste à dimensionner. L'ancienne API d'import ci-dessous reste un outil de diagnostic.

## Chaîne de données

WAV source + provenance → manifeste audité et split groupé → TimeNet/TimeF → transformation déterministe partagée → deux branches ML.

Branche TSLM V0 réelle : quatre bandes d'énergie `[4,64]` → encodeur temporel/projecteur OpenTSLM-SP adaptés → décodeur Qwen 3.5-4B gelé → classe et une description. Mesures DSP exposées séparément du texte généré. Bibliothèque `pipe.tslm`, détails/versions et limites dans [[V0 ML - exécution]].

Branche baseline : statistiques agrégées de ces mêmes séries → Random Forest → classe et scores.

Évaluation : prédictions des deux branches + labels privés au modèle → métriques, erreurs, robustesse et exports.

Application : navigateur → API FastAPI → service d'inférence choisi → résultat structuré → affichage. Le navigateur reçoit waveform/spectrogramme pour présentation ; le TSLM reçoit les séries numériques, pas une capture d'écran. Ne pas faire du TSLM un simple rédacteur de la décision baseline.

## Stack de départ

Python pour data, DSP, modèle et API ; NumPy/SciPy pour signal ; scikit-learn/joblib pour baseline ; PyTorch/OpenTSLM pour TSLM ; TimeNet pour conversion ; FastAPI/Pydantic pour contrats ; React/TypeScript + Vite pour frontend ; pytest pour tests. Ce sont des choix de plan raisonnables, les versions compatibles et les lockfiles restent à établir. Éviter deux environnements GPU concurrents inutiles.

Icham possède pyproject.toml/lockfile de base ; les autres lui demandent leurs dépendances. Safoan possède frontend/package.json et son lockfile. Nevil épingle la révision TimeNet réellement testée, Icham celle d'OpenTSLM et du modèle. Si séparation CPU/GPU nécessaire, l'expliciter dans README.

## Arborescence cible du dépôt code

- `src/pipe/contracts.py` : schémas partagés ; Safoan responsable, approbation Icham/Nevil.
- `src/pipe/data/` et `src/pipe/signal/` : ingestion, connecteur, transformations ; Nevil.
- `src/pipe/tslm/` : chargement, adaptation, entraînement et prédiction ; Icham.
- `src/pipe/baseline/` : modèle scikit-learn et prédiction ; Vincent.
- `src/pipe/eval/` : scoring commun, analyses et robustesse ; Nevil.
- `src/pipe/api/` et `frontend/` : API et UI ; Safoan.
- `configs/data/`, `configs/signal/`, `configs/eval/` : Nevil.
- `configs/tslm/` : Icham ; `configs/baseline/` : Vincent.
- `tests/data/`, `tests/signal/`, `tests/eval/` : Nevil ; `tests/tslm/` : Icham ; `tests/baseline/` : Vincent ; `tests/api/` et tests frontend : Safoan.
- `data/`, `artifacts/`, `runs/` : runtime, ignorés par Git par défaut. Pas de gros WAV/checkpoints committés par inadvertance.
- `docs/` : fiches techniques de livraison ; le suivi courant reste dans le vault partagé.

Le nom de package `pipe` est une proposition interne ; vérifier conflit de dépendance et imports lors du bootstrap. Les modules peuvent être renommés en accord commun avant G1.

## Service

Charger les modèles une fois au démarrage, jamais à chaque requête. Démarrer CPU baseline indépendamment du GPU TSLM. Endpoints indiquent explicitement disponibilité et version ; TSLM indisponible n'autorise pas une réponse baseline étiquetée TSLM.

Limiter taille/durée des uploads, temps de calcul et nombre d'inférences simultanées. Lier l'API à localhost ou réseau privé au départ ; pas de publication ouverte d'un GPU sans protection. Aucun chemin fourni par le client ne doit servir directement à lire le disque. Dossiers temporaires nettoyés, secrets hors logs.

## Risques structurants

Petites données et captures corrélées ; conversion acoustique/OpenTSLM non testée ; accès aux bases HF éventuellement soumis à autorisation ; dépendances pré-release ; scores de confiance non calibrés ; bruit externe pouvant révéler le domaine plutôt que la classe.

Démo de secours : replay d'un vrai résultat archivé avec hash du signal/modèle et badge REPLAY. Il ne constitue pas une inférence live. Si aucun TSLM n'a été entraîné, le livrable ne remplit pas ce volet du challenge.

## Source locale de replay — réalisée séparément

2026-09-12 : `scripts/replay_sensor.py` sur `feat/sensor-replay` fournit WAV validés → échéances monotones et incidents → une fenêtre en traitement et une en attente → journal local. Tests CPU et exécution réelle acquis, aucun modèle/API raccordé. [[Plan simulateur de capteur]] décrit la preuve et le prochain raccordement à convenir ; l’enveloppe reste locale, sans modification de `Prediction`.
