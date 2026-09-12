# Contrats techniques

Icham

Version de plan historique : v0.1. Une implémentation `Prediction` Safoan et la fonction V0 Icham existent désormais ; preuves dans [[V0 ML - exécution]]. Le reste doit être confronté au code et figé à G1. Les changements nécessitent une migration et l'accord des consommateurs.

Extension **proposée, non implémentée** pour [[Plan surveillance continue]] : métadonnées de flux/fenêtre (appareil, session, séquence, horaires source/réception, qualité et fraîcheur), état de surveillance et événement d'alerte séparés de `Prediction`. Événement : identifiant stable, ouverture/dernière preuve/fin éventuelle, état, versions et extraits justificatifs. Safoan/Icham/Nevil doivent convenir du schéma ; ne pas convertir silencieusement un `sample_id` V0 en identité de session ni une absence de données en classe négative. Conserver le CSV d'évaluation Nevil distinct de ce contrat applicatif.

## 1. Source et manifeste, propriétaire Nevil

Source : https://zenodo.org/records/18631450 ; licence déclarée CC BY 4.0. Les archives fuite/non-fuite sont des mesures d'un site expérimental. Les bruits externes constituent un domaine distinct. Lire [[PIPE - proposition ML et démo]] pour les limites.

Une ligne de manifeste par WAV :
- sample_id : identifiant opaque stable, indépendant du label et du nom révélateur.
- source_record_id, source_url, archive_sha256, audio_sha256, license_id.
- relative_audio_path : chemin relatif contrôlé ; nom source conservé dans le manifeste d'audit, jamais prompt.
- sample_rate_hz, n_samples, channels, duration_seconds, dtype.
- source_class : leak / no_leak / environmental_noise.
- target_label : 1 fuite, 0 non-fuite pour la tâche principale ; conserver source_class séparément.
- event_group_id, group_method, group_confidence ; ne pas fabriquer un event_id inconnu.
- split : train / validation / test / quarantine.
- pipe_material, region, acquisition_device, pressure_value/unit, velocity_value/unit si présents ; valeurs manquantes null, parsing avec provenance. Ces champs sont destinés à l'audit/splits, exclus des entrées du MVP.
- exclusion_reason si quarantine.

Ne pas convertir une vitesse m/s en débit L/h sans section connue. Ne pas considérer le WAV comme pression en pascals sans calibration. Les deux valeurs binaires sont un choix de tâche, pas la preuve que tout bruit extérieur équivaut à une canalisation saine.

## 2. Prétraitement partagé, Nevil + Icham

API Python cible : `preprocess_audio(waveform, sample_rate, config) -> SignalExample`.

SignalExample : sample_id opaque, time_series float32 de forme [C,T], channel_names, time_offsets_seconds de longueur T, preprocessing_version, input_sha256 et propriétés mesurées. La dimension C représente les bandes fréquentielles, T les fenêtres temporelles. L'adaptateur convertit explicitement vers le format attendu par la version OpenTSLM installée ; ne pas supposer que son padding/batching correspond à ce contrat interne.

Choisir et figer taux d'échantillonnage cible, traitement mono, taille FFT, hop, nombre de bandes, plage fréquentielle, log-énergie/epsilon, padding et normalisation après inspection des fichiers. Pas de constantes arbitraires cachées dans le frontend. Anti-aliasing si resampling. Préserver le signal original pour écoute.

Toute normalisation apprise est fit sur train uniquement et sérialisée. Pas de normalisation calculée sur un batch mélangeant train et test. Si normalisation individuelle par clip choisie, justifier ses effets sur l'amplitude informative.

La baseline reçoit des agrégats déterministes de SignalExample : moyenne/écart-type/quantiles par bande et autres features convenues. `feature_names` et `feature_version` fixent leur ordre. Prohiber label, ID, nom de fichier et métadonnées expérimentales révélatrices des entrées. Ne pas reproduire un DSP divergent dans la branche Vincent.

## 3. TimeNet et TSLM

Nevil doit implémenter le connecteur avec les APIs de la révision TimeNet vérifiée, produire un artefact TimeF et démontrer un round-trip. Lire le code amont pour les signatures exactes. Le présent manifeste est interne, pas une prétendue copie du schéma TimeF.

Cibles TimeNet : classification et réponse textuelle. Pas de cible de localisation temporelle de la fuite sans annotations d'apparition dans le clip.

Adaptateur OpenTSLM cible à confirmer dans l'amont : pre_prompt, time_series_text, time_series, post_prompt, answer. answer n'est présent que dans la voie d'apprentissage/scoring, jamais injecté au prompt d'inférence. Une description de bandes ne doit pas contenir les observations cibles ou la classe cachée.

Réponse d'entraînement : une classe parmi leak/no_leak, puis quelques propriétés déterministes du signal. Les réponses générées par gabarit ne sont pas des rapports d'experts. Ne pas générer librement des causes de fuite ni de recommandations de réparation. En cas de sortie invalide, enregistrer l'erreur et l'inclure dans le taux d'invalidité, pas la corriger silencieusement avec le vrai label.

## 4. Prédiction partagée, Icham/Vincent → Safoan/Nevil

Fonction cible : `predict(example) -> Prediction`.

Champs : schema_version, sample_id, input_sha256, model_name (tslm/baseline), model_version, preprocessing_version, prediction (leak/no_leak/null), class_scores (objet optionnel), score_type (raw/calibrated/none), abstained (bool), abstention_reason (nullable), observations (liste de propriétés avec valeur/unité/méthode), description (texte court ou null), latency_ms, warnings, execution_mode (live/replay/development_fixture).

Ne jamais inclure ground_truth dans Prediction. La baseline peut ne pas produire de description ; afficher ce fait plutôt que lui inventer un raisonnement. Le TSLM ne doit pas recevoir les scores baseline en entrée.

Les observations calculées par DSP et les assertions générées par le modèle doivent être distinguées dans les exports/UI. Une concordance mesure/texte ne prouve pas une explication causale interne du modèle. Un score non calibré est présenté comme score, pas comme certitude en pourcentage.

## 5. API cible, Safoan

- GET /health : modèles disponibles, versions, device ; aucun secret.
- GET /samples : exemples de démonstration autorisés, IDs opaques, durées ; pas les labels avant révélation.
- POST /predict : sample_id OU fichier validé, model_name, perturbation optionnelle. Interdire les chemins arbitraires du client. Réponse Prediction.
- GET /samples/{id}/audio : audio original autorisé.
- GET /samples/{id}/visualization : waveform/spectrogramme/version et paramètres ; limiter le volume envoyé.
- GET /samples/{id}/label : route de révélation pour le sous-ensemble démo autorisé, isolée du service d'inférence. Jamais accès à tout le test caché via cette route pendant le développement.
- GET /evaluation : métriques d'un run publié et figé, provenance/version ; aucune métrique inventée si absent.

Valider extensions ET décodage audio, taille/durée, fréquence supportée, NaN/Inf, ID inconnu, modèle indisponible. Retourner erreurs structurées, jamais HTTP succès avec une prédiction factice.

## 6. Contrat bruit, Nevil produit / Safoan expose

Perturbation : noise_id d'un pool autorisé, snr_db, seed, version. Mélange calculé sur le serveur par fonction déterministe commune ; même audio perturbé utilisé pour lecture, visualisation et prédiction. Documenter définition de SNR et comportement pour signal/bruit silencieux, clipping, resampling et normalisation. Afficher perturbation synthétique d'un clip réel.

Bruits de test séparés des bruits d'augmentation. Aucun clip dérivé ne traverse les splits. Fixer la grille de robustesse sur validation, pas après inspection du test. Le slider peut produire des scores non monotones ; l'UI ne doit pas les lisser pour raconter une meilleure histoire.

## 7. Artefacts et interfaces de commandes

Commandes cibles à créer, à remplacer par les commandes effectivement testées dans chaque journal :
- `python -m pipe.data.prepare --config configs/data/default.yaml`
- `python -m pipe.baseline.train --config configs/baseline/default.yaml`
- `python -m pipe.tslm.train --config configs/tslm/smoke.yaml`
- `python -m pipe.eval.run --config configs/eval/default.yaml`
- `uvicorn pipe.api.main:app --host 127.0.0.1 --port 8000`
- `npm run dev` depuis frontend.

Éviter la confusion : ces interfaces proposées ne sont pas les commandes natives d'OpenTSLM. Chaque propriétaire doit les implémenter ou documenter leur remplacement. Aucune commande de ce paragraphe n'est annoncée comme opérationnelle.

Artefacts attendus : manifest, split hash, preprocessing config/stats, feature_names, weights/adapters et paramètres nécessaires au reload, training config, environment lock, predictions.jsonl, metrics.json, erreurs, licence et provenance. Les fichiers lourds sont stockés hors Git et livrés par URL contrôlée, avec checksum et procédure de téléchargement. Ne pas publier des poids de base dont la licence ne le permet pas.
