# Agent Safoan - Application

Icham

## Mission

Tu travailles pour Safoan sur PIPE. Tu rends la vraie analyse lisible et manipulable. Le visuel doit provenir du même signal que l'inférence et chaque chiffre doit correspondre à un run. Pas de carte inventée ni score scénarisé.

## Lecture et propriété

Lire [[Plan directeur agents]], [[Architecture]], [[Contrats techniques]], [[Protocole évaluation]], [[Démo et pitch]], [[Coordination et passation agents]], [[Passation]]. Branche feat/safoan-app. Propriétaire frontend, API et schémas partagés validés avec les autres. Ne pas modifier leur modèle/DSP dans ta branche.

## S0 — Contrat et squelette lançable

1. Inspecter le dépôt puis créer frontend React/TypeScript/Vite et API FastAPI selon disponibilité réelle. Définir scripts de lancement, dépendances et ports, documenter si un port est occupé.
2. Traduire [[Contrats techniques]] en schémas validés backend/frontend. Partager avec les trois autres avant de construire le reste.
3. Établir les états live/replay/development_fixture ; fixtures visuellement marquées en permanence. Aucun compteur de performance ou prédiction fictive ne doit apparaître comme résultat réel.
4. Lire un WAV réel de développement fourni par Nevil ; afficher durée et origine. Préparer une UI qui fonctionne sans modèle disponible et dit explicitement « modèle indisponible ».

Premier jalon : lecteur réel, affichage waveform et spectrogramme, endpoint health, lancement documenté.

## S1 — Studio signal

1. Vue principale compacte : sélection du clip, lecture/arrêt, waveform, spectrogramme, panneau modèle, bouton révéler le label et accès comparaison.
2. Synchroniser curseur de lecture et time-axis. Afficher axes, unités connues et source. Indiquer clip d'une seconde et boucle si répétition activée, sans simuler une chronologie continue.
3. Faire produire les valeurs spectrales par le preprocessing partagé ou une visualisation explicitement distincte/documentée ; ne pas appeler les pixels « attention du modèle ».
4. Masquer labels et noms révélateurs avant la prédiction, y compris tooltip, URL visible, console et textes alternatifs. Aucun besoin de protéger cryptographiquement le jeu démo public, mais ne pas créer une fuite dans l'entrée modèle.
5. Prévoir clavier, contraste et textes compréhensibles, sans surcharge de gadgets. L'esthétique ne remplace pas le rendu des erreurs.

## S2 — Inférence réelle

1. Intégrer predict_baseline de Vincent et predict_tslm d'Icham sans changer leur information d'entrée. Charger une seule fois les poids.
2. Exposer endpoints cibles ; modèle choisi réellement exécuté, versions présentes. Si TSLM échoue, afficher erreur, pas une réponse de la baseline sous son nom.
3. Limiter taille/durée et décoder les fichiers sur backend ; ID contrôlés, aucun chemin client direct. Échapper texte généré ; pas de HTML brut du modèle.
4. Annuler/ignorer les réponses obsolètes : un changement de clip ou bruit ne doit pas afficher le résultat de l'ancienne entrée. Associer input_hash et request_id aux vues.
5. Mesurer latence réelle et afficher score selon score_type. Si non calibré, pas de pourcentage de certitude. Les observations DSP et texte modèle ont des libellés distincts.
6. Révéler label via route de démo autorisée séparée du prompt et enregistrer l'exemple utilisé. Aucune lecture des labels test scellés pour arranger le frontend.

## S3 — Robustesse interactive, après G3

1. Consommer mix_noise de Nevil ; noise_id, seed et SNR bornés. Éviter une requête GPU par pixel déplacé : bouton analyser ou debounce avec limite de concurrence.
2. Réécouter et afficher le vrai mélange renvoyé/identifié par le serveur ; toutes les vues partagent input_hash.
3. Afficher explicitement mélange synthétique à partir d'un enregistrement réel. Ne pas faire croire à du capteur live.
4. Ne pas forcer une courbe décroissante ni une abstention. Montrer résultats même erratiques ; pas de cache qui répond avec un autre modèle/version.
5. Page comparaison charge metrics.json réel de Nevil : support, métriques, split, hash du run et limites. Si absent, indiquer non évalué.

## S4 — Recette et livraison

Tester : WAV valide, WAV corrompu, fichier trop gros, silence, ID inconnu, modèle indisponible, timeout, double clic, changement de clip pendant requête, déconnexion réseau et redémarrage backend. Vérifier commandes de build/test et parcours navigateur de bout en bout avec vraie inférence.

Créer mode replay de secours uniquement à partir de résultats réels enregistrés, avec version et badge REPLAY. Préparer une capture vidéo après fonctionnement effectif. Le replay n'est jamais un substitut silencieux à l'inférence live.

Livrer README app/API, variables d'environnement sans secrets, commandes testées, schémas, captures éventuelles et preuve de smoke test. Pas d'API GPU publique sans autorisation/protection.

## Tests d'acceptation

- [ ] Clip réel lu, waveform et spectrogramme synchronisés.
- [ ] Même input_hash pour écoute, affichage et inférence.
- [ ] Baseline/TSLM séparés et versionnés.
- [ ] Vrai résultat, label séparé, aucune fuite au prompt.
- [ ] Uploads/erreurs/timeout et requêtes obsolètes gérés.
- [ ] Bruit contrôlé testé ou marqué non implémenté.
- [ ] Métriques tirées d'un run réel et score correctement nommé.
- [ ] Build et parcours complet effectivement exécutés.

## Passation

[[Journal Safoan]] : commit, commandes, endpoints opérationnels vs proposés, exemple/input hash de smoke test, erreurs connues, screenshots si utiles et demandes aux autres. Si le backend modèle n'est pas prêt, noter UI prête/API non intégrée, pas « démo terminée ».
