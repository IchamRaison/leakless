# Endpoint C1 — suivi temporel et LSTM expérimental

Branche `feat/c1-temporal-alerts`. Service isolé sur `hicham@89.169.123.193`, sans modification du frontend ou du placeholder `195.242.28.46`. Aucun message WhatsApp envoyé : uniquement des aperçus persistants. Ce pipeline ne remplace pas les résultats TSLM du hackathon.

## Accès et lancement

Accès recommandé depuis le poste de l'intégrateur :

```bash
ssh -N -L 8019:127.0.0.1:8019 hicham@89.169.123.193
curl http://127.0.0.1:8019/temporal/health
```

Commande sur le GPU, depuis le dossier de code livré :

```bash
PYTHONPATH=/home/hicham/pipe-v0/.venv-temporal-api/lib/python3.12/site-packages:src \
  /home/hicham/pipe-v0/.venv-repro/bin/python scripts/temporal/serve.py \
  --bundle /home/hicham/pipe-v0/artifacts/c1-temporal-002/bundle \
  --sha256 cea924a6a0f8c3d3ff91930d8125f309a3bcded4c965a4ef53c2c566fc32ebdd \
  --db /home/hicham/pipe-v0/artifacts/c1-temporal-002/service.sqlite --port 8019
```

Le lanceur écoute uniquement en loopback. `PIPE_TEMPORAL_TOKEN` ajoute un Bearer privé à toutes les routes `/temporal` ; ne jamais le mettre dans Git. Pour un accès hors tunnel, prévoir HTTPS et authentification au reverse proxy **sur toutes les routes**, pas une ouverture directe d'Uvicorn. L'API existante reste présente et ses autres routes n'utilisent pas ce Bearer. Aucun environnement TSLM ne doit être configuré dans ce service isolé.

Environnement API séparé épinglé dans `requirements-temporal-api.txt`, chargé en overlay du runtime ML existant. Ne pas installer le lock API général à sa place : il utilise une autre version de NumPy, incompatible avec l'identité enregistrée du checkpoint C1. Aucun paquet du runtime ML initial n'a été remplacé.

Un processus, huit requêtes concurrentes maximum, backlog huit ; surplus refusé, aucune queue audio illimitée. La séquence GPU utilise un verrou et retourne409 si occupée. SQLite garde32sessions/1000événements au maximum ; à saturation, archiver la base et créer une nouvelle campagne, sans effacement automatique. Permissions privées au lancement. Un changement de modèle/politique exige une nouvelle base. Redémarrer sur la même base conserve événements et déduplication, mais signale l'interruption d'observation.

## Contrat de raccordement

1. `POST /temporal/sessions` JSON `{"source_mode":"replay"}` →201, `session_id`. Seuls `replay` et `development_fixture` sont acceptés ; pas de capteur terrain prétendument validé.
2. `POST /temporal/sessions/{id}/windows` multipart : `file` WAV mono PCM16/32,8kHz, exactement1seconde ; `sequence` entier croissant, départ0 ; `source_end_at` ISO8601 avec timezone, fin de disponibilité de la fenêtre. Première fenêtre après création+1s, suivantes non chevauchantes à1Hz. Pas de label/pression/fuite attendue dans la requête.
3. `GET /temporal/sessions/{id}` → état, événements et aperçus (50derniers), santé calculée à l'instant de lecture.
4. `POST /temporal/sessions/{id}/end` → arrête l'observation, ne déclare pas la fuite réparée.

Réponse fenêtre : `probability_leak` continu brut/non calibré, `features` neuf descripteurs C1, SHA audio, séquence, `state`, `preview`, `quality_error`, `duplicate`, `latency_ms`. Les versions du modèle/politique sont dans `state`. Les instants d'événement sont ISO8601 ; `last_end`/`last_received` sont des secondes Unix. Le score vient du checkpoint, jamais du nom du fichier ou d'un scénario.

Silence/constante, format invalide ou pleine échelle → abstention (`probability_leak:null`), santé dégradée, pas de faux «sans fuite». Détection de pleine échelle conservative (rails PCM16 inclus, même dans un conteneur32bits). Payload trop grand413 ; session inconnue404 ; conflit d'identité, ordre/horodatage invalide422 ; backend absent503 ; Bearer manquant/invalide401. Le plafond WAV fenêtre est64KiB, celui des séquences1MiB ; le middleware général refuse aussi les gros corps avant parsing multipart.

Le dernier numéro retransmis avec mêmes octets/date est idempotent ; les numéros plus anciens sont refusés. Un client doit ignorer l'aperçu d'une réponse `duplicate:true` ; dédupliquer toute restitution par `(event_id,kind)`. Les aperçus en base sont déjà uniques sur cette clé. Rien n'est envoyé à un fournisseur.

Les horloges source/serveur doivent être synchronisées : futur toléré250ms, ancienneté maximale2s ; séquence et date doivent exprimer les pertes. Ne pas envoyer les fenêtres historiques en rafale. Trou, silence ou redémarrage réinitialise les compteurs de continuité sans fermer une alerte active. Santé et événement sont deux états distincts.

## Politique de démonstration

C1 gelé, neuf descripteurs d'amplitude/enveloppe et logistiqueC=0,01. Ouverture après3fenêtres consécutives≥0,8 ; fermeture observée après5≤0,4. Seuils non calibrés pour le terrain. Dates de première observation et de confirmation distinctes ; durée = fenêtres valides effectivement surveillées pendant l'événement, y compris l'hystérésis de fermeture, jamais durée physique certaine de fuite. Les trous n'ajoutent aucune durée. Couverture interrompue explicitement signalée. Le gabarit rapporte des faits, sans localisation/débit ni bande fréquentielle inventée.

## LSTM séparé : classification de30secondes

`POST /temporal/sequence`, multipart `file`, mono PCM16/32/8kHz/exactement30s. Réponse score, versions, SHA, latence et `experimental:true`. Cible terminale seulement : pas de score par seconde validé, aucune création d'événement par cette route. Même preprocessing/checkpoint/FP32 que l'évaluation. LSTM sur CUDA ; C1/DSP légers sur CPU de la même machine. Le contrôleur commun restaure TF32 désactivé au chargement.

Expérience Aghashahi inclusive : des signaux saturés sont conservés inchangés, défauts comptés dans la préinscription/rapports. Le service s'abstient sur ces mêmes défauts : **les métriques expérimentales ne représentent donc pas sa couverture**. Aucune borne d'apparition/disparition de fuite disponible ; aucune précision événementielle, délai physique ou fausse alerte/jour annoncés.

## Replay et vérification

`scripts/replay_sensor.py` reprend le simulateur de `feat/sensor-replay` (cadence, pertes, doublons, attente bornée), avec un récepteur HTTP optionnel. Le scénario contient chemins/empreintes/provenance d'audio train ou démo autorisée, pas de scores. `examples/replay/scenario.json` est une fixture synthétique de tests, pas un jeu scientifique prêt à écouter.

```bash
python scripts/temporal/replay_endpoint.py --scenario scenario.json \
  --output nouveau_replay --endpoint http://127.0.0.1:8019
PYTHONPATH=src python -m pytest tests/temporal tests/api tests/replay -q
```

Chronologie de concaténation explicitement artificielle ; un replay réussi ne valide pas la détection terrain. Ne pas utiliser les réserves pour construire/régler une démonstration.

En cas de repli, arrêter uniquement le PID vérifié de ce service isolé et retirer son tunnel/raccordement. Ne jamais arrêter/modifier le placeholder ; son activation reste une décision distincte du propriétaire.

## Résultat livré et preuves

`docs/evidence/c1-temporal-002/` conserve préinscription, journal du fit, empreintes, prédictions/rapports validation et confirmation. Unique LSTM :100époques/500pas sur H100,39/40 corrects train. Il **n'est pas promu** : AUC groupe0,375 en validation et0,590278 en confirmation, contre0,8125 puis1,0 pour le contrôle sans ordre retenu. Pas de changement de seuil/recette après ces résultats. La réserve confirmation a désormais été observée et ne peut plus servir à choisir une nouvelle recette.

L'AUC parfaite de certains contrôles après agrégation de deux hydrophones ne prouve pas une décision fiable : C1 médiane au seuil0,5 classe tous les enregistrements confirmation «fuite». Confirmation sur le même banc, seulement six groupes négatifs ; validation seulement deux. C1 reste une démonstration de chaîne, pas une solution terrain qualifiée.

Recette `http-smoke-001` :12fenêtres à1Hz, une ouverture/fermeture et deux aperçus ; parité C1 HTTP/offline exacte (neuf mesures et scores), parité LSTM HTTP/offline exacte sur une séquence train sans pleine échelle. Latence serveur fenêtre p95≈11,06ms/max12,63ms sur ce court essai ; appel séquence≈0,316s, initialisation comprise. Scénario choisi **exprès aux extrêmes des scores train**, répétitions explicites, chronologie artificielle : ce n'est pas une mesure de qualité du modèle ni un benchmark de charge. Aucun envoi WhatsApp.

Audit indépendant, sans modèle : `python3 scripts/temporal/verify_reports.py docs/evidence/c1-temporal-002` vérifie mapping, séparation des groupes, scores finis, AUC par comparaisons par paires et référence choisie. `service-launch.json` identifie le processus lancé et sa révision ; vérifier le PID courant avant toute opération.
