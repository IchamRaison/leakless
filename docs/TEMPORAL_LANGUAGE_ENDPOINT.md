# C1 + OpenTSLM-SP + Qwen — module IA de démonstration

Ce module remplace **le rôle demandé au modèle**, pas les poids C1 : l'ancien petit LSTM de classification acoustique n'est pas utilisé. Le détecteur C1 existant produit neuf mesures et un score par seconde. OpenTSLM encode leur historique ; Qwen3.5-4B gelé choisit une description temporelle parmi cinq formulations. La fin de l'entraînement et les résultats réels sont consignés dans `docs/evidence/temporal-language-001/` et le vault.

## Responsabilités et contrat

Le module IA prépare une notification destinée logiquement à **Nevil**, après **plus de30secondes consécutives avec score C1≥0,8**. À1Hz, cela signifie la31e fenêtre, jamais la30e. C1/persistance ne dépend pas du résultat du modèle de langage. Les dates/durées proviennent des horodatages et des fenêtres valides ; elles ne datent pas le début physique de la fuite. Une coupure, une mauvaise qualité ou un redémarrage remet la continuité à zéro sans annoncer une réparation.

Le backend utilisateur réalise l'envoi. **Aucun appel WhatsApp**, numéro, identifiant fournisseur ou secret n'est requis dans ce module ; `sent:false`, `delivery:preview_only`. Le frontend n'est pas modifié. Le service initial8019 et le placeholder195 restent préservés. Nouvelle base obligatoire, nouveau service sur8020.

Même protocole audio que [l'endpoint C1](TEMPORAL_ENDPOINT.md) : monoPCM16/32/8kHz, fenêtres1s, séquence et fin source ISO8601 avec timezone. Créer la session puis envoyer les fenêtres à leur cadence réelle. Aucun score ou label attendu dans la requête audio.

```bash
ssh -N -L 8020:127.0.0.1:8020 hicham@89.169.123.193
curl http://127.0.0.1:8020/temporal/health
```

- `POST /temporal/sessions` avec `{"source_mode":"replay"}`.
- `POST /temporal/sessions/{id}/windows`, multipart `file`, `sequence`, `source_end_at`.
- `GET /temporal/sessions/{id}` pour état et `previews` persistants. L'aperçu passe de `pending` à `ready` après la description du modèle ; le flux C1 continue sans l'attendre.
- `POST /temporal/sessions/{id}/describe` pour décrire l'historique courant à la demande : au moins31fenêtres récentes valides, sinon422 ; modèle occupé409. Ne crée aucun envoi. Réponse attachée à `through_sequence` : ne pas confondre un historique décrit avec des fenêtres arrivées pendant le calcul.
- `POST /temporal/sessions/{id}/end` arrête l'observation, sans affirmer que la fuite est réparée.
- L'ancienne route `/temporal/sequence` renvoie410 dans ce service : le LSTM acoustique précédent n'est pas ce système.

Un aperçu `persistent` est créé une fois par événement. Après5fenêtres≤0,4, un aperçu `ended` est préparé seulement si cet événement avait déjà franchi30s. Dédupliquer côté envoi par `(event_id,kind)`, attendre `status:ready`, ignorer les réponses `duplicate:true`. Les tâches et leurs historiques sont stockés dans SQLite ; redémarrer reprend les descriptions en attente, pas les envois externes. Le module ne peut pas garantir la déduplication WhatsApp du backend.

## Ce que Qwen fait réellement

Entrée : jusqu'à64×10 valeurs numériques via **l'encodeur et le projecteur OpenTSLM**, noms des canaux et nombre de fenêtres valides. Les cinq descriptions couvrent calme, bref/récent, intermittent, persistant, retour au calme. Le modèle choisit un token parmiA–E correspondant aux formulations : **vocabulaire fermé, pas rédaction libre**. `candidate_scores` sont relatifs à ces cinq choix, non des probabilités de fuite calibrées.

Sortie `temporal_description` : formulation choisie par le modèle, faits mesurés et provenance. Une incohérence avec les propriétés mesurées déclenche `template_fallback`, avec motif explicite ; le choix brut du modèle reste visible. Avec moins de31fenêtres après une coupure, le modèle n'est pas exécuté et le fallback est déclaré. Une panne du modèle ne change jamais le score C1 ou une durée. Le message final combine le gabarit de faits et la description contrôlée.

Exemple de **format**, pas résultat mesuré : «REPLAY — Nevil, un signal compatible avec une fuite est observé à niveau élevé depuis31secondes consécutives. Une vérification est recommandée. [Description temporelle validée].»

## Apprentissage et portée

Une recette400pas, Qwen gelé, encodeur/projecteur entraînés ; empreintes de base, de C1, de normalisation et du checkpoint. Quarante chronologies artificielles train et25réservées, composées de mesures réelles des598clips train C1. Aucun Aghashahi ni test officiel utilisé. Les étiquettes de dynamique sont dérivées de règles, **pas d'apparitions physiques annotées**. Les scénarios réservés ne représentent pas des acquisitions indépendantes.

Comparer les sorties brutes du modèle aux règles (parfaites par construction sur cette tâche), compter les fallbacks et tester l'inversion temporelle. Un retour sûr au gabarit n'est pas un succès du modèle. Cette démonstration réalise le branchement séries/langage ; elle ne prouve pas une supériorité du TSLM, une réduction des dégâts ou une fiabilité terrain.

## Lancement isolé

Réutiliser le runtime ML et l'overlay `requirements-temporal-api.txt` ; `serve.py` ne propose que le loopback, un processus et une concurrence bornée. Fournir les empreintes publiées, pas un chemin de checkpoint arbitraire transmis par un client.

```bash
PYTHONPATH=/home/hicham/pipe-v0/.venv-temporal-api/lib/python3.12/site-packages:src \
  /home/hicham/pipe-v0/.venv-repro/bin/python scripts/temporal/serve.py \
  --bundle /home/hicham/pipe-v0/artifacts/c1-temporal-002/bundle \
  --sha256 cea924a6a0f8c3d3ff91930d8125f309a3bcded4c965a4ef53c2c566fc32ebdd \
  --language /home/hicham/pipe-v0/artifacts/temporal-language-001 \
  --language-sha256 EMPREINTE_PUBLIEE_DANS_LES_PREUVES \
  --db /home/hicham/pipe-v0/artifacts/temporal-language-001/service.sqlite --port 8020
```

Bearer optionnel `PIPE_TEMPORAL_TOKEN` pour les routes temporelles ; hors tunnel SSH, HTTPS et authentification sur **toutes** les routes via proxy sont requis. Aucun endpoint publiquement exposé par cette commande. Ne jamais toucher au service du placeholder pour déployer ce module.
