# Journal Safoan

## État actuel — 2026-09-13

Intégration applicative de la restitution TSLM V2 terminée sur
`feat/demo-temporal-building` au merge commit `bba67ff`. La branche
`feat/icham-v2-reliability` a été fusionnée, puis son `CoherentPredictor` a été
raccordé à `POST /predict` et au composant partagé `ModelReadout`.

Le bouton lance l'inférence uniquement à la demande. L'API charge un seul backend
au démarrage et exige le bundle sélectionné, `decision.json` et
`validation-evidence.json`. Ces artefacts ne sont pas présents dans Git : sans
configuration, l'application démarre et affiche une indisponibilité explicite,
sans score de remplacement. La réponse exposée lie `request_id`, `sample_id` et
le SHA du signal décodé ; elle conserve seuil/provenance/fallback mais ne publie
pas l'audit brut du modèle. Le score est présenté comme brut non calibré, sans
probabilité terrain ni recommandation d'inspection.

### Preuves du jalon V2

- `.venv/bin/python -m pytest -q` : 23 tests API réussis, deux avertissements de
  dépréciation provenant des bibliothèques de test.
- `.venv/bin/python -m pytest -q tests/tslm/test_coherent.py` : 12 tests du
  wrapper cohérent réussis sur faux backend.
- `npm --prefix frontend test` : 35 tests réussis dans 15 fichiers, dont appel
  explicite, provenance, fallback et rejet d'une réponse d'identité incorrecte.
- `npm --prefix frontend run build` : build réussi ; avertissement de taille du
  chunk Three.js, sans échec.
- Recette navigateur locale : sélection N3 puis enregistrement no-leak, carte et
  signal chargés, bouton TSLM V2 activé ; le clic renvoie les trois variables
  d'artefacts manquantes et propose un retry, sans résultat ni score fictif.
- Documentation et configuration : `docs/APPLICATION.md`,
  `docs/DEMO_TEMPORAL_BUILDING.md`, `.env.example`.

### Limite et prochaine action

Obtenir d'Icham les trois artefacts cohérents produits par la validation V2,
installer le runtime ML Python 3.12, puis exécuter une inférence réelle de bout
en bout et une recette navigateur. Ce jalon ne valide ni la qualité du modèle ni
G3 sur poids réels.

## Jalon précédent — 2026-09-12

Premier jalon S0–S1 livré sur `feat/safoan-app` au commit `504ad14` : studio React/TypeScript/Vite et API FastAPI CPU, import WAV, lecteur, waveform/spectrogramme et erreurs. Recette responsive terminée ; aucune inférence réelle ni G3 validé.

### Preuves

- `.venv/bin/python -m pytest -q` : 19 tests réussis, deux avertissements de dépréciation provenant des bibliothèques de tests.
- `npm --prefix frontend run build` : build TypeScript/Vite réussi.
- `npm --prefix frontend test` : 4 tests React réussis, dont rejet fichier trop gros et réponse de visualisation obsolète.
- Navigateur local : import d'un WAV synthétique de test, durée 2 s / 16 kHz, waveform et spectrogramme affichés ; lecture active vérifiée sans erreur média, boucle et comparaison indisponible contrôlées. Ce signal n'est pas une donnée de démo scientifique.
- Services : http://127.0.0.1:5173 et http://127.0.0.1:8000/docs. L'ouverture du port API a nécessité l'exécution autorisée hors sandbox ; aucune exposition publique.
- Code : `src/pipe/api/`, `src/pipe/contracts.py`, `frontend/`, `tests/api/`. Reproduction et contrats dans `docs/APPLICATION.md`.
- Python 3.13.11, Node 25.6.1 ; dépendances CPU isolées dans `requirements-api.txt`, aucun fichier ML d'Icham modifié.

### Limites et décisions

- Aucun WAV audité de Nevil, modèle baseline/TSLM ou run d'évaluation livré ici. Les routes correspondantes renvoient 404/503 explicites ; boutons d'analyse/révélation désactivés.
- Visualisation `display-stft-v1` distincte du DSP ML : ne pas utiliser ces pixels comme entrée modèle. Son/hash/visuels correspondent au même signal décodé.
- Proposition d'extension API : `POST /samples` (multipart) puis ID partagé pour écoute/visuel/prédiction ; `DELETE /samples/{id}`. Contrat Prediction v0.1 à valider collectivement, pas gelé G1.
- Tous les imports locaux sont marqués `development_fixture`, provenance non vérifiée. 8 Mio / 30 s / 16 clips max ; mémoire perdue au redémarrage.
- Bruit, vraie inférence, replay et recette S2–S4 restent à réaliser après livraisons des autres propriétaires. Aucun entraînement, baseline ou pipeline data/DSP implémenté par Safoan.
- Entire installé lors de la reprise ; approbation `/hooks` et capture effective restent non vérifiées.

### Prochaine action historique

Obtenir le WAV démo et SignalExample de Nevil, valider les contrats avec l'équipe, puis intégrer les adaptateurs/poids d'Icham et Vincent pour G3.

## Historique

- 2026-09-12 : branche publiée, installation Entire et diagnostic local ; aucune API, UI ou inférence testée. Source d’installation : https://docs.entire.io/quickstart.

- 2026-09-12 : premier studio lançable et 23 tests passants ; recette navigateur sur fixture synthétique uniquement, détails ci-dessus.
