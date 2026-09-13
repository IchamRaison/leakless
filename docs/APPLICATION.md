# Studio PIPE — livraison Safoan

## État

Premier jalon S0–S1 : API CPU et studio lançables, import WAV local, lecteur natif,
waveform et spectrogramme. Contrats v0.1 proposés dans `src/pipe/contracts.py` et
validés côté frontend avec Zod. Le gel collectif G1 reste à faire.

Aucun adaptateur baseline/TSLM, poids, prétraitement partagé, WAV de développement
audité ou résultat d'évaluation n'est livré dans cette branche. Les boutons
d'inférence et de révélation sont donc désactivés. S2–S4, le bruit et le replay ne
sont **pas** annoncés comme réalisés. Aucun entraînement/data pipeline modifié.

## Installation et lancement

Depuis la racine, avec Python 3.13 et Node 22.12+ (testés : Python 3.13.11,
Node 25.6.1, npm 11.9.0) :

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-api.txt
npm --prefix frontend ci
```

Terminal API :

```sh
.venv/bin/python -m uvicorn pipe.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Terminal frontend :

```sh
npm --prefix frontend run dev
```

Studio : http://127.0.0.1:5173. OpenAPI interactive : http://127.0.0.1:8000/docs.
Les deux services restent sur localhost. Vite refuse un port occupé (strictPort).
Le proxy `/api` cible le port 8000. Si un port doit changer, adapter explicitement
la commande et `frontend/vite.config.ts` ; ne pas arrêter un service inconnu.
Aucune variable secrète ni GPU n'est nécessaire. `vite preview` sert seulement
le build statique et n'est pas le lancement du studio avec API.

`requirements-api.txt` verrouille uniquement l'environnement CPU Safoan et ses
tests. Le futur `pyproject.toml` et lock ML restent sous responsabilité d'Icham.
Pour actualiser ce lock : `uv pip compile requirements-api.in --python python3.13
--output-file requirements-api.txt`.

## API opérationnelle

| Méthode et route | Comportement |
| --- | --- |
| `GET /health` | Version, CPU, modèles indisponibles et capacités |
| `GET /samples` | Imports de la session serveur, IDs opaques, sans labels ni noms source |
| `POST /samples` | Multipart `file` WAV → métadonnées `Sample`, HTTP 201 |
| `DELETE /samples/{id}` | Retrait de l'import en mémoire, HTTP 204 |
| `GET /samples/{id}/audio` | Octets WAV d'origine et en-tête `X-Input-SHA256` |
| `GET /samples/{id}/visualization` | Waveform min/max, spectrogramme et paramètres d'affichage |
| `POST /predict` | JSON `sample_id`, `model_name`, `request_id` ; HTTP 503 explicite tant que non intégré |
| `GET /samples/{id}/label` | HTTP 404 : aucun label autorisé pour les imports |
| `GET /evaluation` | HTTP 404 : aucune évaluation publiée intégrée |

Les erreurs sont `{ "error": { "code": "…", "message": "…" } }` et portent un
statut HTTP d'échec. `Prediction` suit le vault et interdit `ground_truth`.
`request_id` est porté par la requête ; l'identité du résultat est définie par
`sample_id`/`input_sha256`. Son écho et les adaptateurs restent à convenir à G1.

Extension explicitement proposée au contrat v0.1 : importer d'abord via
`POST /samples`, puis utiliser le même ID pour écoute, visualisation et future
inférence. `POST /predict` n'accepte donc pas encore un fichier direct ni une
perturbation. Ceci évite deux chemins d'import divergents ; à valider ensemble.

## Audio et visualisation

- WAV décodé côté serveur, PCM 16/24/32 bits ou float32, mono/stéréo, 8–192 kHz,
  durée strictement positive ≤30 s, fichier ≤8 Mio. Refus des NaN/Inf. Silence
  accepté avec avertissement. Maximum 16 clips ; doublons dédupliqués par signal.
- Bibliothèque en mémoire, perdue au redémarrage ; aucune écriture audio sur
  disque par l'application. Le parseur multipart peut utiliser un fichier
  temporaire, fermé après lecture. Noms source jamais utilisés comme IDs/prompts.
- SHA-256 de `fréquence:canaux:` UTF-8 suivi des échantillons float32 little-endian
  entrelacés décodés. `sample_id` = 24 premiers caractères. L'audio servi conserve
  les octets d'origine ; le hash identifie le signal décodé, pas le conteneur WAV.
- Tous les imports sont explicitement `development_fixture`, provenance non
  vérifiée ; aucune mesure de performance ne peut en être déduite.
- Affichage seul `display-stft-v1` : moyenne mono, fenêtre Hann, FFT min(1024, N),
  hop FFT/4 (minimum 1), pas de padding ni resampling. Énergie moyennée par blocs
  avant conversion en dB, plancher −120 dB, référence amplitude numérique 1.
  Transport limité à 800 blocs waveform et 256×400 cellules spectrales.
- Ce traitement d'affichage est **distinct du DSP ML de Nevil**. Il n'est ni une
  attention du modèle ni une calibration en pression acoustique. Le modèle ne
  recevra pas les pixels du spectrogramme.

## Vérification

```sh
.venv/bin/python -m pytest -q
npm --prefix frontend run build
npm --prefix frontend test
```

Les WAV des tests sont synthétiques, générés à l'exécution et jamais intégrés
comme exemples scientifiques ou données de démo. Les tests couvrent le vrai
décodage, la cohérence audio/hash/visuels, le pic spectral connu, les limites,
NaN/Inf, silence, fichier corrompu, IDs, absence de faux résultats, suppression,
redémarrage, validation frontend et réponse obsolète après changement de clip.

Recette navigateur : importer un WAV de développement autorisé, vérifier durée,
lecture, boucle, curseur, spectrogramme, erreurs et comparaison indisponible.
Une recette sur signal synthétique ne valide pas G3 (vraie inférence attendue).

## Intégration suivante — hors de ce premier jalon

1. Nevil : livrer un WAV autorisé hors test scellé, sa provenance et le contrat
   final `SignalExample`/prétraitement. Safoan branche la bibliothèque démo.
2. Icham et Vincent : livrer les points d'entrée de chargement/prédiction et
   versions. Safoan charge chaque modèle une seule fois au démarrage, traite
   timeout/concurrence et affiche le `Prediction` validé, sans fallback caché.
3. Nevil : livrer `metrics.json`/provenance du run et, après G3, `mix_noise`.
   Safoan intègre les vues et la perturbation en conservant l'identité du signal.
4. Après vraie inférence : valider G3, enregistrer un replay réel clairement
   marqué, recette complète et vidéo. Aucun GPU public ni dépense lancés ici.
