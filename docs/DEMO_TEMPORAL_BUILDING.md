# Démo bâtiment — LeakLess Temporal AI

Icham

## Périmètre et reprise

Travail demandé par Nevil sur `feat/demo-temporal-building`, basé sur
`origin/feat/safoan-app` au commit `4dd7b88`. Worktree isolé :
`/Users/nevil/dev/sandbox/ehl-zurich/demo-temporal-building`.
Ni merge main, ni push, ni commit dans cette livraison en attente de revue.

Cette démo est la vue d'inspection sur clips demandée par Nevil : bâtiment de
contexte → point de mesure → vrai WAV → propriétés mesurées → question d'inspection.
Elle ne réalise pas le plan de surveillance continue proposé séparément dans le
vault, ni un flux capteur, une alerte automatique ou une localisation de fuite.

## Réutilisation

- LeakLess est l'unique expérience visible. L'ancien écran PIPE (`Studio.tsx`,
  route `#studio`, lien « Acoustic studio ») est retiré ; il reste consultable
  dans l'historique Git (`frontend/src/App.tsx` au commit `4dd7b88`).
- Ses briques techniques sont intégrées à LeakLess : upload WAV, validation
  client (extension, 8 Mio) et serveur, API FastAPI, lecteur audio, `SignalView`
  waveform/spectrogramme, identité SHA-256 et provenance. Lien de navigation
  « Load a recording » → section `06 / INSPECT YOUR OWN SIGNAL`
  (`frontend/src/demo/InspectRecording.tsx`). Un upload valide affiche, dans la
  même page : métadonnées, Temporal Signal Map, waveform, spectrogramme, lecteur
  et TSLM `NOT RUN` (`ModelReadout.tsx`, partagé avec la section 04). Un clic
  explicite appelle `/predict`; sans artefacts V2, l'indisponibilité est affichée.
  Nom de fichier neutre `recording.wav` envoyé ;
  l'upload précédent est libéré côté serveur (DELETE) quand il est remplacé ou
  retiré. Aucun backend ML ajouté.
- Textes visibles en anglais uniquement, y compris les messages d'erreur de
  l'API relayés à l'interface et les libellés de `SignalView` (Waveform,
  Frequency, Digital amplitude). Format des durées `en-US`.
- Scène source inspectée : `/Users/nevil/Projets/leakless-hackathon/building-scene.html`.
  Port minimal des cadres, fenêtres, tubes et contrôles orbitaux. La logique de
  fuite par étage, réparation et animation d'écoulement n'est pas reprise.
- Un seul moteur 3D : Three.js 0.185.0, encapsulé dans React. React Three Fiber
  n'était pas installé ; il n'est pas ajouté. Le studio et la démo sont chargés
  à la demande, sans changement de framework ni routeur supplémentaire.

## Lancement local vérifié

```sh
# À la racine du worktree, environnement CPU seulement
uv venv --python /opt/homebrew/bin/python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-api.txt
npm --prefix frontend ci
.venv/bin/python -m uvicorn pipe.api.main:app --app-dir src --host 127.0.0.1 --port 8018

# Dans un second terminal
PIPE_API_URL=http://127.0.0.1:8018 npm --prefix frontend run dev -- --port 5178
```

Ouvrir `http://127.0.0.1:5178/`, pas `frontend/index.html` via `file://`.
Les ports 8018/5178 évitent le service déjà présent sur 5173.
Sans `PIPE_API_URL`, le proxy Vite garde la cible Safoan par défaut, port 8000.
Sans API : bâtiment disponible, état d'erreur/retry pour le signal, aucune valeur
de mesure fabriquée. `vite preview` seul ne fournit pas l'API.

## Sources et invariants de l'interface

- Résultats autorisés centralisés dans `frontend/src/demo/evidence.ts` : C0, C1,
  C2, C2b, C3, C4 PENDING. AUC clip/cluster, 41 clusters dont 11 non-leak et
  limites visibles. Aucun ancien chiffre T2/T3, aucun nouveau calcul d'évaluation.
- Trois WAV réels de train (environ 48 Ko au total), crédités dans
  `frontend/public/demo/README.md`. Sélection déterministe du premier clip de
  chaque catégorie dans l'ordre du manifeste, pas de sélection selon un score.
- N1–N4 sont des positions illustratives et ne localisent aucune fuite. Choisir
  un point n'associe plus de catégorie : le premier choix charge l'exemple
  no-leak, puis l'enregistrement choisi reste le même d'un point à l'autre.
  Interface : « Selected measurement point: N1 — illustrative building
  position », puis « Demo recording: [No-leak] [Leak-associated]
  [Environmental noise] ». Ce sont trois enregistrements expérimentaux de
  démonstration, pas des classes prédites.
- SHA-256 du WAV vérifié avant import, puis hash du signal décodé et identité
  de la visualisation vérifiés. Import sous nom neutre `recording.wav`.
- Requêtes abandonnées ignorées lors d'une nouvelle sélection ; aucun ancien
  résultat ne peut être affiché sous la nouvelle identité. Cache local aux trois
  exemples, vidé par retry. L'audio original est servi comme asset local.
- Géométrie déterministe, sans RNG : temps vertical ; rayon selon puissance
  spectrale moyenne ; trois épaisseurs selon les tiers fréquentiels ; torsion
  selon centroïde spectral. Échelles fixes, pas de normalisation par clip.
- Ces mesures viennent de la STFT d'affichage Safoan, distincte du DSP ML. Une
  ellipse et la rotation sont des conventions de présentation. Pas de mapping
  flatness/irrégularité non justifié. Méthode et première mesure consultables.
- Matériau neutre avant exécution : aucune couleur de classe déduite du label
  d'exemple. Le bouton TSLM V2 déclenche une vraie requête pour le signal déjà
  importé. Pas de score inventé ni de recommandation terrain.
- ResizeObserver, arrêt de rotation, respect de reduced-motion, nettoyage des
  renderers/géométries/matériaux et sélection accessible en HTML. Repli explicite
  lorsque WebGL n'est pas disponible.

## Validation et limites de livraison

Commandes : `npm --prefix frontend run build`, `npm --prefix frontend test`,
`.venv/bin/python -m pytest -q`.

Résultats après intégration V2, 2026-09-13 : build TypeScript/Vite réussi ;
35 tests frontend (15 fichiers), 23 tests API et les 12 tests unitaires du
wrapper cohérent réussis. Tests de déterminisme/mesures, identité incorrecte, réponse
tardive, retry, indépendance point/enregistrement, upload (type, taille, erreur
serveur, API injoignable, identité incohérente, métadonnées + carte + NOT RUN,
libération au remplacement/retrait), appel explicite V2, provenance et rejet
d'une réponse modèle obsolète. Résultats scientifiques inchangés. Aucun test de
performance ou d'inférence sur poids réels réalisé ici.

Recette navigateur (agent-browser headless, API réelle sur 8028, Vite 5178) :
bâtiment 3D rendu, N2/N3 → exemple no-leak, waveform/spectrogramme en anglais,
upload d'un WAV invalide (message « Unreadable or corrupted WAV. »), upload d'un
WAV stéréo 2 s valide (métadonnées, carte 61 fenêtres, plots, audio 200),
remplacement puis retrait (deux DELETE 204), ancienne URL `#studio` → LeakLess.
Aucun texte PIPE/Safoan/studio ni caractère français dans la page, aucun appel
`/predict` dans le journal API, pas de débordement horizontal en 1440×900 et
390×844, aucune erreur ni avertissement console.

Limites techniques : le bundle 3D déclenche un avertissement Vite >500 Ko
minifié (environ 178 Ko gzip pour la vue démo, zod/API désormais inclus dans ce chunk). Les tests API hérités émettent
deux avertissements de dépréciation Starlette/AnyIO. Ce ne sont pas des échecs
de build ni des erreurs console de la démo. Matériel WebGL nécessaire à la 3D.
L'adaptateur TSLM V2 est intégré. L'inférence réelle attend encore les trois
artefacts validés non versionnés ; la validation terrain reste hors livraison.

Prochaine action : revue de l'UI et du diff par Nevil avant tout push ; puis
intégration d'une sortie modèle réelle au contrat applicatif convenu, sans
modifier les résultats, le split ou le moteur d'évaluation.

## Revue Safoan du 2026-09-12 (notes épinglées)

Notes stockées dans `notes/site-notes.json` (mode `?notes`, serveur de dev
seulement), toutes marquées résolues après traitement :

1. Plein écran du bâtiment → bouton Fullscreen (bâtiment et carte temporelle).
2. Landing appréciée, à ne pas changer → hero inchangé.
3. Pitch de 5 min, slides + démo rapide → `docs/pitch/` (deck HTML, minutage,
   pré-vol, parcours démo 90 s, replis).
4. Dashboard pour la démo → `#monitor` : replay des trois vrais WAV train,
   niveau/centroïde/bandes mesurés, aucune alerte générée, TSLM non exécuté,
   canaux distincts des positions du bâtiment.
