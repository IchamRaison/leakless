# Pitch LeakLess — 5 minutes

Réponse à la note de revue n°3 (Safoan) : 5 minutes, slides puis démo rapide
du projet en action, sans parcourir la landing.

- Deck : `docs/pitch/leakless-pitch.html` (ouvrir dans un navigateur, hors
  ligne). Copie privée partageable : https://claude.ai/code/artifact/d91c68fa-72e2-438e-adc5-6e2defb28ab7
- Commandes : → / Espace suivant, ← précédent, N notes orateur, O vue
  d'ensemble (répétition, copie vers Google Slides/Keynote), F plein écran,
  T remet le chrono à zéro. Clic tiers gauche = précédent.
- Textes des slides en anglais comme le site ; chiffres repris uniquement de
  `frontend/src/demo/evidence.ts` (contrôles) et du rapport officiel gelé
  `frontend/src/demo/official/` (TSLM `tslm-v1`, protocol-freeze-v1, commit
  3e4e73ab6e19) : C4 clip AUC 0.665, cluster AUC 0.861 ; TSLM vs C1 : clip
  compatible with degradation, cluster inconclusive. Aucune probabilité par
  enregistrement.

## Minutage

| Slide | Créneau | Contenu |
|---|---|---|
| Problem | 0:00–0:40 | Dégâts visibles tard, signal acoustique plus tôt ; prototype |
| Data | 0:40–1:30 | 1 000 clips Zenodo, 185 clusters, split figé, contrôles C0–C4 |
| Live demo | 1:30–3:00 | Bascule vers le site (parcours ci-dessous) |
| Results | 3:00–4:10 | Échelle AUC C0–C4, 41 clusters test dont 11 non-leak |
| Limits | 4:10–4:45 | Démontré / non démontré, prochaine étape |
| Close | 4:45–5:00 | « The signal before the decision », équipe |

## Pré-vol (10 min avant)

1. API : `.venv/bin/python -m uvicorn pipe.api.main:app --app-dir src --host 127.0.0.1 --port 8028`
2. Site : `PIPE_API_URL=http://127.0.0.1:8028 npm --prefix frontend run dev -- --port 5178`
3. Onglet 1 : deck, plein écran (F). Onglet 2 : `http://127.0.0.1:5178/` (sans `?notes`).
4. Préchauffer : cliquer N1 puis les trois enregistrements une fois (cache),
   ouvrir `#monitor` et vérifier « Channels 3 / 3 », revenir à l'accueil.
5. Son du Mac activé ; un WAV court prêt sur le bureau pour la section 06.

## Parcours démo (90 s)

1. Bâtiment → **Fullscreen** → **N1** : « position illustrative, ne localise
   pas de fuite ».
2. Panneau → **REC 02** : vrai enregistrement, forme d'onde, label du dataset,
   « No physical association between this point and the recording ».
3. **Inspect this recording** → Monitor avec REC 02 au premier plan,
   « Alerts: N/A · replay only », TSLM · NO PER-RECORDING OUTPUT, **Listen** au besoin.
4. **Overview** : retour avec la même sélection ; 03 → Fullscreen sur la
   Temporal Signal Map si le temps le permet.
5. S'il reste du temps : 06 → **Load a recording** avec le WAV du bureau.

## Replis

- API tombée : les sections affichent « Signal unavailable » / Retry. Ne pas
  improviser de chiffres ; relancer l'API (étape 1) ou passer aux résultats.
- WebGL indisponible : les boutons N1–N4 et les graphiques 2D restent utilisables.
- Retard : sauter l'étape 5 de la démo et la slide Limits (garder sur la slide Results
  la phrase « The first TSLM did not outperform the strongest simple control »).
