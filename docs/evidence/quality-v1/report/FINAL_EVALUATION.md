# FINAL_EVALUATION — évaluation sur le split gelé

> Généré par `scripts/eval/build_final_report.py`. **Aucun chiffre n'est recopié à la main.** Régénérer le rapport régénère tous les nombres.

---

## 1. Identité des données et du split

| | |
|---|---|
| Split | `split_v2.csv` |
| SHA256 | `7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896` |
| Clips | 1000 |
| Clusters de dépendance | 185 |
| Commit du dépôt | `c27a43fdd4de73ccc09f5f4bc02acd87c89f58f6` |
| Règle d'agrégation | median des probabilités du cluster (gelée) |
| Bootstrap | 2000 tirages, graine 20260912, unité : cluster de dépendance |
| Source | Zenodo 18631450, CC BY 4.0 — site d'entraînement expérimental de Dongguan |

## 2. Identité des modèles

| run | modèle | checkpoint | commit d'entraînement | horodatage |
|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — raccourci d'acquisition | `logreg(C=0.01)` | `c27a43fdd4de` | 2026-09-12T20:51:24+00:00 |
| `c1` | C1 — forme d'amplitude seule, aucune information fréquentielle | `logreg(C=0.01)` | `c27a43fdd4de` | 2026-09-12T20:51:26+00:00 |
| `c2` | C2 — spectre agrégé, invariant à l'ordre, sans phase | `logreg(C=0.1)` | `c27a43fdd4de` | 2026-09-12T20:51:26+00:00 |
| `c2b` | C2b — structure temporelle peu profonde — enveloppe, modulation, flux | `logreg(C=0.1)` | `c27a43fdd4de` | 2026-09-12T20:51:36+00:00 |
| `c3` | C3 — baseline historique — mélange C1 + C2 + taux de passages par zéro | `logreg(C=0.01)` | `c27a43fdd4de` | 2026-09-12T20:51:37+00:00 |
| `tslm-v1` | AcousticQwenSP / Qwen3.5-4B V1 | `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle` | `1199789f471a` | 2026-09-12T19:41:52+00:00 |
| `tslm-v1-T1` | AcousticQwenSP / Qwen3.5-4B V1 | `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle` | `1199789f471a` | 2026-09-12T19:48:23+00:00 |
| `tslm-v1-T2` | AcousticQwenSP / Qwen3.5-4B V1 | `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle` | `1199789f471a` | 2026-09-12T19:49:36+00:00 |
| `tslm-v1-T3` | AcousticQwenSP / Qwen3.5-4B V1 | `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle` | `1199789f471a` | 2026-09-12T19:50:48+00:00 |

## 3. Échelle de contrôles et résultats — TEST

| run | modèle | clip AUC | clip PR-AUC | clip F1 | clip Brier | cluster AUC | cluster F1 | clips | clusters (leak/non-leak) |
|---|---|---|---|---|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — racc | 0.878 | 0.870 | 0.808 | 0.163 | 0.839 | 0.802 | 194 | 41 (30/11) |
| `c1` | C1 — forme d'amplitude seule, aucune infor | 0.902 | 0.878 | 0.856 | 0.140 | 0.927 | 0.849 | 194 | 41 (30/11) |
| `c2` | C2 — spectre agrégé, invariant à l'ordre,  | 0.710 | 0.711 | 0.680 | 0.214 | 0.779 | 0.776 | 194 | 41 (30/11) |
| `c2b` | C2b — structure temporelle peu profonde —  | 0.847 | 0.829 | 0.763 | 0.160 | 0.915 | 0.814 | 194 | 41 (30/11) |
| `c3` | C3 — baseline historique — mélange C1 + C2 | 0.824 | 0.792 | 0.758 | 0.173 | 0.900 | 0.856 | 194 | 41 (30/11) |
| `tslm-v1` | AcousticQwenSP / Qwen3.5-4B V1 | 0.665 | 0.637 | 0.646 | 0.241 | 0.861 | 0.799 | 194 | 41 (30/11) |
| `tslm-v1-T1` | AcousticQwenSP / Qwen3.5-4B V1 | 0.652 | 0.637 | 0.560 | 0.241 | 0.824 | 0.760 | 194 | 41 (30/11) |
| `tslm-v1-T2` | AcousticQwenSP / Qwen3.5-4B V1 | 0.612 | 0.604 | 0.587 | 0.244 | 0.700 | 0.684 | 194 | 41 (30/11) |
| `tslm-v1-T3` | AcousticQwenSP / Qwen3.5-4B V1 | 0.541 | 0.547 | 0.486 | 0.249 | 0.536 | 0.518 | 194 | 41 (30/11) |

### Validation (pour information — c'est là que le seuil est choisi)

| run | modèle | clip AUC | clip PR-AUC | clip F1 | clip Brier | cluster AUC | cluster F1 | clips | clusters (leak/non-leak) |
|---|---|---|---|---|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — racc | 0.870 | 0.866 | 0.817 | 0.168 | 0.812 | 0.846 | 208 | 42 (34/8) |
| `c1` | C1 — forme d'amplitude seule, aucune infor | 0.928 | 0.927 | 0.870 | 0.127 | 0.971 | 0.923 | 208 | 42 (34/8) |
| `c2` | C2 — spectre agrégé, invariant à l'ordre,  | 0.777 | 0.810 | 0.673 | 0.199 | 0.783 | 0.797 | 208 | 42 (34/8) |
| `c2b` | C2b — structure temporelle peu profonde —  | 0.870 | 0.875 | 0.773 | 0.149 | 0.665 | 0.635 | 208 | 42 (34/8) |
| `c3` | C3 — baseline historique — mélange C1 + C2 | 0.886 | 0.895 | 0.773 | 0.155 | 0.996 | 0.963 | 208 | 42 (34/8) |
| `tslm-v1` | AcousticQwenSP / Qwen3.5-4B V1 | 0.753 | 0.723 | 0.721 | 0.239 | 0.985 | 0.929 | 208 | 42 (34/8) |
| `tslm-v1-T1` | AcousticQwenSP / Qwen3.5-4B V1 | 0.675 | 0.635 | 0.555 | 0.242 | 0.938 | 0.863 | 208 | 42 (34/8) |
| `tslm-v1-T2` | AcousticQwenSP / Qwen3.5-4B V1 | 0.740 | 0.752 | 0.692 | 0.239 | 0.952 | 0.859 | 208 | 42 (34/8) |
| `tslm-v1-T3` | AcousticQwenSP / Qwen3.5-4B V1 | 0.654 | 0.612 | 0.591 | 0.245 | 0.831 | 0.846 | 208 | 42 (34/8) |

## 4. Incertitude — bootstrap sur les clusters

Les intervalles rééchantillonnent des **clusters entiers**, pour tenir compte des dépendances supposées entre clips. Ces groupes heuristiques ne sont pas des sessions d'acquisition indépendantes démontrées.

| run | clip AUC IC95 | cluster AUC IC95 | clip F1 IC95 |
|---|---|---|---|
| `c0` | [0.679, 0.932] | [0.647, 0.975] | [0.615, 0.862] |
| `c1` | [0.803, 0.988] | [0.788, 1.000] | [0.708, 0.909] |
| `c2` | [0.609, 0.819] | [0.576, 0.943] | [0.568, 0.765] |
| `c2b` | [0.758, 0.922] | [0.811, 0.984] | [0.643, 0.849] |
| `c3` | [0.751, 0.962] | [0.724, 1.000] | [0.656, 0.820] |
| `tslm-v1` | [0.577, 0.889] | [0.706, 0.977] | [0.527, 0.795] |
| `tslm-v1-T1` | [0.565, 0.876] | [0.648, 0.959] | [0.404, 0.777] |
| `tslm-v1-T2` | [0.523, 0.825] | [0.496, 0.878] | [0.474, 0.726] |
| `tslm-v1-T3` | [0.447, 0.686] | [0.335, 0.733] | [0.367, 0.639] |

## 5. Différences appariées

Un **seul** tirage de clusters sert aux deux modèles comparés. Comparer deux intervalles indépendants serait une erreur : cela ignore que les deux modèles voient les mêmes données.

| comparaison | métrique | Δ observé | IC95 | lecture |
|---|---|---|---|---|
| `tslm-v1_vs_c0` | clip_roc_auc | -0.213 | [-0.334, +0.162] | inconclusive |
| `tslm-v1_vs_c0` | cluster_roc_auc | +0.021 | [-0.187, +0.242] | inconclusive |
| `tslm-v1_vs_c0` | clip_macro_f1 | -0.162 | [-0.279, +0.128] | inconclusive |
| `tslm-v1_vs_c0` | cluster_macro_f1 | -0.003 | [-0.206, +0.202] | inconclusive |
| `tslm-v1_vs_c1` | clip_roc_auc | -0.237 | [-0.334, -0.001] | compatible with degradation |
| `tslm-v1_vs_c1` | cluster_roc_auc | -0.067 | [-0.180, +0.021] | inconclusive |
| `tslm-v1_vs_c1` | clip_macro_f1 | -0.210 | [-0.322, +0.015] | inconclusive |
| `tslm-v1_vs_c1` | cluster_macro_f1 | -0.050 | [-0.225, +0.127] | inconclusive |
| `tslm-v1_vs_c2` | clip_roc_auc | -0.045 | [-0.134, +0.202] | inconclusive |
| `tslm-v1_vs_c2` | cluster_roc_auc | +0.082 | [-0.064, +0.231] | inconclusive |
| `tslm-v1_vs_c2` | clip_macro_f1 | -0.035 | [-0.126, +0.162] | inconclusive |
| `tslm-v1_vs_c2` | cluster_macro_f1 | +0.023 | [-0.162, +0.212] | inconclusive |
| `tslm-v1_vs_c2b` | clip_roc_auc | -0.183 | [-0.292, +0.108] | inconclusive |
| `tslm-v1_vs_c2b` | cluster_roc_auc | -0.055 | [-0.248, +0.132] | inconclusive |
| `tslm-v1_vs_c2b` | clip_macro_f1 | -0.117 | [-0.223, +0.105] | inconclusive |
| `tslm-v1_vs_c2b` | cluster_macro_f1 | -0.016 | [-0.263, +0.238] | inconclusive |
| `tslm-v1_vs_c3` | clip_roc_auc | -0.159 | [-0.244, +0.032] | inconclusive |
| `tslm-v1_vs_c3` | cluster_roc_auc | -0.039 | [-0.161, +0.064] | inconclusive |
| `tslm-v1_vs_c3` | clip_macro_f1 | -0.112 | [-0.192, +0.038] | inconclusive |
| `tslm-v1_vs_c3` | cluster_macro_f1 | -0.058 | [-0.206, +0.084] | inconclusive |
| `tslm-v1_vs_tslm-v1-T1` | clip_roc_auc | +0.012 | [-0.032, +0.065] | inconclusive |
| `tslm-v1_vs_tslm-v1-T1` | cluster_roc_auc | +0.036 | [-0.027, +0.111] | inconclusive |
| `tslm-v1_vs_tslm-v1-T1` | clip_macro_f1 | +0.086 | [-0.075, +0.151] | inconclusive |
| `tslm-v1_vs_tslm-v1-T1` | cluster_macro_f1 | +0.038 | [-0.106, +0.208] | inconclusive |
| `tslm-v1_vs_tslm-v1-T2` | clip_roc_auc | +0.052 | [+0.003, +0.166] | compatible with improvement |
| `tslm-v1_vs_tslm-v1-T2` | cluster_roc_auc | +0.161 | [+0.037, +0.323] | compatible with improvement |
| `tslm-v1_vs_tslm-v1-T2` | clip_macro_f1 | +0.059 | [+0.007, +0.128] | compatible with improvement |
| `tslm-v1_vs_tslm-v1-T2` | cluster_macro_f1 | +0.115 | [-0.018, +0.261] | inconclusive |
| `tslm-v1_vs_tslm-v1-T3` | clip_roc_auc | +0.124 | [+0.058, +0.304] | compatible with improvement |
| `tslm-v1_vs_tslm-v1-T3` | cluster_roc_auc | +0.324 | [+0.162, +0.500] | compatible with improvement |
| `tslm-v1_vs_tslm-v1-T3` | clip_macro_f1 | +0.160 | [+0.065, +0.247] | compatible with improvement |
| `tslm-v1_vs_tslm-v1-T3` | cluster_macro_f1 | +0.281 | [+0.094, +0.461] | compatible with improvement |

> ⚠️ **Aucune significativité statistique n'est revendiquée.** Le test compte 41 groupes de dépendance heuristiques, dont 11 du côté *non-leak*. Les verdicts sont qualitatifs.

## 6. Résultat TSLM

Run `tslm-v1` — AcousticQwenSP / Qwen3.5-4B V1

- clip : AUC 0.665, macro-F1 0.646, Brier 0.241
- cluster : AUC 0.861, macro-F1 0.799
- effectifs : 194 clips, 41 clusters (30 leak / 11 non-leak)

### Décisions et erreurs au seuil retenu — TEST

| niveau | fuites | non-fuites | FP | FN | précision fuite | F1 fuite | rappel fuite | FPR | taux manquées |
|---|---|---|---|---|---|---|---|---|---|
| clip | 98 | 96 | 43 | 25 | 62.9 % | 68.2 % | 74.5 % | 44.8 % | 25.5 % |
| groupe | 30 | 11 | 2 | 5 | 92.6 % | 87.7 % | 83.3 % | 18.2 % | 16.7 % |

F1 fuite ≠ macro-F1. Pourcentages dérivés des comptes TP/FP/TN/FN au seuil du run choisi sur validation ; — indique un dénominateur nul.

Comparaisons appariées du TSLM contre chaque contrôle : voir le tableau §5.

## 7. Tests de stress temporel

_(aucun rapport d'invariants fourni ; les métriques des runs stressés éventuellement présents figurent aux §3–5)_

## 8. Limites connues

1. **11 groupes *non-leak* en test, 8 en validation.** L'effectif limite la précision ; lire les intervalles et verdicts effectivement calculés aux §4–5, sans préjuger de leur largeur ni de leur recouvrement.
2. **C0 et C1 sondent les raccourcis d'acquisition.** La comparaison du niveau absolu (C0) et de la forme d'amplitude normalisée (C1) doit se lire dans les résultats fournis ; aucune supériorité n'est présumée par le gabarit.
3. **Les clusters ne sont pas des sessions d'acquisition démontrées** : les regroupements sont heuristiques, pas une preuve d'indépendance des acquisitions.
4. **La chaîne d'acquisition n'est pas calibrée.** Rien ne garantit qu'un écart mesuré ici survive à un autre matériel.
5. **Aucune donnée de terrain, aucun client.** Le dataset vient d'un site d'entraînement expérimental.

## 9. Affirmations permises

- « Nous construisons un pipeline logiciel reproductible avec des contrôles de fuite stricts, sur un split gelé et vérifié. »
- Toute affirmation de performance ou de raccourci d'acquisition doit renvoyer aux contrôles, métriques et comparaisons effectivement présents dans ce rapport.
- « Les métriques sont group-aware : l'unité de rééchantillonnage est la grappe de dépendance, et son effectif accompagne chaque chiffre. »
- « Les comparaisons sont appariées, sur les mêmes tirages de clusters. »
- Tout écart, uniquement avec *compatible with improvement* / *inconclusive* / *compatible with degradation*.

## 10. Affirmations interdites

- ❌ « LeakLess détecte des fuites sur le terrain. » — aucune donnée client, aucune validation terrain, aucun matériel testé.
- ❌ « Le modèle localise la fuite. » — le jeu ne supporte pas la localisation.
- ❌ « L'écart est statistiquement significatif. » — 41 clusters.
- ❌ Un score sans son nombre de clusters.
- ❌ Un intervalle de confiance bootstrapé sur les clips.
- ❌ Toute utilisation de `split_v1`, invalide.
- ❌ « La modélisation temporelle apporte de la valeur » sans comparaison appropriée et preuves suffisantes ; la seule présence d'un run TSLM ne le démontre pas.

---

Matrice claim → artefact : [`JURY_EVIDENCE_MATRIX.md`](JURY_EVIDENCE_MATRIX.md). Contrat de livraison du modèle : [`MODEL_EVAL_CONTRACT.md`](MODEL_EVAL_CONTRACT.md).
