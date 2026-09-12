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
| Commit du dépôt | `2f61695072e0f16205d6a7047243c59600eb2792` |
| Règle d'agrégation | median des probabilités du cluster (gelée) |
| Bootstrap | 2000 tirages, graine 20260912, unité : cluster de dépendance |
| Source | Zenodo 18631450, CC BY 4.0 — site d'entraînement expérimental de Dongguan |

## 2. Identité des modèles

| run | modèle | checkpoint | commit d'entraînement | horodatage |
|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — raccourci d'acquisition | `logreg(C=0.01)` | `1289095fccd5` | 2026-09-12T16:16:12+00:00 |
| `c1` | C1 — forme d'amplitude seule, aucune information fréquentielle | `logreg(C=0.01)` | `2f61695072e0` | 2026-09-12T16:26:40+00:00 |
| `c2` | C2 — spectre agrégé, invariant à l'ordre, sans phase | `logreg(C=0.1)` | `1289095fccd5` | 2026-09-12T16:16:13+00:00 |
| `c3` | C3 — baseline historique — mélange C1 + C2 + taux de passages par zéro | `logreg(C=0.01)` | `1289095fccd5` | 2026-09-12T16:16:13+00:00 |

## 3. Échelle de contrôles et résultats — TEST

| run | modèle | clip AUC | clip PR-AUC | clip F1 | clip Brier | cluster AUC | cluster F1 | clips | clusters (leak/non-leak) |
|---|---|---|---|---|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — racc | 0.878 | 0.870 | 0.808 | 0.163 | 0.839 | 0.802 | 194 | 41 (30/11) |
| `c1` | C1 — forme d'amplitude seule, aucune infor | 0.902 | 0.878 | 0.856 | 0.140 | 0.927 | 0.849 | 194 | 41 (30/11) |
| `c2` | C2 — spectre agrégé, invariant à l'ordre,  | 0.710 | 0.711 | 0.680 | 0.214 | 0.779 | 0.776 | 194 | 41 (30/11) |
| `c3` | C3 — baseline historique — mélange C1 + C2 | 0.824 | 0.792 | 0.758 | 0.173 | 0.900 | 0.856 | 194 | 41 (30/11) |

### Validation (pour information — c'est là que le seuil est choisi)

| run | modèle | clip AUC | clip PR-AUC | clip F1 | clip Brier | cluster AUC | cluster F1 | clips | clusters (leak/non-leak) |
|---|---|---|---|---|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — racc | 0.870 | 0.866 | 0.817 | 0.168 | 0.812 | 0.846 | 208 | 42 (34/8) |
| `c1` | C1 — forme d'amplitude seule, aucune infor | 0.928 | 0.927 | 0.870 | 0.127 | 0.971 | 0.923 | 208 | 42 (34/8) |
| `c2` | C2 — spectre agrégé, invariant à l'ordre,  | 0.777 | 0.810 | 0.673 | 0.199 | 0.783 | 0.797 | 208 | 42 (34/8) |
| `c3` | C3 — baseline historique — mélange C1 + C2 | 0.886 | 0.895 | 0.773 | 0.155 | 0.996 | 0.963 | 208 | 42 (34/8) |

## 4. Incertitude — bootstrap sur les clusters

Les intervalles rééchantillonnent des **clusters entiers**. Un bootstrap au clip rééchantillonnerait à l'intérieur de sessions quasi identiques et produirait des intervalles bien trop étroits.

| run | clip AUC IC95 | cluster AUC IC95 | clip F1 IC95 |
|---|---|---|---|
| `c0` | [0.679, 0.932] | [0.647, 0.975] | [0.615, 0.862] |
| `c1` | [0.803, 0.988] | [0.788, 1.000] | [0.708, 0.909] |
| `c2` | [0.609, 0.819] | [0.576, 0.943] | [0.568, 0.765] |
| `c3` | [0.751, 0.962] | [0.724, 1.000] | [0.656, 0.820] |

## 5. Différences appariées

Un **seul** tirage de clusters sert aux deux modèles comparés. Comparer deux intervalles indépendants serait une erreur : cela ignore que les deux modèles voient les mêmes données.

| comparaison | métrique | Δ observé | IC95 | lecture |
|---|---|---|---|---|
| `c0_vs_c1` | clip_roc_auc | -0.024 | [-0.219, +0.035] | inconclusive |
| `c0_vs_c1` | cluster_roc_auc | -0.088 | [-0.284, +0.077] | inconclusive |
| `c0_vs_c1` | clip_macro_f1 | -0.047 | [-0.173, +0.017] | inconclusive |
| `c0_vs_c1` | cluster_macro_f1 | -0.047 | [-0.175, +0.063] | inconclusive |
| `c0_vs_c2` | clip_roc_auc | +0.168 | [-0.059, +0.247] | inconclusive |
| `c0_vs_c2` | cluster_roc_auc | +0.061 | [-0.192, +0.294] | inconclusive |
| `c0_vs_c2` | clip_macro_f1 | +0.128 | [-0.042, +0.201] | inconclusive |
| `c0_vs_c2` | cluster_macro_f1 | +0.026 | [-0.115, +0.181] | inconclusive |
| `c0_vs_c3` | clip_roc_auc | +0.054 | [-0.177, +0.121] | inconclusive |
| `c0_vs_c3` | cluster_roc_auc | -0.061 | [-0.249, +0.135] | inconclusive |
| `c0_vs_c3` | clip_macro_f1 | +0.050 | [-0.131, +0.119] | inconclusive |
| `c0_vs_c3` | cluster_macro_f1 | -0.055 | [-0.199, +0.082] | inconclusive |
| `c1_vs_c2` | clip_roc_auc | +0.192 | [+0.094, +0.284] | compatible with improvement |
| `c1_vs_c2` | cluster_roc_auc | +0.148 | [+0.041, +0.284] | compatible with improvement |
| `c1_vs_c2` | clip_macro_f1 | +0.175 | [+0.047, +0.256] | compatible with improvement |
| `c1_vs_c2` | cluster_macro_f1 | +0.073 | [-0.065, +0.225] | inconclusive |
| `c1_vs_c3` | clip_roc_auc | +0.078 | [-0.026, +0.111] | inconclusive |
| `c1_vs_c3` | cluster_roc_auc | +0.027 | [-0.011, +0.087] | inconclusive |
| `c1_vs_c3` | clip_macro_f1 | +0.098 | [-0.033, +0.173] | inconclusive |
| `c1_vs_c3` | cluster_macro_f1 | -0.007 | [-0.131, +0.105] | inconclusive |
| `c2_vs_c3` | clip_roc_auc | -0.114 | [-0.235, -0.059] | compatible with degradation |
| `c2_vs_c3` | cluster_roc_auc | -0.121 | [-0.264, -0.019] | compatible with degradation |
| `c2_vs_c3` | clip_macro_f1 | -0.077 | [-0.174, +0.014] | inconclusive |
| `c2_vs_c3` | cluster_macro_f1 | -0.080 | [-0.222, +0.042] | inconclusive |

> ⚠️ **Aucune significativité statistique n'est revendiquée.** Le test compte 41 clusters indépendants, dont 11 du côté *non-leak*. Les verdicts sont qualitatifs.

## 6. Résultat TSLM

🕐 **Aucun run de TSLM fourni.** Les contrôles sont mesurés, la question temporelle reste ouverte.

Pour l'intégrer : `docs/MODEL_EVAL_CONTRACT.md`, puis relancer cette commande en ajoutant le dossier du run.

## 7. Tests de stress temporel

Jeux préparés, invariants **mesurés** et non supposés :

| | transformation | records | clusters | `|FFT|` dév. médiane | histogramme d'amplitude | violations de mapping |
|---|---|---|---|---|---|---|
| T0 | original, aucune transformation | 1000 | 185 | 0.00e+00 | identique | 0 |
| T1 | inversion temporelle | 1000 | 185 | 2.67e-16 | identique | 0 |
| T2 | permutation de blocs de 250 échantillons (31,25 ms) | 1000 | 185 | 6.95e-01 | identique | 0 |
| T3 | randomisation de phase, module préservé | 1000 | 185 | 2.79e-16 | modifié | 0 |

> Ce ne sont **pas** des augmentations préservant l'étiquette. Un modèle dont le score ne bouge pas sous T1/T2/T3 n'utilise pas l'ordre temporel.

🕐 **Non évalués** : nous ne possédons pas le checkpoint du TSLM.

## 8. Limites connues

1. **11 clusters *non-leak* en test, 8 en validation.** Tous les intervalles sont larges et se recouvrent. Aucune comparaison n'est concluante à cette taille.
2. **La normalisation d'amplitude ne neutralise pas le raccourci d'acquisition.** C1 (forme d'enveloppe seule, audio normalisé) fait au moins aussi bien que C0 (niveau absolu). Le barreau à dépasser est C1, pas C0.
3. **Les clusters ne sont pas des sessions d'acquisition démontrées** : la source ne publie ni site, ni conduite, ni horodatage. Ce sont des grappes de dépendance.
4. **La chaîne d'acquisition n'est pas calibrée.** Rien ne garantit qu'un écart mesuré ici survive à un autre matériel.
5. **Aucune donnée de terrain, aucun client.** Le dataset vient d'un site d'entraînement expérimental.

## 9. Affirmations permises

- « Nous construisons un pipeline logiciel reproductible avec des contrôles de fuite stricts, sur un split gelé et vérifié. »
- « Le niveau sonore absolu est un raccourci mesuré sur ce jeu. »
- « La forme d'amplitude seule, après normalisation, atteint au moins le niveau du contrôle de volume : la normalisation ne suffit pas. »
- « Les métriques sont group-aware : l'unité d'indépendance est la grappe de dépendance, et son effectif accompagne chaque chiffre. »
- « Les comparaisons sont appariées, sur les mêmes tirages de clusters. »
- Tout écart, uniquement avec *compatible with improvement* / *inconclusive* / *compatible with degradation*.

## 10. Affirmations interdites

- ❌ « LeakLess détecte des fuites sur le terrain. » — aucune donnée client, aucune validation terrain, aucun matériel testé.
- ❌ « Le modèle localise la fuite. » — le jeu ne supporte pas la localisation.
- ❌ « L'écart est statistiquement significatif. » — 41 clusters.
- ❌ Un score sans son nombre de clusters.
- ❌ Un intervalle de confiance bootstrapé sur les clips.
- ❌ Toute utilisation de `split_v1`, invalide.
- ❌ « La modélisation temporelle apporte de la valeur » tant que la comparaison TSLM n'a pas été faite — **elle ne l'est pas encore**.

---

Matrice claim → artefact : [`JURY_EVIDENCE_MATRIX.md`](JURY_EVIDENCE_MATRIX.md). Contrat de livraison du modèle : [`MODEL_EVAL_CONTRACT.md`](MODEL_EVAL_CONTRACT.md).
