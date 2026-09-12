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
| Commit de génération du rapport | `00efdb4ed7ff9377c54c3635e5ce470c61002a4f` |
| Règle d'agrégation | median des probabilités du cluster (gelée) |
| Bootstrap | 2000 tirages, graine 20260912, unité : cluster de dépendance |
| Source | Zenodo 18631450, CC BY 4.0 — site d'entraînement expérimental de Dongguan |

## 2. Identité des modèles — échelle de contrôles

Trois rôles de commit, qui peuvent pointer vers le même commit, tous lus dans `metadata.json` et jamais recalculés ici : **définition** (où les descripteurs ont été figés), **ajustement** (`training_commit`, HEAD capturé juste avant le fit), **exécution** (HEAD à l'écriture du run). Aucun contrôle n'a de checkpoint sérialisé : la régression logistique est réajustée de façon déterministe sur T0/train à chaque exécution.

| run | modèle | checkpoint | commit de définition | commit d'ajustement | commit d'exécution | horodatage |
|---|---|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — raccourci d'acquisition | `aucun checkpoint sérialisé : logreg(C=0.01) ajustée de façon déterministe sur T0/train` | `2f61695072e0` | `0a22990f6451` | `0a22990f6451` | 2026-09-12T19:23:46+00:00 |
| `c1` | C1 — forme d'amplitude seule, aucune information fréquentielle | `aucun checkpoint sérialisé : logreg(C=0.01) ajustée de façon déterministe sur T0/train` | `2f61695072e0` | `0a22990f6451` | `0a22990f6451` | 2026-09-12T19:23:47+00:00 |
| `c2` | C2 — spectre agrégé, invariant à l'ordre, sans phase | `aucun checkpoint sérialisé : logreg(C=0.1) ajustée de façon déterministe sur T0/train` | `2f61695072e0` | `0a22990f6451` | `0a22990f6451` | 2026-09-12T19:23:47+00:00 |
| `c2b` | C2b — structure temporelle peu profonde — enveloppe, modulation, flux | `aucun checkpoint sérialisé : logreg(C=0.1) ajustée de façon déterministe sur T0/train` | `355f0743fb30` | `0a22990f6451` | `0a22990f6451` | 2026-09-12T19:23:55+00:00 |
| `c3` | C3 — baseline historique — mélange C1 + C2 + taux de passages par zéro | `aucun checkpoint sérialisé : logreg(C=0.01) ajustée de façon déterministe sur T0/train` | `2f61695072e0` | `0a22990f6451` | `0a22990f6451` | 2026-09-12T19:23:48+00:00 |

## 3. Échelle de contrôles et résultats — TEST

| run | modèle | clip AUC | clip PR-AUC | clip F1 | clip Brier | cluster AUC | cluster F1 | clips | clusters (leak/non-leak) |
|---|---|---|---|---|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — racc | 0.878 | 0.870 | 0.808 | 0.163 | 0.839 | 0.802 | 194 | 41 (30/11) |
| `c1` | C1 — forme d'amplitude seule, aucune infor | 0.902 | 0.878 | 0.856 | 0.140 | 0.927 | 0.849 | 194 | 41 (30/11) |
| `c2` | C2 — spectre agrégé, invariant à l'ordre,  | 0.710 | 0.711 | 0.680 | 0.214 | 0.779 | 0.776 | 194 | 41 (30/11) |
| `c2b` | C2b — structure temporelle peu profonde —  | 0.847 | 0.829 | 0.763 | 0.160 | 0.915 | 0.814 | 194 | 41 (30/11) |
| `c3` | C3 — baseline historique — mélange C1 + C2 | 0.824 | 0.792 | 0.758 | 0.173 | 0.900 | 0.856 | 194 | 41 (30/11) |

### Validation (pour information — c'est là que le seuil est choisi)

| run | modèle | clip AUC | clip PR-AUC | clip F1 | clip Brier | cluster AUC | cluster F1 | clips | clusters (leak/non-leak) |
|---|---|---|---|---|---|---|---|---|---|
| `c0` | C0 — niveau RMS absolu, signal brut — racc | 0.870 | 0.866 | 0.817 | 0.168 | 0.812 | 0.846 | 208 | 42 (34/8) |
| `c1` | C1 — forme d'amplitude seule, aucune infor | 0.928 | 0.927 | 0.870 | 0.127 | 0.971 | 0.923 | 208 | 42 (34/8) |
| `c2` | C2 — spectre agrégé, invariant à l'ordre,  | 0.777 | 0.810 | 0.673 | 0.199 | 0.783 | 0.797 | 208 | 42 (34/8) |
| `c2b` | C2b — structure temporelle peu profonde —  | 0.870 | 0.875 | 0.773 | 0.149 | 0.665 | 0.635 | 208 | 42 (34/8) |
| `c3` | C3 — baseline historique — mélange C1 + C2 | 0.886 | 0.895 | 0.773 | 0.155 | 0.996 | 0.963 | 208 | 42 (34/8) |

## 4. Incertitude — bootstrap sur les clusters

Les intervalles rééchantillonnent des **clusters entiers**. Un bootstrap au clip rééchantillonnerait à l'intérieur de sessions quasi identiques et produirait des intervalles bien trop étroits.

| run | clip AUC IC95 | cluster AUC IC95 | clip F1 IC95 |
|---|---|---|---|
| `c0` | [0.679, 0.932] | [0.647, 0.975] | [0.615, 0.862] |
| `c1` | [0.803, 0.988] | [0.788, 1.000] | [0.708, 0.909] |
| `c2` | [0.609, 0.819] | [0.576, 0.943] | [0.568, 0.765] |
| `c2b` | [0.758, 0.922] | [0.811, 0.984] | [0.643, 0.849] |
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
| `c0_vs_c2b` | clip_roc_auc | +0.031 | [-0.155, +0.093] | inconclusive |
| `c0_vs_c2b` | cluster_roc_auc | -0.076 | [-0.271, +0.087] | inconclusive |
| `c0_vs_c2b` | clip_macro_f1 | +0.045 | [-0.132, +0.122] | inconclusive |
| `c0_vs_c2b` | cluster_macro_f1 | -0.013 | [-0.175, +0.174] | inconclusive |
| `c0_vs_c3` | clip_roc_auc | +0.054 | [-0.177, +0.121] | inconclusive |
| `c0_vs_c3` | cluster_roc_auc | -0.061 | [-0.249, +0.135] | inconclusive |
| `c0_vs_c3` | clip_macro_f1 | +0.050 | [-0.131, +0.119] | inconclusive |
| `c0_vs_c3` | cluster_macro_f1 | -0.055 | [-0.199, +0.082] | inconclusive |
| `c1_vs_c2` | clip_roc_auc | +0.192 | [+0.094, +0.284] | compatible with improvement |
| `c1_vs_c2` | cluster_roc_auc | +0.148 | [+0.041, +0.284] | compatible with improvement |
| `c1_vs_c2` | clip_macro_f1 | +0.175 | [+0.047, +0.256] | compatible with improvement |
| `c1_vs_c2` | cluster_macro_f1 | +0.073 | [-0.065, +0.225] | inconclusive |
| `c1_vs_c2b` | clip_roc_auc | +0.055 | [-0.058, +0.164] | inconclusive |
| `c1_vs_c2b` | cluster_roc_auc | +0.012 | [-0.157, +0.157] | inconclusive |
| `c1_vs_c2b` | clip_macro_f1 | +0.093 | [-0.062, +0.188] | inconclusive |
| `c1_vs_c2b` | cluster_macro_f1 | +0.034 | [-0.171, +0.249] | inconclusive |
| `c1_vs_c3` | clip_roc_auc | +0.078 | [-0.026, +0.111] | inconclusive |
| `c1_vs_c3` | cluster_roc_auc | +0.027 | [-0.011, +0.087] | inconclusive |
| `c1_vs_c3` | clip_macro_f1 | +0.098 | [-0.033, +0.173] | inconclusive |
| `c1_vs_c3` | cluster_macro_f1 | -0.007 | [-0.131, +0.105] | inconclusive |
| `c2_vs_c2b` | clip_roc_auc | -0.137 | [-0.269, +0.004] | inconclusive |
| `c2_vs_c2b` | cluster_roc_auc | -0.136 | [-0.368, +0.061] | inconclusive |
| `c2_vs_c2b` | clip_macro_f1 | -0.082 | [-0.193, +0.022] | inconclusive |
| `c2_vs_c2b` | cluster_macro_f1 | -0.039 | [-0.252, +0.179] | inconclusive |
| `c2_vs_c3` | clip_roc_auc | -0.114 | [-0.235, -0.059] | compatible with degradation |
| `c2_vs_c3` | cluster_roc_auc | -0.121 | [-0.264, -0.019] | compatible with degradation |
| `c2_vs_c3` | clip_macro_f1 | -0.077 | [-0.174, +0.014] | inconclusive |
| `c2_vs_c3` | cluster_macro_f1 | -0.080 | [-0.222, +0.042] | inconclusive |
| `c2b_vs_c3` | clip_roc_auc | +0.023 | [-0.154, +0.123] | inconclusive |
| `c2b_vs_c3` | cluster_roc_auc | +0.015 | [-0.152, +0.225] | inconclusive |
| `c2b_vs_c3` | clip_macro_f1 | +0.005 | [-0.118, +0.134] | inconclusive |
| `c2b_vs_c3` | cluster_macro_f1 | -0.042 | [-0.266, +0.173] | inconclusive |

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

> Ces transformations ne sont **pas** des augmentations physiques démontrées comme préservant l'étiquette. Elles mesurent la sensibilité des prédictions à l'organisation temporelle, pas la pertinence physique causale de celle-ci.

### Scores mesurés sous stress

Aucun modèle n'est ajusté ni entraîné sur les données stressées : les transformations ne s'appliquent qu'à l'évaluation. Identité du modèle, vérifiée pour chaque run de stress avant toute comparaison :

- `c2b` : **aucun checkpoint sérialisé**. Définition figée `355f074`, régression logistique réajustée de façon déterministe sur T0/train dans le même processus que l'évaluation sous stress ; empreinte `model_fingerprint` identique entre le run T0 et ses runs de stress.

Seules des métriques **indépendantes du seuil** figurent ici : AUC, corrélation des probabilités avec T0 et distribution de Δp, sur le test. Le seuil d'un run stressé est recalculé sur sa propre validation transformée ; les métriques qui en dépendent (macro-F1, exactitude) ne servent à aucune conclusion de sensibilité temporelle.

| modèle | variante | clip AUC | cluster AUC | Δ clip AUC (stress − T0) | IC95 | lecture | corrélation des probabilités avec T0 | Δp médiane [Q1, Q3] | \|Δp\| médiane |
|---|---|---|---|---|---|---|---|---|---|
| `c2b` | **T0** original | 0.847 | 0.915 | — | — | référence | 1.000 | — | — |
| `c2b` | T1 | 0.849 | 0.906 | +0.002 | [-0.011, +0.026] | inconclusive | 0.992 | +0.001 [-0.014, +0.017] | 0.015 |
| `c2b` | T2 | 0.751 | 0.785 | -0.097 | [-0.192, -0.031] | compatible with degradation under stress | 0.729 | +0.048 [-0.045, +0.196] | 0.125 |
| `c2b` | T3 | 0.726 | 0.697 | -0.121 | [-0.316, -0.059] | compatible with degradation under stress | 0.705 | +0.022 [-0.077, +0.144] | 0.107 |

> Δ clip AUC et Δp sont orientés **stress − T0**. Δ clip AUC et son IC95 sont appariés sur les mêmes tirages de clusters ; Δp est mesuré clip à clip sur le test, sans intervalle. Un Δ clip AUC négatif signifie que la discrimination baisse sous la transformation : les prédictions sont sensibles à cette perturbation. Cela n'établit pas que l'information détruite est physiquement pertinente.

🕐 **TSLM non évalué sous stress** : son checkpoint sérialisé n'est pas en notre possession. Les jeux et le protocole l'attendent.

## 8. Limites connues

1. **11 clusters *non-leak* en test, 8 en validation.** Tous les intervalles sont larges et se recouvrent. Aucune comparaison n'est concluante à cette taille.
2. **La normalisation d'amplitude ne neutralise pas le raccourci d'acquisition.** C1 (forme d'enveloppe seule, audio normalisé) fait au moins aussi bien que C0 (niveau absolu). Le barreau à dépasser est C1, pas C0.
3. **Les clusters ne sont pas des sessions d'acquisition démontrées** : la source ne publie ni site, ni conduite, ni horodatage. Ce sont des grappes de dépendance.
3bis. **Le niveau cluster est instable en validation, avec 8 clusters non-leak.** C2b y obtient 0,665 d'AUC cluster contre 0,915 en test, un écart de +0,250 sans commune mesure avec les autres échelons. Cause vérifiée : trois petits clusters non-leak (3, 3 et 4 clips) passent au-dessus de la médiane des clusters leak, et à 8 négatifs chacun pèse 12,5 % de la métrique. En test, aucun non-leak ne passe au-dessus. Le gros cluster de 80 clips est correctement classé dans les deux folds. C'est un artefact de petit échantillon, et c'est aussi ce sur quoi le seuil et l'hyperparamètre ont été choisis : à garder en tête devant tout écart val -> test.
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
