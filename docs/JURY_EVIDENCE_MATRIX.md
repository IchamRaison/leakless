# JURY_EVIDENCE_MATRIX — chaque affirmation du pitch pointe vers un artefact

> Règle : **rien ne se dit devant le jury qui n'ait une ligne ici.** Si une phrase du
> pitch n'a pas d'artefact en face, elle ne se dit pas.
>
> Split gelé `manifests/split_v2.csv`, `sha256 7a8716a352844342…`.
> Test : 194 clips, **41 clusters de dépendance** (30 *leak* / 11 *non-leak*).

---

## ✅ Démontré sur ce jeu de données

| CLAIM | EVIDENCE | STATUS | LIMITATION | DEMO ARTIFACT |
|---|---|---|---|---|
| Le niveau sonore absolu est un raccourci | C0, contrôle RMS sur signal brut, évalué en held-out : clip AUC **0,878**, cluster AUC 0,839 | **démontré** | chaîne d'acquisition non calibrée ; rien ne dit que l'écart survit à un autre matériel | `artifacts/final_evaluation/FINAL_EVALUATION.md` §3 |
| **La normalisation d'amplitude ne suffit pas à retirer le raccourci d'acquisition** | C1 (forme d'enveloppe seule, audio normalisé, aucune fréquence) : clip AUC **0,902**, cluster AUC **0,927** — **au-dessus de C0** | **démontré** | 41 clusters ; IC larges | `FINAL_EVALUATION.md` §3, `run_controls.py --controls C1` |
| L'information spectrale agrégée est la plus faible des trois | C2 (bandes, centroïde, étalement, platitude ; sans phase ni ordre) : clip AUC **0,710** | **mesuré** | un vecteur spectral plus riche ferait peut-être mieux ; C2 est délibérément simple | `FINAL_EVALUATION.md` §3 |
| C1 fait mieux que C2 | bootstrap apparié sur les mêmes tirages de clusters : Δ clip AUC **+0,192**, IC95 [+0,094, +0,284] | **compatible with improvement** | jamais « significatif » : 41 clusters | `comparison.json`, clé `c1_vs_c2` |
| **C2b est réellement temporel, pas du spectre déguisé** | même modèle, sans réentraînement : corrélation des probabilités avec T0 de **0,729** sous permutation de blocs et **0,705** sous phase randomisée ; dégradations d'AUC **−0,097** (T2, IC95 [+0,031, +0,192]) et **−0,121** (T3, IC95 [+0,059, +0,316]) en apparié | **démontré** | T1 (inversion) ne le bouge presque pas — prédit avant mesure, c'est un stress faible pour ces descripteurs | `docs/C2B_HYPOTHESIS.md` §Stress |
| Les transformations de stress sont reproductibles entre processus | `clip_rng` dérivait sa graine de `hash()`, salé par processus : 4 processus donnaient 4 graines. Corrigé en `b23601a` par une dérivation SHA-256 canonique, avec 11 tests dont 4 en sous-processus réels | **démontré, après correction** | les résultats T2/T3 antérieurs à `b23601a` sont SUPERSEDED ; T0 et T1 inchangés au bit près | `git show b23601a` |
| Les définitions de C2b ont été figées avant tout résultat | commit `355f074` contient les six descripteurs, les bandes, les fenêtres et 10 tests unitaires, **et aucun chiffre** ; l'évaluation est dans un commit postérieur | **démontré par l'historique Git** | — | `git show --stat 355f074` |
| Une structure temporelle simple ne surpasse pas le raccourci d'enveloppe | C2b − C1 : Δ clip AUC −0,055, IC95 [−0,164, +0,058] | **inconclusive** | toutes les estimations ponctuelles favorisent C1, mais 41 clusters ne permettent pas de trancher | `C2B_HYPOTHESIS.md` §Réponse |
| La baseline historique n'apportait rien de plus qu'une représentation temporelle à 6 descripteurs | C2b − C3 : Δ clip AUC +0,023, IC95 [−0,154, +0,123] | **inconclusive**, les deux sont indiscernables | — | `comparison.json` |
| Le split est sans fuite sur cinq invariants | `verify_split_invariants.py` : I1-I5 à 0 violation, reconstruits depuis l'audio, `group_id` ignoré | **démontré** | I4 couvre 43 % des clips, I5 39 % — les champs n'existent pas ailleurs | `docs/SPLIT_V2_AUDIT.md` §6, T1 |
| Le vérificateur n'est pas complaisant | il **échoue** sur `split_v1` (exit 1, 3 invariants violés) et passe sur `split_v2` | **démontré** | — | `SPLIT_V2_AUDIT.md` §6, T1 |
| Le pipeline d'évaluation rend le hasard quand la relation étiquette/signal est détruite | contrôle négatif à étiquettes permutées par cluster : clip AUC **0,475** et **0,481**, IC95 contenant 0,5 | **démontré** | c'est un test de sanité, pas une expérience de performance | `run_controls.py --shuffle-labels` |
| Les métriques sont group-aware | clip-level **et** cluster-level partout, effectif de clusters avec chaque chiffre, bootstrap sur clusters | **démontré** | — | `EVAL_PROTOCOL.md` §7ter, `harness/metrics.py` |
| Les comparaisons de modèles sont appariées | un seul tirage de clusters sert aux deux modèles ; test : comparer un modèle à lui-même donne Δ = 0 et IC nul | **démontré** | — | `tests/run_tests.py::paired_bootstrap_is_paired_and_zero_for_identical` |
| Le jeu TimeF est valide et groupé | 1000 records relus depuis le disque, 185 clusters en `subject_ids`, 500/500 | **démontré** | jeu local, non publié dans un registre | `scripts/timenet/`, `PIPELINE_RESULTS.md` §3 |
| Les transformations de stress préservent ce qu'elles annoncent | T1 et T3 : `\|FFT\|` préservé à 3,5e-16 ; T2 (blocs de 250 échantillons) : histogramme d'amplitude identique, `\|FFT\|` détruit (déviation médiane 0,70) ; 0 violation de mapping | **démontré** | ce ne sont pas des augmentations préservant l'étiquette | `eval-out/stress_invariants.json`, `FINAL_EVALUATION.md` §7 |
| Aucune métadonnée d'acquisition n'atteint un modèle | contrat : 19 colonnes interdites ; `split_v2_audit.csv` n'est ouvert que pour `path` ; sentinelle testée | **démontré** | la résolution `clip_id -> chemin` reste une règle, pas une impossibilité physique | `docs/MODEL_EVAL_CONTRACT.md` §2, `tests` S8 |

---

## 🕐 En attente — mesurable dès que le TSLM arrive

| CLAIM | EVIDENCE | STATUS | LIMITATION | DEMO ARTIFACT |
|---|---|---|---|---|
| **La modélisation temporelle apporte quelque chose** | comparaison appariée TSLM − C1, et TSLM − C2b | **PAS ENCORE DÉMONTRÉ** | le barreau est C1 = 0,902 / 0,927. Battre C2b (0,847 / 0,915) ne suffira pas : C2b ne bat pas C1 non plus | commande §Intégration |
| Le TSLM est sensible à l'organisation temporelle | écart de score entre T0 et T1/T2/T3 sur le même checkpoint | **PAS ENCORE DÉMONTRÉ** — jeux prêts, checkpoint non possédé | un écart nul est une réponse valide, et ce serait un résultat | `timef-stress/T{0,1,2,3}` |
| Le TSLM dépasse une baseline sur held-out | `build_final_report.py` avec le run de Hicham | **PAS ENCORE DÉMONTRÉ** | 41 clusters : l'issue la plus probable est *inconclusive* | `FINAL_EVALUATION.md` §6 |

---

## 🚫 Interdit — ne se dit pas, quel que soit le résultat

| CLAIM | POURQUOI C'EST INTERDIT |
|---|---|
| « LeakLess détecte des fuites sur le terrain » | aucune donnée client, aucune validation terrain, **aucun matériel testé**. Le jeu vient d'un site d'entraînement expérimental à Dongguan. |
| « Le modèle localise la fuite » | le jeu ne supporte aucune localisation |
| « L'écart est statistiquement significatif » | 41 clusters indépendants en test, 11 côté *non-leak*. Les verdicts autorisés sont *compatible with improvement* / *inconclusive* / *compatible with degradation*. |
| « Notre approche est algorithmiquement nouvelle » | la détection acoustique de fuite par ML existe. La contribution est la **discipline d'évaluation** et l'échelle de contrôles. |
| Un score sans son nombre de clusters | un chiffre *non-leak* sans son dénominateur est ininterprétable |
| Un intervalle de confiance bootstrapé sur les clips | les clips d'un cluster ne sont pas indépendants |
| Toute utilisation de `split_v1` | invalide : 30 conditions physiques traversaient les folds |
| « Le débit est de X L/min » | les noms de fichiers donnent des m/s, une vitesse. Pas de section, pas de débit. |
| « Les étiquettes sont vérifiées » | les distributions de signal présentent des différences mesurables **associées aux étiquettes fournies**. La provenance des étiquettes reste celle du dataset source. |

---

## L'histoire que ces artefacts racontent

1. On a pris un dataset public et on a trouvé que son découpage naïf fuyait. On l'a
   prouvé, on a refait le découpage, et notre vérificateur **échoue** sur l'ancien.
2. On a mesuré qu'un volumètre atteint 0,878 d'AUC. Puis on a normalisé l'amplitude
   pour retirer ce raccourci, et **découvert que ça ne suffit pas** : la forme
   d'enveloppe seule fait mieux (0,902). Et 53 % des clips *leak* ont leur pic dans
   les 5 % supérieurs de la pleine échelle sans jamais saturer, contre 9 % des
   non-leak : c'est un réglage de gain, pas une physique de fuite.
3. **C'est pour ça que l'échelle de contrôles existe.** Sans C1, on aurait attribué à
   la « signature acoustique » ce qui vient de la chaîne d'acquisition.
4. On a construit un contrôle temporel peu profond (C2b) et **figé ses définitions
   dans un commit sans résultats**, avant de l'évaluer. Il est réellement temporel —
   les stress tests le désorganisent — mais il **ne dépasse pas C1**. La question
   temporelle reste donc entière, avec le bon barreau et des jeux de stress dont les
   invariants sont mesurés. **Elle attend le TSLM.**

> Le résultat que nous pouvons défendre aujourd'hui n'est pas un score. C'est un
> protocole qui empêche de se tromper — et qui nous a déjà attrapés deux fois.
