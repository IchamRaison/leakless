# V1 — verdict de l'évaluation complète

Évaluation exécutée le 12 septembre 2026, sans changer le TSLM, le prompt, le
prétraitement ni les scores publiés. Modèle : `pipe-qwen3.5-4b-v1-1199789f-e4`.

## Verdict

**La V1 fonctionne techniquement, mais ces résultats ne justifient pas de la
présenter comme un détecteur fiable. Aucun gain face au contrôle simple C1
n'est démontré sur le critère principal par groupe.** À l'échelle des clips,
elle manque environ une fuite sur quatre et alerte sur près de la moitié des
non-fuites. Le texte décrit souvent correctement une bande fréquentielle,
mais ne fournit pas une explication causale et peut contredire le score.

Il s'agit d'une mesure sur ce dataset expérimental, pas d'une validation de
surveillance continue. Aucun objectif métier chiffré d'acceptation n'avait été
fixé : « insuffisant » est ici notre appréciation des erreurs observées et de
l'absence de gain démontré, pas le résultat d'un seuil d'acceptation préinscrit.

## 1. Détection sur tous les exemples test

Seuil T0 **`0.46893701143635425`**, choisi uniquement sur les 208 validations
(42 groupes), maximum de macro-F1 après médiane par groupe, égalité départagée
au plus petit seuil. Décision : `probability_leak >= seuil`. Le test contient
194 clips, 98 fuite et 96 non-fuite, regroupés en 41 groupes heuristiques
(30 fuite / 11 non-fuite). Aucun clip test n'a été exclu.

| Mesure test T0 | Par clip | Par groupe, médiane |
|---|---:|---:|
| TP / FP / TN / FN | 73 / 43 / 53 / 25 | 25 / 2 / 9 / 5 |
| Fuites manquées | **25/98 — 25,5 %** | 5/30 — 16,7 % |
| Fausses alertes parmi les non-fuites | **43/96 — 44,8 %** | 2/11 — 18,2 % |
| Précision fuite | 62,9 % | 92,6 % |
| Rappel fuite | 74,5 % | 83,3 % |
| F1 fuite | 68,2 % | 87,7 % |
| Macro-F1 | 0,646 | 0,799 |
| ROC-AUC, IC95 | 0,665 [0,577 ; 0,889] | **0,861 [0,706 ; 0,977]** |
| Précision moyenne (AP) | 0,637 | 0,940 |
| Brier brut, plus bas est meilleur | 0,241 | 0,242 |

Les scores sont non calibrés : ne pas afficher leur valeur comme une chance
de fuite validée. La meilleure AUC groupée ne « répare » pas les erreurs par
clip ; les deux calculs pondèrent différemment les exemples. AP groupée 0,940
ne signifie ni 94 % de rappel ni 94 % de décisions correctes.

Les intervalles disponibles proviennent de 2 000 rééchantillonnages de groupes
entiers, graine `20260912`, et couvrent ROC-AUC et macro-F1 seulement. Ils ne
couvrent ni l'incertitude d'entraînement ni celle de sélection du seuil.
Les groupes sont heuristiques, pas des sessions indépendantes démontrées.

Source de tous les comptes et métriques : [metrics.json](evidence/quality-v1/report/metrics.json).
Le [rapport généré](evidence/quality-v1/report/FINAL_EVALUATION.md) donne aussi
validation, IC macro-F1 et tous les contrôles. Les F1 de ses tableaux généraux
sont des macro-F1, à distinguer du F1 fuite ci-dessus.

## 2. Comparaison aux modèles simples

Les cinq contrôles existants ont été reproduits avec leurs descripteurs et
leur procédure de sélection fixés : scaler et ajustement sur 598 train,
grille officielle `C = 0.01, 0.1, 1, 10` sélectionnée sur validation groupée.
Cela n'ajoute pas de candidat TSLM : `n_configs_compared = 3` reste inchangé.
Les mêmes 194 test et 41 groupes sont évalués pour tous les runs.

| Modèle | AUC clip | AUC groupe |
|---|---:|---:|
| C0 — RMS absolu, signal brut | 0,878 | 0,839 |
| C1 — forme d'amplitude normalisée | **0,902** | **0,927** |
| C2 — spectre agrégé | 0,710 | 0,779 |
| C2b — descripteurs temporels peu profonds | 0,847 | 0,915 |
| C3 — mélange historique de descripteurs | 0,824 | 0,900 |
| TSLM V1 | 0,665 | 0,861 |

Le critère principal préinscrit est l'AUC groupée. Pour **T0 − C1**, l'écart
est **−0,067, IC95 [−0,180 ; +0,021]** : indécis, pas de gain démontré.
Par clip, l'écart secondaire est **−0,237 [−0,334 ; −0,001]**, compatible avec
une dégradation dans ce bootstrap. Aucun des cinq comparateurs ne donne une
supériorité TSLM établie par son intervalle apparié. Les huit comparaisons sont
publiées, sans retenir seulement celles favorables ; aucune correction pour
comparaisons multiples ni revendication de significativité n'est ajoutée.

Les fichiers des contrôles contiennent aussi leurs prédictions train pour
traçabilité, mais aucune performance train n'entre dans le verdict. C0 utilise
intentionnellement le niveau brut ; ne pas prétendre que tous les contrôles ont
exactement le même prétraitement.

La **Random Forest de Vincent n'est pas comparée** : sa branche
`feat/vincent-baseline-v1` à `fc837f7` documente un export 402 lignes compatible,
mais ses deux fichiers originaux ne sont pas accessibles dans les dépôts
inspectés. Une demande non bloquante a été faite ; aucune RF de remplacement
n'a été entraînée, aucune provenance reconstituée artificiellement.

Sources : [contrôles](evidence/quality-v1/controls/),
[comparaisons appariées](evidence/quality-v1/report/comparison.json).

## 3. Sensibilité temporelle

Les exports déjà publiés T1/T2/T3 ont été évalués tels quels : même checkpoint
et score, aucun réentraînement, aucune nouvelle transformation pendant cette
campagne. T2 utilise 250 échantillons par bloc, soit 31,25 ms à 8 kHz ; RNG
officiel dérivé de SHA-256, pas le `hash()` Python instable.

| Run | AUC clip | AUC groupe | Δ groupe T0 − stress, IC95 |
|---|---:|---:|---:|
| T0, original | 0,665 | 0,861 | — |
| T1, inversion temporelle | 0,652 | 0,824 | +0,036 [−0,027 ; +0,111] |
| T2, permutation de blocs | 0,612 | 0,700 | +0,161 [+0,037 ; +0,323] |
| T3, randomisation de phase | 0,541 | 0,536 | +0,324 [+0,162 ; +0,500] |

T2 et T3 dégradent la séparation, T1 reste indécis. Cela indique une
sensibilité aux transformations, **pas la preuve d'un bénéfice temporel par
rapport à C1**, d'une compréhension physique, ni d'une robustesse terrain.
Les transformations ne garantissent pas de conserver un signal physiquement
réaliste avec la même étiquette.

Chaque run possède son propre seuil sélectionné sur sa validation transformée.
Les F1/rappels stress du rapport ne sont donc **pas** des mesures au seuil T0
constant. Aucun nouvel audit exhaustif des invariants n'a été lancé ici ; la
provenance des transformations reste celle des exports gelés.

## 4. Texte et temps de réponse : les 402 sorties

Audit préinscrit : 208 validations puis 194 tests, IDs triés dans chaque fold.
Une chauffe fixe exclue des comptes. WAV original vérifié par MD5 avant DSP et
inférence ; aucun label, ID ou métadonnée d'acquisition transmis au modèle.
La classe générée est comparée à la décision du score T0 déjà exporté.

| Audit | Validation | Test |
|---|---:|---:|
| Exemples tentés / attendus | 208/208 | 194/194 |
| Format valide | 208/208 — 100 % | 194/194 — 100 % |
| Bande annoncée conforme à la mesure DSP | 202/208 — 97,1 % | **184/194 — 94,8 %** |
| Bande erronée | 6 | **10** |
| Désaccord classe générée / score seuillé | 27/208 — 13,0 % | **36/194 — 18,6 %** |
| Erreurs d'exécution / abstentions | 0 / 0 | 0 / 0 |
| Latence API texte après chauffe, médiane | 711 ms | **705 ms** |
| Latence API texte après chauffe, p95 | 751 ms | **819 ms** |

Les dix bandes test erronées contredisent aussi l'observation DSP fournie dans
le même payload, sans abstention. Premiers exemples dans l'ordre préinscrit,
non choisis pour leur apparence :

- Ligne 223, `c1534fe458cda` : texte « 1000–2000 Hz », mesure DSP/API « 0–1000 Hz ».
- Ligne 210, `c033723905116` : texte `leak`, score `0.4688219404568153`, inférieur
  au seuil `0.46893701143635425`, donc décision du score `no_leak`.

Ce dernier indicateur mesure une **incohérence entre deux sorties**, pas le
taux d'erreurs de classification contre le vrai label. Une génération libre
et un score de continuations seuillé sont des mécanismes distincts ; le modèle
et les règles n'ont pas été modifiés pour les faire concorder après observation.

La description est un gabarit de bande dominante parmi quatre, calculable
directement par DSP. Sa fidélité ne démontre pas l'utilité d'un LLM pour
expliquer une fuite ; aucune localisation, cause ou quantité perdue n'est
mesurée. La valeur de langage face à un simple gabarit DSP reste non démontrée.

Sur H100, chargement 9,23 s ; chauffe API 1,31 s, exclue. Latences séquentielles
du chemin `predict` texte, pas du score seul, du réseau, du streaming ni d'un
système concurrent. Pas de mémoire de pointe mesurée. Le GPU est revenu à
0 Mio / 0 % après le job ; l'instance n'a pas été arrêtée.

Le payload conserve un avertissement historique « V0 mécanique » et
`score_type=none` : aucune modification d'API n'a été faite pendant l'audit.
L'identité du modèle et les empreintes attestent V1 ; les probabilités viennent
du CSV T0 distinct, pas de ce champ vide ni d'un pourcentage généré.

Preuves : [402 lignes brutes](evidence/quality-v1/text-audit/raw.jsonl),
[compteurs et latences](evidence/quality-v1/text-audit/summary.json),
[provenance](evidence/quality-v1/text-audit/provenance.json).

## 5. Limite de parité campagne → livraison

La revue a trouvé une différence réelle entre les probabilités de validation
de la campagne de sélection et celles de l'export T0. Les deux AUC sont
correctes pour leurs probabilités respectives : **0,963235** en campagne,
**0,985294** dans l'export. Même checkpoint e4, mêmes 208 IDs / 42 groupes,
même médiane et même définition AUC ; ce n'est ni un effet de seuil ni l'arrondi
d'affichage. Les 208 scores diffèrent, dont 192 de plus de `1e-6` : écart absolu
médian `0,0000104996`, moyen `0,00942138`, maximal `0,0623688`.

Le chemin campagne lit le cache TimeF ; le chemin livré prétraite le WAV.
La cause numérique précise n'est **pas isolée**. Différences d'entrée ou
d'exécution en précision réduite restent des hypothèses, pas une explication
prouvée. La divergence existait avant ce nouveau harness ; le clip de reload
reproduit bien sa référence WAV, différente du score de campagne.

Les métriques ci-dessus évaluent bien les **exports livrés et figés** ; elles
ne doivent pas être présentées comme une preuve de parité exhaustive avec les
scores de sélection. Aucun changement de checkpoint ni resélection n'en a
découlé. Le [development.json original](evidence/quality-v1/development-original.json)
est conservé, SHA `104a24b2b733e12df89935c86a9ce5bb116d7fb58e1de045fc4b918a6863b130`.
Une future isolation de cette divergence doit rester sur développement et
ne pas servir à réécrire rétroactivement le résultat test.

## 6. Protocole, vérifications et reprise

Code exécuté et préinscription publiés **avant** les métriques réelles :
`c27a43fdd4de73ccc09f5f4bc02acd87c89f58f6`, branche `feat/icham-quality-eval`.
Le [protocole scellé](evidence/quality-v1/preregistration.json) fige modèle,
empreintes, population, comparateurs, seuil, audit et interdiction de tuning test.
Une correction minimale de l'AP en cas de scores égaux et la neutralisation des
conclusions prédéterminées du rapport ont été faites **avant évaluation** :
[provenance et commandes](HARNESS_QUALITY_INTEGRATION.md). Les autres formules
de métriques, agrégation et bootstrap sont inchangées. Ne pas confondre cette
AP corrigée avec une AP historique calculée par l'ancien code en cas d'ex aequo.

Vérifications exécutées :

- 32 tests passent dans le runtime cible, aucun ignoré :
  [7 harness](evidence/quality-v1/tests-eval.log),
  [25 TSLM/audit](evidence/quality-v1/tests-tslm.log).
- [Contrôles d'entrée](evidence/quality-v1/input-checks.json) : 1 000 WAV MD5 et
  format strict, empreintes des huit fichiers T0–T3 et couvertures 402 vérifiées.
- Revue numérique indépendante depuis CSV/split : 396 valeurs contre
  scikit-learn 1.7.2, seuils validation exacts, 68 intervalles reproduits par un
  bootstrap indépendant (36 IC test des runs, 32 IC appariés), huit comparaisons.
  Aucun écart au-delà de l'arrondi six décimales.
- Revue texte indépendante : toutes les couvertures, compteurs, latences et
  empreintes du résumé recalculés depuis les 402 lignes brutes, concordants.
- Les 16 fichiers report/contrôles/audit sont identiques par SHA256 entre
  H100 et copie locale : [empreintes](evidence/quality-v1/OUTPUT_SHA256SUMS).
  Poids et sources du modèle ainsi que les exports T0–T3 restent inchangés.

Runtime d'exécution : Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
scikit-learn 1.9.1, PyTorch 2.8.0+cu128, environnement existant
`/home/hicham/pipe-v0/.venv-repro`. Archive exécutée :
`/home/hicham/pipe-v0/code-quality-c27a43fd`. Sorties :
`/home/hicham/pipe-v0/quality-v1-001/{controls,report,text-audit}`.
Logs : [contrôles](evidence/quality-v1/controls.log),
[évaluation](evidence/quality-v1/evaluation.log),
[audit texte](evidence/quality-v1/text-audit.log).

Pour relire ou régénérer uniquement le rapport CPU, les commandes sont dans
le guide d'intégration ; utiliser les cinq contrôles conservés ci-dessus et les
quatre exports de `artifacts/tslm_runs/`, sans relancer d'entraînement. Employer
un dossier de sortie **neuf**, car les scripts officiels de rapport ne protègent
pas tous contre l'écrasement. L'audit texte est terminé : ne pas le relancer
pour consulter les résultats. Aucune dépendance GPU supplémentaire installée.

**Prochaine décision, non implémentée :** présenter honnêtement cette V1 comme
prototype et discuter d'une V2 avec un protocole de développement distinct.
Le test consulté n'est plus un test vierge. De nouvelles acquisitions réservées
et continues seront nécessaires pour vérifier généralisation, délai de
détection et fausses alertes par appareil-heure ; ces clips ne les mesurent pas.
