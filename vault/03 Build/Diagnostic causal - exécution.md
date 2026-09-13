# Diagnostic causal - exécution

Icham — 2026-09-13

## État courant

**Goal élargi par Icham jusqu'aux étapes 6 à 10 de [[Diagnostic causal TSLM vs C1#Plan global en dix étapes]].** Après le diagnostic : corrections et comparaison, diversification si nécessaire, mesure de son apport, gel/confirmation indépendante, puis intégration continue et benchmark à trois approches. Le diagnostic n'est plus le critère de fin global. Les portes de passage restent obligatoires ; aucune nouvelle recette ni réaffectation d'Aghashahi n'est adoptée d'avance. L'ancienne campagne reste une référence historique, pas une recette à relancer automatiquement. Six fits déjà vérifiés, derniers artefacts `b65042a` ; aucun réentraînement pour les récupérer ni score externe.

**D0 terminé, rapatrié et vérifié :1516 observations/3032 forwards, zéro apprentissage.** A reçu `4810270a…`, preuves `f64c943` ; C reçu `c6d2121f…`, preuves `9dd5b73`. Reçus et tous fichiers vérifiés localement/distamment avec `finished()`. Reload98 et échanges d'entrées complètes reproduisent exactement leurs scores ; poids inchangés dans chaque état. Revue indépendante des694 lignes A et822 lignes C : identités, cibles, masques et agrégats concordants. Ne pas relancer D0. Code `733b9c6`,181 tests runtime sans skip, préinscription SHA `cd2c3916cbac2b61b979f463fa793dec1be3585c6e558d7743a0d4c43a7e6d1f`. Checkout sain `/home/animus/ehl-hackathon-zurich-v2-recovery`, branche `feat/icham-v2-reliability` ; ancien Git endommagé à préserver.

**D1 terminé, rapatrié et vérifié : 3 588 forwards, aucun apprentissage.** A `640d29c` / reçu `f9402c5d…`, C `9c163d4` / reçu `00d8de16…`. Traversée Qwen réelle sur 598 clips par modèle, trois politiques ; entrées, états cachés et poids identiques entre voies, reproduction exacte de D0. Résultats et limites ci-dessous : résolution accrue, mais pas d'amélioration uniforme ni de correction suffisante de l'écart à C1. Code `306d330`, 203 tests runtime sans skip (152 TSLM + 51 évaluation), préinscription SHA `37cfcae18aa00cc402001d15f53f2df6a1802a9c0fd1c6ec13f7ca210e00ce05`, publiée à `1f89933` avant observation. Aucun gain final ni politique adopté, collecte conditionnelle toujours en dernier recours.

## Inventaire V2 complet — acquis de développement

Le contrôle existant `run_v2_campaign.finished()` a vérifié les six reçus et tous les fichiers distants, poids inclus, contre la préinscription SHA `5c5e0b4bbed6b973853e53b658350815b2d963547b68d0a08a9ade50c28b1359`. IDs fit/réservés et variantes concordent avec les trois folds préinscrits. Les trois nouveaux dossiers ont été rapatriés sans poids dans `docs/evidence/tslm-v2/campaign-c13fd47/`, puis leurs reçus, CSV et journaux revérifiés localement. Qwen identique avant/après chaque fit ; encodeur et projecteur changent pour les six fits.

| Variante | Fold0 AUC clip/groupe | Fold1 AUC clip/groupe | Fold2 AUC clip/groupe | Moyenne clip/groupe |
|---|---|---|---|---|
| A | 0,709589 /0,759615 | 0,562541 /0,735577 | 0,470545 /0,519231 | 0,580892 /0,671474 |
| C | 0,498630 /0,500000 | 0,762227 /0,807692 | 0,338366 /0,379808 | 0,533075 /0,562500 |
| C1, C=0,01 retenu | 0,883288 /0,961538 | 0,930188 /0,980769 | 0,860670 /0,903846 | 0,891382 /0,948718 |

Les moyennes A/C viennent de `choose_variant()` appliqué aux rapports complets existants, sans AUC regroupant artificiellement les scores de plusieurs modèles. A est premier de l'ancien classement, mais reste derrière C1 ; **aucun candidat n'est adopté pour un refit**. Aucun seuil diagnostique0,5 transformé en seuil de produit. La variabilité de C entre folds contredit l'affirmation d'un échec uniforme ; elle ne prouve pas l'usage utile du texte numérique ni une cause du retard.

Nouveaux reçus : A/fold2 `fac43ce7…`, C/fold1 `0020cfeb…`, C/fold2 `d0a8a284…`. Preuves antérieures A0/A1/C0 `83bcbe0` / `fa1b9b4`. Contrôle SSH du 13 septembre à 03:37 Paris : aucun processus de campagne actif, H100 0Mio /0%. Instance laissée allumée, aucun processus arrêté ou relancé.

## D0 — protocole fixé avant exécution

- **Checkpoints étudiés : A/fold0 et C/fold0**, pas un modèle final global ni le meilleur fold choisi après score. Même fold pour comparer les chemins ; les trois folds complets restent visibles dans l'inventaire.
- **Population : les 598 train officiels**, avec résultats obligatoirement séparés entre les 500 clips vus par chaque checkpoint et les98 réservés de ce fold. Les98 restent du développement déjà consulté. Comparateurs constants0,5 et fréquence de fuite estimée sur les500, même constante appliquée aux deux partitions.
- **32 témoins fixés avant score :** 16fuite/16non-fuite, un clip par groupe parmi les500 via `debug_rows`, ordre déterministe. Cela couvre16/16 groupes non-fuite mais16/52 groupes fuite : témoin équilibré, pas échantillon de prévalence.
- **État initial :** mêmes composants acoustiques fraîchement initialisés avec seed20260912, Qwen inchangé ; observer les32 témoins pour A puis C avant chargement des poids entraînés. Aucun optimiseur exécuté. Distinguer état initial/terminal dans les rapports.
- **État entraîné :** observer les598 pour chaque checkpoint via les vrais chemins `compute_loss` et score officiel. Séparer NLL classe complète, premier token discriminant, description, EOS, NLL binaire renormalisée et normes d'embeddings. NLL et classement ne sont pas interchangeables.
- **Interventions :** deux correspondances donneur→receveur déterministes sans points fixes sur les32 témoins : rotation au sein de chaque classe et appariement entre classes. A : échange des séries. C : séries seules, neuf mesures textuelles seules, échange conjoint. Même cible de réponse du receveur conservée pour la NLL, description incluse ; les cibles descriptives peuvent donc être en désaccord avec l'entrée perturbée. Rapporter les deltas, pas un gain de détection sur ces entrées artificielles.
- **Contrôles mécaniques :** score après échange intégral =score du donneur à `1e-6` près ; scores des98 après reload =CSV historique du même checkpoint à `1e-6` près. Un échec bloque l'interprétation et impose sa localisation, pas un relèvement de tolérance.
- **Budget total :** A=32+598+64=694 observations ; C=32+598+192=822 ; total1516. L'observateur prévu exécute deux forwards LLM par observation, soit3032 forwards, sans génération, fit, sélection de seuil ou recherche de scoring. Les appels réellement effectués seront comptés.
- **Gel :** batch1, mode eval, versions et sources liées à la préinscription, empreintes de poids avant/après chaque état ; garde locale des paramètres/buffers et restauration des hooks. Dossiers neufs, erreurs conservées, pas de résultat de remplacement ni relance automatique d'un dossier incomplet.

D0 mesure comparaison, cohérence des objectifs/scoring et sensibilité aux entrées. Il ne suffit pas à expliquer les gradients en début d'entraînement ni à démontrer une limite d'architecture. Les contrôles d'optimisation/mémorisation, tête BCE, sonde non linéaire et mesures C1 par voie entraînable restent conditionnés aux observations ; ils ne sont pas lancés en parallèle.

## Audit de complétion du nouvel objectif

Les contrôles diagnostiques ci-dessous couvrent seulement les étapes 1 à 5 ; leur achèvement ne suffit plus à terminer le goal élargi.

- [x] Six fits précédents vérifiés, récupérés et inventoriés sans réentraînement.
- [x] D0 observateur/runner testés, préinscrit puis exécuté ; résultats et revue indépendante consignés ci-dessous.
- [x] Données/comparaison : mappings/cibles et supports examinés en lecture seule ; limites de diversité et d'acquisition consignées ci-dessous. Aucune causalité ni exactitude physique des annotations démontrée par ces contrôles.
- [x] D0 : vrais scores/NLL train, alignement structurel, contrôles de reload et interventions, états initial/terminal. Les écarts numériques loss/scoring restent une piste ouverte.
- [x] D1 : trois politiques de tête aux mêmes checkpoints A0/C0, contrôles exacts, artefacts scellés et résultats ci-dessous ; pas de relance ni de sélection de politique.
- [ ] Avant tout nouveau fit : journalisation classe/description/EOS, comptes exacts, gradients/clipping/mises à jour et résumés d'époque testés puis vérifiés réellement.
- [ ] Test de mémorisation32 à budget préinscrit, si nécessaire ; verdict limité à la capacité observée.
- [ ] Contrastes complémentaires choisis selon preuves : lecteur non linéaire, tête sans Qwen, neuf mesures par voie entraînable ; exécution ou omission justifiée explicitement, jamais marquée réussie sans preuve.
- [ ] Matrice finale par hypothèse : démontrée / non soutenue dans les conditions testées / inconnue ; preuve, alternatives et portée pour chaque verdict.
- [ ] Nouvelle recette proposée seulement après restitution causale, ou constat motivé des données manquantes. Aucun réglage sur test officiel/externe.
- [ ] Code, commandes, tests, artefacts et passation publiés ; revue indépendante des conclusions.
- [ ] **6 — Corrections et nouvelle version :** tests de régression, campagne bornée et comparaison vérifiée sur développement, décision de progression explicite.
- [ ] **7 — Diversification conditionnelle :** si le progrès reste insuffisant, audit global, sources et rôles décidés, nouveau protocole séparé par acquisition ; sinon omission motivée.
- [ ] **8 — Apport des données :** même recette avec/sans ajout, réserves et budget comparables, rapport vérifié ; conditionnelle à 7.
- [ ] **9 — Confirmation indépendante :** modèle/chaîne gelés, reload neuf, réserve intacte, prédictions et rapport, aucun réglage après consultation.
- [ ] **10 — Produit et valeur :** intégration réelle, flux/événements/santé testés, replay distinct du terrain annoté, métriques événementielles et benchmark gabarit/Qwen/TSLM. Données ou preuves manquantes explicitement signalées.

Critères détaillés : [[Diagnostic causal TSLM vs C1#Critère de complétion du goal élargi]]. Les étapes 6 à 10 sont ajoutées au périmètre, pas exécutées par cette mise à jour.

## Prochaine action

**Exécuter D2 une fois puis vérifier ses trois fits**, selon le protocole fixé ci-dessous. Runner et 11 tests publiés `303609e` ; 214 tests runtime sans skip. Préinscription machine `67ed6bd5…` publiée avec les logs à `3505928`, avant tout fit réel D2. D1 ne résout pas le retard ; la sonde linéaire faible ne départage pas représentation et capacité de lecture. Réutiliser les métriques/agrégations et les comparateurs réservés existants, sans retuning. Aucun ajout de données ou nouvelle recette Qwen à ce stade.

D0 et D1 restent scellés ; aucun score à reproduire pour récupérer les résultats. D1 : handles A 22963 / C 5435 terminés avec code zéro, respectivement 312,744 s et 339,558 s. H100 après C : 0 Mio / 0 %, instance laissée allumée. Artefacts locaux `docs/evidence/tslm-v2/causal-d1-001/`, runtime `/home/hicham/pipe-v0/artifacts/causal-d1-306d330-001/`, logs `/home/hicham/pipe-v0/quality-causal-306d330/`.

### Repères pour le contrôle d'apprentissage conditionnel

Lecture de code uniquement, pas nouveau protocole adopté ni expérience exécutée. Si les résultats D2 justifient le contrôle de mémorisation :

- Reprendre les 32 IDs exacts de `diagnose_causal.cohorts(...)["subset_ids"]` du fold 0 : 16/16 classes et 32 groupes du fit. Ne pas refaire une sélection sur les 598 clips.
- Réutiliser `run_v2_campaign.initialize_training`, `examples(training=True)` et `train_epoch` avec initialisation neuve ; pas les checkpoints terminaux A0/C0. Avec batch effectif 8, 32 clips donnent quatre pas par époque ; 1 000 pas représenteraient 250 époques, budget/cadence/critères restant à préinscrire.
- `capture_training_diagnostics` s'utilise par époque, `pop_step` après chaque pas, puis `epoch_summary`. Fermer le logger avant les observations `compute_loss`, sinon elles contaminent les traces d'entraînement. Scorer aux frontières d'époques : `score_rows` laisse le modèle en mode évaluation.
- Garder la loss d'origine pour un premier témoin de capacité ; aucune pondération de classe supplémentaire choisie. Sauvegarde acoustique existante et reload dans un processus neuf à réutiliser. Éviter l'ancien overfit V0 `pipe.tslm.train.main`, qui lit la validation officielle, et ne pas modifier les anciens reçus de campagne.

## D0 — résultats et limites vérifiés

Mesures à checkpoints terminaux figés, pas pertes de batches pendant l'apprentissage. Les deux constantes de référence sont0,5 et0,442 (fréquence calculée seulement sur500fit).

| Modèle / population | NLL binaire officielle | AUC clip | AUC groupe | FN / FP au seuil diagnostique0,5 |
|---|---:|---:|---:|---:|
| A /500 fit |0,675040|0,639307|0,730769|221 /0|
| A /98 réservés |0,740530|0,709589|0,759615|73 /0|
| C /500 fit |0,577496|0,878461|0,730769|0 /154|
| C /98 réservés |0,573121|0,498630|0,500000|0 /21|

NLL du prédicteur constant0,442 :0,686404 surfit et0,756994 surles98 ; constante0,5 :ln2=0,693147 partout. C illustre qu'une NLL inférieure àln2 peut coexister avec un classement hors groupes proche du hasard lorsque la prévalence change. Son AUC train par clip ne vaut pas celle par groupe : les situations nombreuses pèsent davantage dans la première. A a un classement empirique modeste malgré toutes ses décisions négatives à0,5 ; ce seuil n'est pas réglé ni adopté en produit.

Sur les32 mêmes témoins, NLL binaire initiale→terminale : A3,244997→0,680683 ; C1,887097→0,683933. La description représente54,67% de la NLL initiale A et58,95% initiale C, contre7,53% et0,86% en fin. Elle n'est donc pas négligeable à tous les stades ; son effet sur les gradients reste inconnu. Chez A, la norme L2 moyenne du projecteur avant cast vaut17,661→18,473, contre0,746 pour les seuls tokens de réponse valides : pas une entrée projetée de norme minuscule, mais aucune cause d'échelle excessive démontrée.

Les interventions modifient les scores. A : deltas absolus moyens0,030676 (donneurs même classe),0,032511 (classes opposées). C : séries seules0,106197/0,126311 ; texte des mesures seul0,173343/0,207452 ; conjoint0,235073/0,327953. Revue des empreintes C : les échanges séries/texte ne modifient que leur voie respective ; conjoint et A séries reproduisent le donneur exactement. Sensibilité réelle aux deux entrées, pas preuve de compréhension des nombres ni d'utilité hors groupes. Une moyenne signée nulle d'une permutation bijective n'est pas une absence d'effet.

**Écart numérique distinct de la parité d'interface :** prompts loss/scoring exactement identiques, mais LP du token de classe différentes entre forward supervision1 et score2. Sur598 terminaux A, écart maximal par token0,140450 ; classe complète, moyenne absolue0,026105 surfit et0,025126 surles98. Pour C, maximum token0,223227 ; maximum absolu NLLclasse surfit0,223253. Ce n'est pas une différence de NLL binaire et aucun score alternatif n'est reconstruit depuis la seule LP vraie. Chez A, les premières marges de logits ont sept valeurs, multiples de0,125 ;366 clips à−0,25 et162 à−0,125. Les tokens suivants contribuent au différentiel pour moins de0,000722. Signature compatible avec BF16, cause à isoler. Le runtime `Qwen3_5ForCausalLM.forward` lu via `inspect` applique directement sa `nn.Linear` lm_head aux hidden states ; caster ensuite les logits arrondis enfloat32 ne récupère pas leur résolution.

### Matrice provisoire après D0 — pas verdict final

| Hypothèse | État après D0 | Limite / contrôle manquant |
|---|---|---|
| Tout est étiqueté fuite ou cibles décalées | Non soutenue par audit IDs/cibles/masques | Annotations physiques source non revérifiées |
| Aucun classement appris, même sur train | Non soutenue pour A0/C0 | Capacité à mémoriser parfaitement et autres folds non mesurées ici |
| Qwen ignore complètement séries / texte C | Non soutenue par interventions | Sensibilité ≠ exploitation utile ni compréhension temporelle |
| La description est toujours négligeable | Contredite à l'initialisation | Impact causal sur optimisation/gradients non isolé |
| Projecteur de norme trop petite | Non soutenue dans ces états A | Autre défaut d'échelle/gradient encore possible |
| Précision du score limite le classement | Piste renforcée par grille/écarts LP | D1 pour isoler la tête de sortie ; pas attribution BF16 acquise |
| Représentation insuffisante / données peu diverses | Limites et écarts de sondes observés | Sonde non linéaire et causalité du manque de diversité restent ouvertes |

## D1 — précision de tête, protocole fixé avant exécution

- Mêmes checkpoints terminaux A0/C0, mêmes598train et mêmes partitions500/98. Aucun nouveau fit, prompt, label de classe, seuil ou méthode de normalisation des continuations.
- Trois voies par clip : tête originale BF16 ; même couche linéaire recalculée avec hidden states et poids convertis enFP32, sortie FP32 ; même recalcul FP32 réarrondi enBF16. Ce dernier témoin distingue perte de résolution et différences arithmétiques du produit matriciel ; ne pas présupposer qu'il égale parfaitement le calcul BF16 original.
- Chaque voie passe par le score officiel des deux continuations complètes. Les deux politiques alternatives sont déclarées comme variantes numériques diagnostiques, pas exports conformes ni recette adoptée. Même état caché, mêmes entrées et mêmes poids à vérifier par empreintes ; seule la tête varie temporairement, restauration obligatoire.
- Budget :598×3=1794 forwards par modèle, total3588, sans génération ni backward. Baseline rechargée contreD0 sur598 à tolérance1e-6 ; échec arrête l'interprétation. Dossiers neufs, artefacts incomplets conservés, sources/runtime/checkpoints liés à la préinscription machine avant lancement.
- Rapporter résolution des marges, deltas de scores/NLL et AUC par clip/groupe séparément pourfit etréservés. Une grille plus fine ne prouve pas un meilleur classement ; un gain ne devient ni une preuve externe ni une explication de tout l'écart à C1. Aucune sélection du meilleur résultat ni recherche de seuil. Évaluer aussi les écarts du témoin réarrondi avant d'attribuer un effet au seul arrondi.

La journalisation des futurs fits est implémentée (`66f390b`) et testée : neuf tests runtime, dont équivalence exacte avec/sans observation des losses, gradients, mises à jour, moments et états aléatoires sur le vrai `train_epoch` et un petit décodeur différentiable. Preuves `docs/evidence/tslm-v2/training-observations-001/`, publiées `306d330`. Elle n'a pas encore accompagné un nouvel entraînement sur les vrais poids Qwen. Statistiques détachées sans forward supplémentaire, sommes/comptes par clip/époque, gradients avant/après clipping et mises à jour. La NLL binaire officielle reste une observation séparée à checkpoint figé, non reconstructible depuis le seul forward supervisé.

## D1 — résultats et portée

Commandes exécutées une fois par modèle dans des processus distincts, depuis `/home/hicham/pipe-v0/code-causal-306d330` : `PYTHONPATH=src /home/hicham/pipe-v0/.venv-repro/bin/python scripts/tslm/diagnose_head_precision.py observe --output /home/hicham/pipe-v0/artifacts/causal-d1-306d330-001 --variant A`, puis `--variant C`. Aucun pas d'optimiseur, génération, réglage de seuil, validation officielle, test ou réserve externe.

`run_v2_campaign.finished()` vérifie localement et sur le serveur les reçus et toutes leurs empreintes contre la préinscription `37cfcae1…`. A : `f9402c5d43ceb9f36b596050e27985338ade0c1a7528ba9502b20b4b4bda2cac` ; C : `00d8de1658cb4cec3914b0aef4d5969f8e5dc9c417563af9fe82b309b08ddd54`. Chaque reçu compte 598 clips et 1 794 forwards ; poids avant/après inchangés, inputs/masques/hidden states identiques entre voies, écart maximal du score BF16 avec D0 égal à zéro.

Revue indépendante CPU des 598 lignes de chaque modèle : reçus/préinscription/identités/D0 conformes, scores/NLL/agrégats et comparaisons reconstruits avec le harness, aucun blocage détecté. Les logits FP32 capturés avant réarrondi sont identiques entre les deux voies alternatives, et leur réarrondi BF16 est exact. Cette vérification porte sur les logits de décision capturés, pas sur tout le vocabulaire.

Mesures issues des `summary.json`, arrondies ici à six décimales ; mêmes partitions de développement, pas de moyenne mêlant fit et réservé :

| Modèle / population | Tête | AUC clip | AUC groupe | NLL binaire |
|---|---|---:|---:|---:|
| A / 500 fit | BF16 originale | 0,639307 | 0,730769 | 0,675040 |
| A / 500 fit | FP32 | 0,688723 | 0,774038 | 0,674231 |
| A / 500 fit | FP32 réarrondie BF16 | 0,640101 | 0,730769 | 0,675040 |
| A / 98 réservés | BF16 originale | 0,709589 | 0,759615 | 0,740530 |
| A / 98 réservés | FP32 | 0,722740 | 0,745192 | 0,742207 |
| A / 98 réservés | FP32 réarrondie BF16 | 0,708493 | 0,759615 | 0,740531 |
| C / 500 fit | BF16 originale | 0,878461 | 0,730769 | 0,577496 |
| C / 500 fit | FP32 | 0,879109 | 0,731971 | 0,578160 |
| C / 500 fit | FP32 réarrondie BF16 | 0,878461 | 0,730769 | 0,577412 |
| C / 98 réservés | BF16 originale | 0,498630 | 0,500000 | 0,573121 |
| C / 98 réservés | FP32 | 0,504658 | 0,509615 | 0,576711 |
| C / 98 réservés | FP32 réarrondie BF16 | 0,498630 | 0,500000 | 0,573121 |

Les premières marges distinctes BF16→FP32 passent de 7→471 (A fit), 4→98 (A réservé), 41→473 (C fit) et 22→98 (C réservé). **La tête numérique change réellement le scoring à poids fixes ; supprimer cette grille ne suffit pas à rendre le modèle bon.** A gagne en AUC clip fit, mais son AUC groupe réservée et sa NLL se dégradent ; C reste presque au hasard sur les groupes réservés. Aucun FP32 promu automatiquement, aucun gain de généralisation ou de calibration établi.

Le témoin FP32 réarrondi n'est pas exactement la tête BF16 originale : A conserve toutes les premières marges, mais 22 scores complets changent, maximum 0,000020085 ; C change 41 scores, maximum 0,027828214 sur fit, avec un écart maximal de première marge de 0,125. Ses AUC restent identiques à l'originale. Les deux calculs matriciels peuvent différer près d'une frontière d'arrondi ; ne pas attribuer tout le contraste original→FP32 au seul cast final. FP32 versus son propre réarrondi isole la résolution de sortie de ce calcul FP32. Cela ne teste ni tous les calculs internes de Qwen ni l'effet de la précision pendant l'entraînement, et ne résout pas directement la divergence loss/scoring entre contextes de forward.

Le seul changement de première marge C concerne le clip fit `c51eb51347f21` : −0,625→−0,75. Logit `no` original 25,75 ; recalcul FP32 25,8125534, juste au-dessus de la frontière 25,8125, donc réarrondi 25,875. Exemple vérifié de la différence entre produits matriciels, pas anomalie à masquer ni justification pour relancer D1.

**Verdict causal limité :** effet numérique du scoring démontré ; hypothèse « la seule tête FP32 corrige le retard » non soutenue pour ces deux checkpoints/fold0. Optimisation, supervision et capacité de lecture/représentation restent à départager. Prochaine étape choisie : sonde non linéaire bornée sur la représentation exacte, pas collecte immédiate ni nouvelle grande campagne.

## D2 — sonde non linéaire, protocole avant fit

**Décision après D1 ; implémentation/test/préinscription terminés, pas encore de résultat D2.** Vérifier si une autre capacité de lecture exploite mieux les mêmes entrées que la sonde linéaire. C'est un contrôle séparé du TSLM : seuls ses trois petits fits utilisent le CPU de la machine existante ; les apprentissages du TSLM restent sur H100. Aucun basculement de Qwen sur CPU, aucune bibliothèque/GPU à installer ou provisionner.

- **Entrée exacte :** les 598 train / 102 groupes, quatre séries TimeNet de 64 valeurs aplaties en 256, padding inclus. Cache courant `prepared-v2-ac-04d53b6`, SHA train `9b7b14e1…` ; vérifier son égalité des séries avec l'ancien cache `prepared-v2-aaab4af` (`8bf27c0a…`) utilisé par les sondes. Les NPZ enrichis diffèrent, pas nécessairement leurs séries. Aucun descripteur C1, label ou métadonnée ajouté aux features.
- **Partitions :** réutiliser strictement `folds.json`, SHA `5c2bea733f8f2a2dc525b9738a5aa40ae3ce220cf6d76d712cab984afb4d5efb`. Trois fits ne voyant chacun que le train du fold ; chaque clip est réservé une fois, groupes disjoints. Aucun split automatique d'early stopping.
- **Une famille, une configuration :** `HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.05, max_iter=200, max_leaf_nodes=7, min_samples_leaf=10, l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=20260913, class_weight=None, categorical_features=None)`. Tous les paramètres effectifs `get_params()` seront liés à la préinscription ; autres paramètres sklearn inchangés. Binning appris dans chaque fit seulement, pas de scaler externe. Pondération par clip comme les références, pas par groupe.
- **Budget :** exactement trois appels de fit réels, 200 itérations de boosting chacun, un thread via `threadpoolctl`, aucun réglage intermédiaire ni prolongation automatique. Tests sur données synthétiques distincts. Préinscrire source, environnement/interpréteur, paramètres, caches, gate, manifeste, folds et comparateurs avant tout score D2.
- **Runtime vérifié sans fit :** `/home/hicham/pipe-v0/.venv-repro/bin/python`, Python 3.12.3 ; NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.6.0. Le Python local sans sklearn n'est pas un environnement de remplacement.
- **Comparateurs sans nouveaux fits :** `probes.json` SHA `aa16a7b1230ed0badbaac8db942b721ae07a3d213903645025bc2d80af456fc8`, `TimeNet256` linéaire et `C1_fixed` à `C=1`. Vérifier couverture/affectation aux folds et recalculer leurs agrégats via le harness. Le C1 retuné à `C=0,01` reste une autre référence historique.
- **Mesures :** conserver prédictions et métriques fit/réservés séparées pour D2 ; comparaisons appariées aux références **uniquement sur les réservés**, car leurs scores fit ne sont pas disponibles. AUC groupe réservée primaire, AUC clip et NLL secondaires ; moyenne des trois AUC, jamais AUC des scores de trois modèles regroupés. Seuil 0,5 exclusivement diagnostique, sans sélection ni export produit.
- **Critère descriptif avant score :** trois deltas d'AUC groupe strictement positifs face à la sonde linéaire = amélioration cohérente sur ces trois folds ; sinon résultat hétérogène ou non amélioré, avec tous les deltas publiés. Comparaison à C1 distincte. Ce n'est ni un test de significativité ni trois confirmations indépendantes. Aucun modèle retenu automatiquement ; la persistance de l'écart à C1 et le comportement fit orienteront le prochain contrôle.
- **Limites :** sept feuilles, minimum dix observations et régularisation limitent cette sonde. Un fit faible confond information disponible et capacité/optimisation de ce lecteur ; un résultat négatif ne démontre aucun plafond non linéaire absolu. Un résultat positif ne valide ni l'exploitation par Qwen, ni la compréhension temporelle, ni une nouvelle recette TSLM. Les gros groupes gardent plus de poids à l'apprentissage, même si l'AUC groupe réduit leur poids à l'évaluation.

Revue méthodologique indépendante favorable sous ces limites. Pas de validation officielle, test ou réserve externe consultés ; D0/D1 ne sont pas relancés. Reçus scellés et dossiers neufs, échec conservé sans remplacement silencieux. Root seul responsable de l'exécution distante et des publications ; agent propriétaire des deux nouveaux fichiers, helpers partagés conservés.

### D2 — vérifications avant fit

Code `303609edd5e48367390ebe332c92cb777da920b9`, snapshot `/home/hicham/pipe-v0/code-causal-d2-303609e`. Revue statique indépendante sans blocage ; ajout des moyennes absolues des trois AUC, séparées fit/réservés, sans AUC regroupée. Syntaxe locale validée ; environnement local sans TimeNet/sklearn non utilisé comme preuve de validation. Runtime réel : 11 tests ciblés passent, puis 163 tests TSLM et 51 évaluation sans skip. Les fits de tests sont synthétiques, pas les trois fits D2.

Préinscription créée à 06:58:46 Paris, SHA-256 `67ed6bd5a70caa7201d961fb692b0a0aa7eaccea53e3d41ee157fd0bc0ae8977`, publiée avec les logs à `3505928` avant fit. Comparateurs recalculés à l'identique, séries anciennes/courantes identiques, 16 sources revérifiées contre le snapshot ; paramètres complets/interpréteur/bibliothèques liés au JSON. Dossier distant `/home/hicham/pipe-v0/artifacts/causal-d2-303609e-001`, copie locale `docs/evidence/tslm-v2/causal-d2-001/`. Dossier `run/` encore absent au contrôle préalable.

Commande d'exécution prévue, depuis le snapshot : `PYTHONPATH=src /home/hicham/pipe-v0/.venv-repro/bin/python scripts/tslm/diagnose_nonlinear.py run --output /home/hicham/pipe-v0/artifacts/causal-d2-303609e-001`. Ne pas recréer une préinscription ni relancer silencieusement un dossier incomplet. Les chemins et la commande `preregister` sont reconstructibles depuis les champs `paths` du JSON ; aucune base Qwen chargée.

## Audit des populations et des cibles

Comptages de développement uniquement, revérifiés contre `manifests/split_v2.csv`, `split_v2_audit.csv`, `docs/evidence/tslm-v2/train-diagnostic-001/folds.json` et les listes `training_ids`/`heldout_ids` des six reçus `campaign-c13fd47/{A,C}/fold-{0,1,2}/complete.json`, code `06d6540`. Même couverture A/C, aucun groupe partagé fit/réservé. Pas de nouvelle inférence ni lecture de WAV pour ce recomptage.

| Population | Fuite | Tuyau sans fuite | Bruit environnemental | Total |
|---|---:|---:|---:|---:|
| Ensemble train disponible |294|242|62|598|
| Entraînement fold0 |221|240|39|500|
| Entraînement fold1 |205|128|40|373|
| Entraînement fold2 |162|116|45|323|
| Réservé fold0 |73|2|23|98|

Les entraînements ont respectivement44,2%,54,96% et50,15% de fuites : aucun n'est « tout fuite ». La classe binaire négative regroupe vrais sans-fuite et bruit. Les242 vrais sans-fuite représentent7 groupes, contre78 groupes fuite et17 bruit ; deux groupes sans-fuite totalisent197 clips (104+93). Dans le réservé fold0, les deux vrais sans-fuite n'ont qu'un groupe ; aucun hydrophone négatif, et aucune cellule appareil×matériau×région ne contient à la fois fuite et vrai sans-fuite. Pression/débit renseignés uniquement côté fuite dans les métadonnées train, sans être transmis au modèle : asymétrie de protocole observée, raccourci acoustique non démontré. Groupes heuristiques, pas sessions instrumentées indépendantes.

Audit de code ciblé : `split_loader.py:119` teste exactement `label == "leak"`, `diagnose_train.py:73` conserve0/1, `run_v2_campaign.py:322` transmet la classe à `target_text` (`preprocessing.py:100`). Aucun filtre « tout fuite » trouvé ; bruit reste `no_leak`. Signal et cible sont joints par le même clip_id, avec unicité/couverture du cache et vérifications WAV/cache dans le chargeur. Cela établit la fidélité au manifeste, pas la vérité physique de chaque annotation. Le manque de diversité est une piste, pas une explication causale acquise du plateau ni de tout l'écart à C1.

## Relecture du prétraitement — preuve de code, pas résultat expérimental

Les deux chemins retirent moyenne et RMS global : C1 n'a pas ici un accès exclusif au gain brut. `preprocessing.band_series` conserve61 fenêtres Hann de32ms, espacées de16ms, puis trois pas nuls ; la somme des quatre bandes avant `log1p` conserve une énergie locale. L'affirmation « toute l'enveloppe est effacée » est donc trop forte. La transformation est destructive (phase/détail spectral perdus ; x et−x donnent les mêmes bandes alors que la skewness C1 change de signe), sans que cela démontre un plafond prédictif fuite/non-fuite.

C1 expose directement huit statistiques globales non linéaires et la dispersion de20 RMS de50ms ; la sonde linéaire fixe sur256 valeurs doit exploiter une autre représentation. Son score inférieur ne démontre pas même le meilleur plafond linéaire atteignable. Enfin, C emploie les mêmes définitions C1 mais après passage float32/renormalisation et texte à six chiffres significatifs : identité de définition, pas bit à bit avec la baseline WAV float64. Aucun effet de ces arrondis sur la qualité n'est prouvé. Sources : `scripts/eval/harness/features.py:64`, `scripts/timenet/leakless_acoustic/connector.py:67`, `scripts/eval/harness/split_loader.py:151`, `src/pipe/tslm/preprocessing.py:50`, `:114`, `:135`, `scripts/tslm/diagnose_train.py:363`, code `733b9c6`.

Revue indépendante de l'initialisation : même seed, ordre RNG et constructeur que les fits ; hashes réels encore à confirmer au lancement. D0 utilise le mode eval pour toutes ses observations : ce sont des NLL à poids fixes, pas une reproduction exacte des pertes en ligne avec encodeur/projecteur en mode train.
