# Plan V2 - fiabilité et parité des scores

Icham — 2026-09-12

## Statut et objectif

**Extension ultérieure demandée par Icham :** [[Diagnostic causal TSLM vs C1#Plan global en dix étapes]] inclut désormais les corrections après diagnostic, la diversification conditionnelle, la confirmation et le produit décrit en section 7. Ce plan V2 reste une référence historique et technique, pas une campagne à relancer automatiquement. L'état courant est dans [[Diagnostic causal - exécution]] ; les étapes 6 à 10 du goal élargi sont à réaliser, pas déjà validées.

**13 septembre : enchaînement d'implémentation en pause.** Icham propose [[Diagnostic causal TSLM vs C1]] pour expliquer l'écart avant toute nouvelle recette. Le plan ci-dessous conserve la référence historiquement autorisée ; il ne doit plus déclencher automatiquement de nouveaux entraînements, refit final ou évaluation externe. [[V2 ML - exécution]] distingue résultats vérifiés et derniers fits dont les logs annoncent la fin, reçus à contrôler. Les portes de vérification, la préservation de V1 et l'interdiction de réglage sur le test consulté restent applicables. Le cap produit de la section7 n'est pas abandonné.

Trois problèmes distincts : reproductibilité numérique, qualité de détection, cohérence de restitution. Corriger le premier ou rendre le texte cohérent ne prouve pas une amélioration de classification. Référence immuable : [[Évaluation qualité V1 - exécution]], rapport/code sur `feat/icham-quality-eval` (`c8dcb9d`, miroir final `16a6df0`). Aucun écrasement du bundle V1, des validations de campagne ou de T0–T3.

## 0. Figer le cadre avant exécution

- Nouvelle branche et nouveaux dossiers pour les corrections/expériences, pas de modification des livraisons historiques.
- Conserver les 598 train et 208 validation comme développement. Les 194 test sont désormais consultés : référence historique, pas nouveau test indépendant ni critère de sélection V2.
- Préparer en parallèle, après feu vert, une réserve de données nouvelles : licence, signal/modalité, labels, sessions et dédoublonnage audités ; groupes séparés avant fenêtrage, adaptation du format figée avant scores. [[Datasets utiles pour PIPE]] contient des pistes, pas un dataset déjà validé/importable. Une source utilisée pour adapter le modèle ne peut pas être simultanément son holdout externe.
- Objectif relatif proposé : rejoindre puis dépasser C1 avec un compromis rappel/fausses alertes comparable, pas maximiser seulement une AUC groupée en ignorant les erreurs par clip. Le budget réel de fausses alertes par appareil-heure et le délai restent à convenir avec Icham ; aucune cible terrain n'est inventée.

## 1. Localiser l'écart, sans entraîner

Constat établi : mêmes poids, code/config et 208 IDs, mais 192 scores diffèrent de plus de `1e-6`, maximum absolu `0,0623688`. Chemins cache TimeF / WAV identifiés, cause exacte inconnue. Ne pas l'attribuer d'avance à BF16 ou à un arrondi.

1. Répéter le scoring d'une entrée strictement identique avec le checkpoint figé : même processus puis processus neuf, batch 1, `eval()`, versions/dtypes/kernels/modes de cache consignés. D'abord vérifier la répétabilité du modèle lui-même.
2. Tracer trois exemples de validation déjà documentés, dont le reload `c00c343da6afa` et l'écart maximal `ce0594e503562`. C'est une localisation de bug, pas une évaluation de qualité sur trois exemples.
3. Comparer dans l'ordre : octets WAV et mapping → waveform directe / relue de TimeF → normalisation → tableaux quatre bandes / cache historique → tenseurs, types et padding. Identifier le premier stade différent ; une proximité DSP ne garantit pas la proximité du score.
4. Injecter les deux tableaux dans **le même callable de scoring et le même modèle chargé**, puis le même tableau exact via les deux adaptateurs. Si nécessaire, poursuivre : collator → encodeur → projecteur → embeddings/masques → logits de classe → log-probabilités par token → sommes/softmax. Contrôler positions et terminaison sans changer les libellés pour améliorer un score.
5. Si les entrées sont identiques mais les scores non : isoler état numérique, batch/padding et sérialisation ; comparer base d'origine + mêmes poids temporels à bundle rechargé. Precision plus élevée uniquement comme expérience contrôlée ciblée, pas migration générale présumée nécessaire.

**Sortie attendue :** cause démontrée par une comparaison contrôlée, avec premier stade divergent et test minimal qui reproduit l'erreur. Si non localisée après cette passe, livrer les traces et hypothèses restantes ; ne pas lancer une recherche d'hyperparamètres pour contourner le défaut. Aucun recalcul d'AUC requis pour résoudre la parité.

## 2. Corriger la cause et démontrer la parité

- Réutiliser la transformation et le scoring existants comme chemin canonique commun à préparation, entraînement/validation, API et export. Le cache doit être une sérialisation vérifiée de cette représentation, pas une autre définition du signal. Choisir la référence selon le contrat du signal, jamais selon la meilleure AUC.
- Corriger seulement le stade fautif. Versionner dtype, preprocessing et runtime pertinents ; invalider un cache incompatible explicitement. Aucun arrondi, bruit ou relèvement opportuniste de tolérance pour faire passer le contrôle.
- Vérifier ensuite **les 208 validations**, un contrôle train fixé, l'ordre des clips et les tailles de lot utiles, puis un rechargement dans un processus neuf. Réutiliser le seuil de parité du reload : différence absolue de probabilité `<= 1e-6` dans le runtime de référence. Contrôler aussi entrées canoniques, finitude, répétabilité et éventuels changements de décision près du seuil ; ne pas confondre tolérance de score et preuve de décisions stables.
- Le test de régression doit échouer avant correction et réussir après. Si la tolérance n'est pas atteinte, documenter la cause et réexaminer le choix numérique avant de poursuivre, pas modifier le critère après coup.
- Toute correction changeant les scores servis reçoit une nouvelle version et ses propres artefacts. Si elle change la représentation effectivement apprise, décider explicitement du réentraînement nécessaire ; ne pas vendre un simple patch d'inférence comme un modèle réentraîné. Ne pas resélectionner e2/e4/e8 à la lumière du test déjà vu.

**Porte bloquante avant entraînement V2 :** cause corrigée, parité de bout en bout et reload vérifiés, aucun écart inexpliqué du type actuel. La parité ne promet aucune hausse d'AUC.

## 3. Rendre la sortie utilisateur cohérente

- Une seule décision exposée : score continu + seuil versionné. La génération libre ne décide plus séparément d'une classe concurrente ; conserver score brut non calibré et décision comme deux champs distincts.
- La bande présentée vient de la mesure DSP. Toute reformulation générée est contrôlée contre cette mesure ; en cas d'invalidité, un gabarit factuel de secours est **explicitement identifié**, et le texte brut / l'échec restent enregistrés pour l'audit.
- Garder séparés qualité du texte brut du modèle et cohérence du message finalement affiché. Un gabarit correct par construction ne transforme pas les 94,8 % de V1 en « 100 % d'intelligence du modèle ».
- Réutiliser les erreurs existantes pour signal absent/invalide, modèle indisponible et score non fini : jamais interpréter une erreur comme « pas de fuite ». Adapter le contrat API/documentation de version avec l'application lors de l'implémentation, pas déployer implicitement une interface incompatible.

**Critère logiciel :** zéro contradiction affichée classe/score et bande/mesure sur le lot audité ; erreurs explicites, audit brut conservé. Ce critère est indépendant du rappel et des fausses alertes.

## 4. Diagnostiquer le manque de discrimination sur développement

Utiliser les groupes du train pour les expériences d'apprentissage ; la validation reste réservée au réglage final du seuil après choix du candidat. Le diagnostic de parité sur ses entrées n'est pas une nouvelle sélection de modèle.

- Reprendre C1 comme comparateur, pas comme vérité terrain : ses descripteurs peuvent aussi exploiter des particularités d'acquisition.
- Tester une seule sonde simple sur **la représentation TimeNet réellement fournie au TSLM**, avec splits groupés. Le contrôle C2 existant utilise d'autres descripteurs ; il ne répond pas exactement à cette question.
- Examiner la supervision des tokens de classe versus les nombreux tokens descriptifs, les masques, les gradients des modules adaptés et la représentation des groupes/classes dans les lots. Le code V1 optimise la réponse complète ; une priorité insuffisante à la classe est une hypothèse, pas une cause démontrée.
- Distinguer : information perdue dans les quatre bandes ; information présente mais mal exploitée ; biais de groupe/acquisition. Une sonde faible ne prouve pas à elle seule l'absence d'information ; elle guide une expérience ciblée.

**Sortie attendue :** une faiblesse étayée sur développement, justifiant les variantes à comparer. Ne pas promettre qu'un Qwen plus gros ou davantage d'époques la résoudra.

## 5. Campagne V2 limitée et comparable

Proposition : trois variantes maximum pour la première campagne, Qwen 3.5-4B et H100 existants conservés.

| Variante proposée | Modification isolée | Hypothèse |
|---|---|---|
| A — référence corrigée | Chaîne canonique, recette d'apprentissage de référence | Reproductibilité établie, gain de qualité non présumé |
| B — priorité à la détection | Supervision dédiée/pondérée des classes, texte auxiliaire ; même représentation | La tâche descriptive domine trop l'apprentissage |
| C — représentation enrichie ciblée | Une seule variante prédéfinie, par exemple une information d'enveloppe justifiée par le diagnostic ; même recette de référence | Les quatre bandes perdent une information utile |

Ces variantes sont **proposées**, à figer après le diagnostic, avant leurs comparaisons. Ne pas cumuler B+C d'emblée. Si le diagnostic ne les justifie pas, les omettre plutôt qu'essayer au hasard. Une LoRA Qwen ou un changement de libellés/scoring remplace un candidat préannoncé, ou demande une campagne suivante : pas de quatrième essai caché. Une tête de classification séparée reste un comparateur diagnostique ; la choisir comme produit demanderait une décision explicite, sans attribuer son gain au TSLM.

Protocole proposé :

- Trois partitions groupées fixes à l'intérieur des **598 train**, sous réserve d'effectifs suffisants des deux classes. C1 et chaque candidat voient les mêmes partitions. Aucun fit de normalisation ni sélection sur les groupes réservés d'un fold.
- Réinitialiser les composants acoustiques pour chaque fold : **ne pas repartir du checkpoint V1 entraîné sur les 598 clips**, qui a déjà vu les groupes tenus à l'écart. La base Qwen préentraînée peut rester identique et gelée.
- Une seed et une durée/budget d'entraînement fixés par variante, sans choisir librement de multiples checkpoints. Trois variantes × trois folds font au plus neuf entraînements de comparaison, pas neuf configurations différentes ; consigner tous les runs, sondes, variantes de scoring et checkpoints effectivement consultés.
- Sélection préannoncée : moyenne des AUC groupées des folds, avec métriques clip, rappel et fausses alertes obligatoirement examinés. Ne pas mélanger sans précaution des scores bruts de modèles différents pour fabriquer une AUC hors-fold unique ; ne pas présenter ces résultats de sélection comme une preuve finale indépendante.
- Entraîner le seul candidat retenu sur les 598 train ; régler son seuil sur les 208 validations **pour ce modèle précis**, pas transférer silencieusement le seuil d'un modèle de fold. Fixer avant comparaison la règle et le compromis rappel/fausses alertes retenus, et appliquer la même procédure à C1.
- Ne pas modifier le seuil V1 pour améliorer rétroactivement son rapport. Une calibration éventuelle est une étape séparée, ajustée sur développement si réellement nécessaire ; elle ne résout ni la divergence d'entrées ni une mauvaise séparation. Sans validation de calibration, afficher un score brut, pas une probabilité terrain prétendument fiable.

**Porte de décision :** parité maintenue et progrès de développement documenté contre C1, sans dissimuler une régression par clip derrière la médiane par groupe. Si aucune variante ne convainc, arrêter cette campagne et expliquer la limite ; pas prolonger les essais jusqu'à obtenir un joli chiffre.

## 6. Nouvelle preuve finale, puis surveillance continue

Après choix : figer checkpoint, preprocessing, score, seuil, règles de restitution et versions ; rechargement neuf, nouvelle évaluation sur le holdout inédit audité, mêmes signaux/protocole pour C1. Publier comptes FN/FP, précision/rappel, AUC/AP, Brier si pertinent, groupes/incertitudes, texte brut et affiché, couverture/abstentions, latences. Réutiliser le harness et ses métriques ; adapter explicitement le contrat au nouveau manifeste, sans changer le v2/V1 historique ni créer un second moteur.

Stress uniquement après gel, sans réentraînement. Pour vérifier la robustesse de la décision déployée, prévoir cette fois une analyse **au seuil T0 fixé** ; les AUC restent sans seuil. Les seuils réajustés par stress de l'ancien rapport ne doivent pas être présentés comme équivalents.

Sans nouvelles données réservées, livrer honnêtement « V2 corrigée et étudiée sur développement », pas « fiabilité confirmée ». Ne pas rebaptiser un nouveau partage des anciens 194 clips « test vierge ».

La surveillance réelle reste une étape distincte : acquisitions continues annotées, règle d'ouverture/fin/persistance d'alerte fixée sur développement, puis rappel par événement, faux événements par appareil-heure et délai de détection. Un replay ou un lissage logiciel ne suffit pas à démontrer une meilleure détection terrain.

## 7. Objectif final — démontrer la valeur du TSLM

**Ajout explicitement demandé par Icham le 13 septembre 2026.** Pour l'appareil en écoute continue, le TSLM doit démontrer un intérêt au-delà d'une classe binaire. Un score et « bande dominante : 1–2 kHz » peuvent venir d'un classifieur léger, de mesures DSP et d'une phrase fixe. Notre sortie actuelle est proche de ce cas : elle ne prouve pas une valeur particulière de Qwen.

Cap produit à construire et tester :

- **Évolution d'un événement :** distinguer une activité persistante d'un bruit bref, comparer plusieurs périodes et expliquer ce qui a changé par rapport au fonctionnement habituel. Une durée ou une comparaison exige les fenêtres horodatées et la référence correspondantes ; un clip d'une seconde ne permet pas d'affirmer « persiste depuis deux minutes ».
- **Plusieurs signaux et du contexte :** acoustique, vibration, débit, état d'une pompe lorsque ces sources sont réellement disponibles, synchronisées et correctement supervisées. Aucune de ces modalités n'est considérée comme intégrée par le seul branchement de Qwen.
- **Investigation interactive :** répondre après alerte à « qu'est-ce qui distingue cet événement des précédents ? » et « quels éléments soutiennent l'alerte ou la rendent incertaine ? », avec des preuves consultables et sans inventer le contexte manquant.

Le benchmark final doit distinguer trois approches sur les mêmes événements réservés, périodes de référence, informations contextuelles disponibles et questions :

| Approche | Question à trancher |
|---|---|
| Classifieur + mesures DSP + phrases fixes | Le besoin est-il déjà couvert simplement ? |
| Classifieur + mesures/contextes transmis à Qwen | Le langage apporte-t-il une aide utile à l'investigation ? |
| TSLM recevant les représentations temporelles et le même contexte disponible | L'accès aux séries apporte-t-il davantage que les mesures transmises au langage ? |

Évaluer la fidélité aux mesures et à la chronologie, les comparaisons entre périodes, la gestion des incertitudes/informations absentes, l'utilité pour l'investigation, ainsi que coût/latence et qualité de détection. Figer tâches, critères et données réservées avant comparaison. Un texte plausible ou un meilleur score binaire ne suffisent pas à prouver ces bénéfices.

Si la deuxième approche fait aussi bien que la troisième, cela ne démontre pas l'intérêt spécifique du TSLM ; l'intérêt du langage lui-même n'est établi que si la deuxième apporte quelque chose face à la première. Si le classifieur + gabarit couvre le besoin aussi bien à moindre coût, reconnaître cette conclusion plutôt que présumer Qwen nécessaire.

**Articulation avec la V2 :** les étapes 0–6 restent la première brique autorisée, avec toutes leurs portes de vérification. La campagne A/C sur extraits n'est pas ce benchmark à trois approches ; ajouter neuf mesures textuelles à C ne valide ni compréhension d'événement, ni multimodalité, ni investigation interactive. La livraison V2 doit rendre explicite cet objectif final encore non démontré. Le continu annoté, le contexte/historique et les critères métier manquants restent à obtenir ; pas de collecte privée, nouvelle dépense ni déploiement implicites.

## Ordre et prochaine action

**Gel/protocole → cause de l'écart → correctif/parité → cohérence de restitution → diagnostic de détection → campagne bornée → nouveau holdout.** L'audit d'une source de confirmation peut avancer en parallèle après autorisation, ses résultats de performance restant fermés jusqu'au gel.

Prochaine action au moment de la planification : discuter/valider ce plan avec Icham, sans chantier technique. L'objectif d'implémentation a ensuite été explicitement lancé ; l'état actuel, ses preuves et la prochaine action sont désormais dans [[V2 ML - exécution]]. Aucun délai de réparation ni gain de qualité n'était garanti avant le diagnostic.

## Provenance de cette planification

Lecture des preuves à `c8dcb9d`, du code réel campagne/prétraitement/scoring et du vault source à `73274f0` après récupération des mises à jour. Skill `using-entire` puis recherche ciblée : retrouve le contrat `9135754`, aucune transcription expliquant la divergence. Les hypothèses causales sont donc issues du code et des artefacts, pas d'une intention enregistrée supposée. Deux relectures indépendantes du plan (parité et protocole V2), sans calcul, SSH ou édition technique.
