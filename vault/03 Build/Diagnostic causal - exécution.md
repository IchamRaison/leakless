# Diagnostic causal - exécution

Icham — 2026-09-13

## État courant

**Objectif d'implémentation désormais autorisé par Icham : [[Diagnostic causal TSLM vs C1]].** L'ancienne suite V2 de refit/confirmation reste suspendue ; l'autorisation porte sur le diagnostic et ses expériences contrôlées, pas sur une nouvelle recette produit. Premier jalon réalisé : les six fits existants sont complets et vérifiés, derniers artefacts publiés au commit code `b65042a`. Aucun réentraînement pour les récupérer, aucun score externe.

Le bloc D0 sans apprentissage est en cours d'implémentation. Son budget et ses contrastes sont fixés ci-dessous avant toute observation nouvelle. La préinscription machine, ses empreintes et les tests runtime restent à produire avant lancement. Code dans `/home/animus/ehl-hackathon-zurich-v2-recovery`, branche `feat/icham-v2-reliability` ; ne pas commiter dans l'ancien dépôt Git endommagé.

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

- [x] Six fits précédents vérifiés, récupérés et inventoriés sans réentraînement.
- [ ] Données/comparaison : mappings, groupes, support des sous-populations et particularités d'acquisition examinés ; limites publiées.
- [ ] D0 implémenté/testé/préinscrit puis exécuté : vrais scores/NLL train, alignement causal, contrôles de reload et interventions, états initial/terminal.
- [ ] Avant tout nouveau fit : journalisation classe/description/EOS, comptes exacts, gradients/clipping/mises à jour et résumés d'époque testés puis vérifiés réellement.
- [ ] Test de mémorisation32 à budget préinscrit, si nécessaire ; verdict limité à la capacité observée.
- [ ] Contrastes complémentaires choisis selon preuves : lecteur non linéaire, tête sans Qwen, neuf mesures par voie entraînable ; exécution ou omission justifiée explicitement, jamais marquée réussie sans preuve.
- [ ] Matrice finale par hypothèse : démontrée / non soutenue dans les conditions testées / inconnue ; preuve, alternatives et portée pour chaque verdict.
- [ ] Nouvelle recette proposée seulement après restitution causale, ou constat motivé des données manquantes. Aucun réglage sur test officiel/externe.
- [ ] Code, commandes, tests, artefacts et passation publiés ; revue indépendante des conclusions.

## Prochaine action

Terminer et relire l'observateur de loss/scoring et le runner D0, tester sur CPU puis dans le runtime H100 sans modifier les sources numériques historiques, produire/publier la préinscription machine et exécuter les observations A puis C. Ne pas démarrer de mini-entraînement avant le verdict du bloc sans apprentissage.
