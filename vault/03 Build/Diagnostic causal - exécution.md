# Diagnostic causal - exécution

Icham — 2026-09-13

## État courant

**Objectif d'implémentation désormais autorisé par Icham : [[Diagnostic causal TSLM vs C1]].** L'ancienne suite V2 de refit/confirmation reste suspendue ; l'autorisation porte sur le diagnostic et ses expériences contrôlées, pas sur une nouvelle recette produit. Premier jalon réalisé : les six fits existants sont complets et vérifiés, derniers artefacts publiés au commit code `b65042a`. Aucun réentraînement pour les récupérer, aucun score externe.

**D0 terminé, rapatrié et vérifié :1516 observations/3032 forwards, zéro apprentissage.** A reçu `4810270a…`, preuves `f64c943` ; C reçu `c6d2121f…`, preuves `9dd5b73`. Reçus et tous fichiers vérifiés localement/distamment avec `finished()`. Reload98 et échanges d'entrées complètes reproduisent exactement leurs scores ; poids inchangés dans chaque état. Revue indépendante des694 lignes A et822 lignes C : identités, cibles, masques et agrégats concordants. Ne pas relancer D0. Code `733b9c6`,181 tests runtime sans skip, préinscription SHA `cd2c3916cbac2b61b979f463fa793dec1be3585c6e558d7743a0d4c43a7e6d1f`. Checkout sain `/home/animus/ehl-hackathon-zurich-v2-recovery`, branche `feat/icham-v2-reliability` ; ancien Git endommagé à préserver.

**Prochain contraste : D1, précision de la tête de sortie, sans fit.** Motif observé ci-dessous : marge de décision discrétisée et écarts LP supervision/scoring. Implémentation/journalisation en cours ; tests et préinscription machine avant toute nouvelle inférence. Pas de nouvelle recette adoptée, collecte conditionnelle toujours en dernier recours.

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
- [x] D0 observateur/runner testés, préinscrit puis exécuté ; résultats et revue indépendante consignés ci-dessous.
- [x] Données/comparaison : mappings/cibles et supports examinés en lecture seule ; limites de diversité et d'acquisition consignées ci-dessous. Aucune causalité ni exactitude physique des annotations démontrée par ces contrôles.
- [x] D0 : vrais scores/NLL train, alignement structurel, contrôles de reload et interventions, états initial/terminal. Les écarts numériques loss/scoring restent une piste ouverte.
- [ ] Avant tout nouveau fit : journalisation classe/description/EOS, comptes exacts, gradients/clipping/mises à jour et résumés d'époque testés puis vérifiés réellement.
- [ ] Test de mémorisation32 à budget préinscrit, si nécessaire ; verdict limité à la capacité observée.
- [ ] Contrastes complémentaires choisis selon preuves : lecteur non linéaire, tête sans Qwen, neuf mesures par voie entraînable ; exécution ou omission justifiée explicitement, jamais marquée réussie sans preuve.
- [ ] Matrice finale par hypothèse : démontrée / non soutenue dans les conditions testées / inconnue ; preuve, alternatives et portée pour chaque verdict.
- [ ] Nouvelle recette proposée seulement après restitution causale, ou constat motivé des données manquantes. Aucun réglage sur test officiel/externe.
- [ ] Code, commandes, tests, artefacts et passation publiés ; revue indépendante des conclusions.
- [ ] **Après les autres pistes seulement, si l'amélioration reste insuffisante :** préparer la diversification des acquisitions sans fuite et des paires fuite/sans fuite comparables. [[Diagnostic causal TSLM vs C1#Dernière étape conditionnelle — diversifier les acquisitions sans fuite]]. Planifié, aucune collecte lancée.

## Prochaine action

Terminer/tester le diagnostic D1 de précision de tête et la journalisation transparente du train ; publier code/préinscription avant le contraste numérique. Aucun mini-entraînement avant ce contrôle. D0 conservé dans `/home/hicham/pipe-v0/artifacts/causal-d0-733b9c6-001` et `docs/evidence/tslm-v2/causal-d0-001/`, handles30906/67983 terminaux. H100 revenue à0Mio/0% après C, instance allumée. Aucune relance des observations terminées.

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

La journalisation des futurs fits est implémentée en parallèle sans expérience : enveloppe transparente du vrai `compute_loss`, statistiques détachées sans forward supplémentaire, sommes/comptes parclip/époque, gradients avant/après clipping et mises à jour. La NLL binaire officielle reste une observation séparée à checkpoint figé. Elle ne peut pas être reconstruite à partir du seul forward supervisé.

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
