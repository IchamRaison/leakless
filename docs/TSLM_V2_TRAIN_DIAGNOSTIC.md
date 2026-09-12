# Diagnostic de discrimination V2 — train uniquement

`scripts/tslm/diagnose_train.py` répond à une question limitée : une régression
logistique fixe exploite-t-elle la représentation réellement fournie au TSLM,
comparativement aux descripteurs C1 sur les mêmes groupes réservés ? Une sonde
faible ne prouve pas l'absence d'information. C1 peut exploiter des particularités
d'acquisition. Aucun résultat de ce diagnostic n'est une évaluation indépendante.

## Protocole fixé avant exécution

- 598 clips et 102 groupes du train officiel seulement. Les fichiers de split
  sont lus pour leur intégrité et leur découpage ; aucun WAV, cache ou score de
  validation officielle/test n'est ouvert par le diagnostic.
- Trois plis : groupes triés par classe (`no_leak`, puis `leak`), mélangés par
  `random.Random(20260912)`, affectés circulairement aux trois plis. Chaque
  groupe est réservé une fois. La stratification porte sur les groupes, pas sur
  les effectifs de clips, qui restent explicitement rapportés.
- Une sonde `StandardScaler` + `LogisticRegression(C=1, max_iter=5000,
  solver="lbfgs", random_state=20260912)`, sans pondération des classes ni grille.
  Le scaler et le classifieur voient seulement la partie d'apprentissage du pli.
- Entrées TimeNet : tableau canonique `(4,64)` aplati par lignes, y compris ses
  douze valeurs de padding nul. Comparateur : neuf descripteurs C1 officiels de
  `harness.features.c1_envelope`, même recette logistique fixe. Ce C1 diagnostique
  n'est pas le C1 historique dont `C` était choisi sur la validation officielle.
- Six fits au total : trois plis pour l'unique configuration de sonde et trois
  pour l'unique C1 fixé. Convergence non atteinte = erreur, pas un nouvel essai.
- Métriques du harness par pli, puis moyennes des AUC par pli. Pas d'AUC globale
  des prédictions issues de plusieurs modèles. Le seuil `0.5` est fixé uniquement
  pour afficher rappel/FPR/FP/FN diagnostiques ; ni choisi ni livré au produit.
  La validation officielle demeure réservée au seuil du futur modèle final.

## Commande et artefacts

```bash
PYTHONPATH=src:scripts/timenet python scripts/tslm/diagnose_train.py \
  --prepared /chemin/cache-canonique \
  --parity-report /chemin/parite-processus-neuf/report.json \
  --data-root /chemin/WAV --manifests manifests \
  --output /chemin/diagnostic-neuf --code-revision SHA_GIT_COMPLET
```

Le receipt `pipe-parity-v2` doit avoir `all_checks_pass=true` et les cinq contrôles
vrais, tolérance `1e-6`, version canonique et empreintes du code/cache correspondantes.
Le SHA du `train.npz` est vérifié sans ouvrir `val.npz`. Tous les WAV train passent
le contrôle MD5 du manifeste gelé et leur tableau canonique doit être identique
au cache (dtype et valeurs), avant le premier fit. Un dossier existant est refusé.

Fichiers produits :

- `folds.json` : IDs d'apprentissage/réservés, groupes, seed, algorithme et SHA de
  split/cache/parité ; écrit avant le premier fit. La campagne phase 5 doit
  réutiliser ces IDs plutôt que reconstruire un autre découpage.
- `metadata.json` : configurations réellement consultées, seuil diagnostique,
  versions, SHA audio train/code et provenance de parité.
- `probes.json` : métriques et prédictions par pli, moyennes des AUC, pas de score
  final indépendant. Ce fichier interne n'est pas un run du contrat de livraison.
- `batch-composition.json` : comptages de classes/groupes dans les lots de la
  recette V1 (8 époques, lots de 8), sans lancer d'entraînement ni choisir le
  budget de V2 d'après ces comptages.

Un échec laisse les artefacts intermédiaires présents : absence du message final
`completed=true` signifie diagnostic incomplet. Aucun fichier n'est écrasé.

## Autopsie optionnelle des tokens et gradients

Ajouter `--checkpoint /chemin/bundle-v1` exige un GPU et l'autorisation de lancement
par le responsable de campagne. Le bundle doit correspondre au SHA du receipt de
parité. Le chargeur existant vérifie les fichiers ; base Qwen et poids temporels
V1 sont chargés, sans réinitialisation ni `optimizer.step`.

`supervision.json` porte sur huit clips train issus de huit groupes fixés via
`train.debug_rows` : quatre groupes par classe, premiers IDs triés. Ces poids ont
déjà vu le train : il s'agit d'une autopsie mécanique, jamais d'une mesure
hors-groupes. L'entrée est le cache canonique ; la version déclarée par l'ancien
checkpoint reste consignée séparément. L'exécution est `eval()` avec gradients
activés, pour retirer le dropout, pas simuler une étape stochastique de campagne.

Le wrapper observe le véritable appel `compute_loss` sans changer ses arguments
ni sa sortie. Il vérifie tokens de classe complets avec `;`, frontière du texte,
EOS, masques prompt/padding et décalage causal. Classe et description + EOS sont
décomposées avec **le même dénominateur : tous les tokens supervisés**, pour que
leur somme reconstruise la loss réelle. Les normes de gradients de l'encodeur et
du projecteur sont calculées pour ces deux contributions et pour la loss entière.
Le texte et l'EOS restent comptés séparément. Les poids/buffers des trois modules
sont hachés avant/après, Qwen reste gelé ; aucune étape d'optimiseur n'existe.

Un poids plus important des tokens descriptifs ou une norme plus élevée n'établit
pas à lui seul la cause de la mauvaise discrimination : il faut une comparaison
contrôlée de la variante de supervision après ce diagnostic.

## Vérification locale

```bash
PYTHONPATH=src python -m unittest discover -s tests/tslm -p test_diagnose_train.py -v
```

Ces tests utilisent des fixtures synthétiques CPU, pas le dataset ni les poids.
Le fit réel, la décomposition PyTorch et les gradients exigent le runtime ML de
référence et restent à vérifier après la porte de parité.
