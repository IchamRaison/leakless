# V0 ML - exécution

Icham

## Mandat et périmètre

2026-09-12 : objectif explicite « Fais tout ça », étapes 1 à 5 du fichier fourni :
environnement H100, archives/split v2, TimeNet → bandes temporelles, batch OpenTSLM,
petit apprentissage avec preuve des gradients/poids, checkpoint complet et reload
dans un processus neuf, fonction Prediction pour Safoan. Pas de score final requis.

## État vérifié

**Étapes 1 à 5 réalisées pour la V0 mécanique.** Modèle `pipe-qwen3.5-4b-v0-a968405f`.
Ce jalon ne signifie pas que le modèle est précis, ni que l'application ou le hackathon sont terminés.

- Branche `feat/icham-tslm`, worktree `/home/animus/ehl-hackathon-zurich-icham-tslm`,
  créée depuis `ddcbd75`. Le checkout principal n'a pas changé de branche.
- Livraisons relues et reprises sélectivement : Nevil `1289095` (connecteur TimeNet,
  carte, pilote, v2 et vérificateur) ; Safoan `4dd7b88` (contrat Prediction inchangé).
  Leurs branches n'ont pas été fusionnées en bloc.
- H100 80 Go accessible ; environnement isolé `/home/hicham/pipe-v0/.venv`,
  Python 3.12.3, PyTorch `2.8.0+cu128`, CUDA 12.8, driver 580.173.02.
- **Calcul GPU et backward réussis** via `scripts/tslm/check_runtime.py` :
  perte finie 249.751068, norme gradient 8.807761 ; log distant
  `/home/hicham/pipe-v0/runs/runtime.json`. Ce calcul aléatoire n'est pas un training métier.
- Dépendances installées, dont OpenTSLM `2968f4b891baab4307f7e9d0043e87677b593a30`
  et TimeNet `c39ca32b64ad0c89ea54093dbcb285c1a93eb006`. Imports vérifiés.
- **Changement demandé par Icham** : Qwen 3.5 ou Gemma 4, abandon de Llama.
  Qwen 3.5-4B retenu pour V0 ; Transformers passé à 5.13.0, import OpenTSLM compatible.
- **Poids téléchargés sans compte HF**, dépôt officiel `Qwen/Qwen3.5-4B`, révision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, licence déclarée Apache-2.0.
  Deux shards Safetensors, 9 319 828 096 octets au total ; empreintes dans
  `/home/hicham/pipe-v0/artifacts/base-qwen3.5-4b/download-receipt.json`.
- **Décodeur Qwen chargé sur H100**, aucun poids manquant/inattendu ; dimension 2560.
  Encodeur/projecteur OpenTSLM réutilisés, initialisés à neuf, 2 441 472 paramètres
  entraînables. Décodeur Qwen gelé. Log : `runs/load-qwen.log`.
- Code : `3064fc3` (choix/téléchargement) puis `994990f` (adaptateur Qwen).
  Désormais cinq tests CPU passent, dont exclusion des labels à l'inférence et sélection train/groupes.
- **1 000 WAV vérifiés et relus depuis TimeF**, 185 groupes ; folds 598/208/194,
  v2 inchangé. Aucun doublon de signal décodé entre folds. Les cinq invariants
  indépendants de Nevil passent ; couvertures I4=426/1000, I5=386/1000, autres=1000/1000.
  Ce résultat ne prouve toujours pas des sessions réelles indépendantes.
- Premier exemple train : `c0128b879694e`, groupe `gd2da168526`, RMS 1,
  bandes `[4,64]`, erreur max du round-trip 2,03e-7. Transformation unique réutilisable
  dans `pipe.tslm.preprocessing`, tolérance float32 1e-6. 61 fenêtres + 3 pas de padding.
- **Premier batch métier** : loss 0,247565, gradients encodeur 4,9889 / projecteur
  4,3217, Qwen gelé. Log `runs/check-batch.log`.
- **Petit entraînement réel terminé** : 40 étapes, 8 clips de 8 groupes train,
  2 441 472 paramètres temporels adaptés. Perte de batch 0,48272 → 0,03166,
  toutes finies ; normes de changement des poids 1,81215 / 0,44250.
  Empreinte des états Qwen identique avant/après : `c1b469d8d93ec5ff...`.
  70,3 s incluant contrôles/sauvegarde, pic alloué 19 853 237 248 octets (~18,5 Gio).
  Run numérique au commit `a968405f3c3e505582c3dcd8be0248d1075431c1` ; packaging/documentation `8e70a4c`.
- **Checkpoint autonome et reload neuf hors ligne réussis**, puis répétés dans
  `.venv-repro` reconstruit du lockfile (132 paquets compatibles). Même classe,
  texte, hash d'entrée et mesures sur l'exemple validation fixé ; WAV invalide rejeté.
  Latence de cet exemple environ 0,75–1,35 s, pas une garantie générale.

## Livraison et preuves

- Fonction : `from pipe.tslm.predict import Predictor, PredictionError`, puis
  `Predictor(checkpoint).predict(wav_bytes) -> Prediction` v0.1 Safoan inchangée.
  Charger une fois sur le backend GPU ; pas d'endpoint public créé.
- Bundle H100 : `/home/hicham/pipe-v0/artifacts/qwen-v0-smoke-001`.
  Poids texte Qwen + tokenizer/config, encodeur/projecteur, optimizer, config,
  lockfile, licences/provenance, rapports et un WAV validation original.
- SHA-256 du `checksums.json` final :
  `24f27afedfb3e26f853ff0029d8648e03ee08b46b50b906b1775ec69c96a46fe`.
  L'ancien hash `5a84da98...` dans le premier rapport est celui d'avant ajout des
  fichiers de livraison ; les poids ne changent pas. Reload final vérifié ensuite.
- Code et commandes : `docs/TSLM_V0.md`, preuves brutes versionnées dans
  `docs/evidence/tslm-v0/`. Copie des rapports, pas des poids/données dans Git.
- Contrat strict : PCM16 mono 8 kHz, exactement 1 s ; pas de conversion silencieuse.
  Erreurs : `model_unavailable`, `unsupported_audio`, `silent_audio`, `model_busy`,
  `gpu_out_of_memory`. Sortie invalide = classe nulle, texte conservé, abstention explicite.
  Pas de confiance calibrée. Mesures DSP et texte généré restent distincts.

## Décision de périmètre avant apprentissage

Run mécanique nommé **`split_v2_binary_with_noise_v0`** : folds et classes binaires
de Nevil conservés, aucun changement de groupe ni de split. Les bruits sont inclus
dans les négatifs de cette variante. Le protocole principal recommandé dans
[[Protocole évaluation]] reste fuite/non-fuite du site avec bruit externe séparé ;
la variante ne doit pas être présentée comme ce test principal.
Pas de scoring final V0. Une future comparaison exige mêmes IDs/vues pour les modèles.
Le rapport baseline de Nevil a déjà exposé son test : ne pas régler la V0 sur ces résultats.

## Accès modèle débloqué

Llama avait renvoyé HTTP 401. Sur demande d'Icham, le projet utilise désormais
une autre famille : Qwen 3.5. Téléchargement avec `token=False` réussi ; aucune
connexion HF n'est requise pour ces poids publics. Aucun accès gated contourné.
Sources : https://huggingface.co/Qwen/Qwen3.5-4B et
https://huggingface.co/docs/transformers/v5.13.0/model_doc/qwen3_5.
Les anciens adapters temporels Llama ne sont pas utilisés sous le nom de Qwen.

## Prochaine action

Safoan peut intégrer le callable réel avec le bundle et le guide. L'application
n'a pas été modifiée ni prétendue intégrée. Ensuite V1 sur le train complet,
revue collective G1, validation de qualité et comparaison baseline/TSLM sur un
même périmètre ; G4/G5 restent à faire. Ne pas présenter ce surapprentissage de
diagnostic ou la validation d'un seul clip comme une performance.
