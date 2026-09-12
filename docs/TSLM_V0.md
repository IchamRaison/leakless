# V0 TSLM — travail en cours

Icham

La V0 prouve une intégration et un apprentissage effectifs, pas une qualité validée.
État courant et preuves dans le vault partagé, Journal Icham.

## Provenance reprise après revue

- Connecteur TimeNet, carte et pilote ; manifestes **v2 seulement** et vérificateur indépendant :
  Nevil, commit `1289095fccd5f7c7c0553e22894e1fdf378d9856`.
- `src/pipe/contracts.py` et package : Safoan,
  commit `4dd7b8868b8e44297e0c838edfe3aa055c9e6ad8`. Contrat conservé sans changement.
- OpenTSLM : `2968f4b891baab4307f7e9d0043e87677b593a30`.
- TimeNet : `c39ca32b64ad0c89ea54093dbcb285c1a93eb006`.

Le vérificateur reconstruit ses clés depuis les WAV ; ses « sessions » restent des
heuristiques de dépendance, pas des sessions de capture démontrées. La carte
TimeNet amont ne doit pas être interprétée comme une garantie d'indépendance réelle.
Le lecteur WAV de Nevil tronque/pad implicitement : nos frontières valident le
format exact avant de l'appeler, sans changer sa livraison.

## Périmètre explicite

`split_v2_binary_with_noise_v0` : manifeste v2 intact, `leak` contre `no_leak`,
bruits inclus dans le négatif comme dans la livraison de Nevil. Ce nom résout
l'ambiguïté pour le run mécanique ; il **ne remplace pas** le test principal
recommandé par le vault (fuite/non-fuite du site, bruit externe séparé).
Une comparaison ultérieure exige une vue identique pour baseline et TSLM.
La V0 n'effectue aucun scoring final. Nevil a déjà publié un résultat baseline
sur test ; aucun choix de ce run ne doit être optimisé sur ces chiffres.

## Machine et accès

Espace distant `/home/hicham/pipe-v0` : `code/`, `.venv/`, `data/raw/`,
`data/extracted/`, `artifacts/`, `runs/`, `tools/`. Pas de service réseau public.
PyTorch 2.8.0 CUDA 12.8 ; Python système 3.12.3 dans un venv isolé.
Le bootstrap utilise uv 0.10.9 copié depuis le poste, sans sudo ni Python système modifié.
Les poids Llama officiels nécessitent l'accès Hugging Face du propriétaire.
Ne jamais mettre un token dans Git, les logs ou le chat.
