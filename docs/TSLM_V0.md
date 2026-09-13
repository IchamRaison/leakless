# V0 TSLM — livraison technique

Icham

La V0 prouve une intégration et un apprentissage effectifs, **pas une qualité validée**.
Modèle : `pipe-qwen3.5-4b-v0-a968405f`. Run numérique au commit `a968405f3c3e505582c3dcd8be0248d1075431c1`.
Les commits suivants ajoutent tests, packaging/licences et documentation, sans changer les poids.

## Résultat vérifié, étapes 1 à 5

| Étape | Preuve observée |
|---|---|
| 1 — runtime | H100, PyTorch 2.8.0+cu128, calcul et backward réussis |
| 2 — données | Trois archives MD5 vérifiées ; 1 000 WAV PCM16 mono, 8 kHz, 1 s ; v2 inchangé ; invariants reconstruits depuis les sources |
| 3 — TimeNet | Premier clip train `c0128b879694e`, groupe `gd2da168526`, round-trip erreur max 2,03e-7 ; puis 1 000 records relus, 185 groupes, cache par fold |
| 4 — batch | Qwen chargé sans poids manquants ; loss 0,24756 ; gradients encodeur 4,9889 et projecteur 4,3217 ; Qwen gelé |
| 5 — apprentissage | 40 étapes, 8 clips de 8 groupes train, 2 441 472 paramètres adaptés ; toutes les pertes finies ; changements de poids mesurés ; bundle complet et reload hors ligne |

Loss de diagnostic : 0,48272 au premier batch, 0,03166 au dernier (batches différents,
**ce n'est pas un score de classification**). Normes des changements encodeur/projecteur :
1,81215 / 0,44250. L'empreinte intégrale des états Qwen gelés est inchangée.
Pic alloué observé pendant le run : 19 853 237 248 octets (~18,5 Gio).
Run, vérifications et sauvegarde : 70,3 s, hors téléchargement/installations.

Sur un exemple validation fixé avant entraînement : sortie initiale invalide
`no_leak; 0-1000`, conservée comme telle ; après adaptation, sortie structurée
`no_leak; Greatest mean spectral energy: 0-1000 Hz.`. Reload : même classe,
description et mesures. Latences observées ~0,75–1,35 s, pas un SLA ni un benchmark.

Preuves brutes : [dossier evidence](evidence/tslm-v0/), dont rapport training,
préparation, invariants, premier batch et reload. La validation d'un seul exemple
ne mesure ni précision, ni utilité, ni robustesse. Aucun scoring du test effectué ici.

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
Llama a été abandonné sur demande d'Icham. Qwen 3.5-4B officiel téléchargé avec
`token=False`, révision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Transformers 5.13.0 ; décodeur texte Qwen, pas son encodeur vision. Encodeur et
projecteur SP initialisés à neuf puis entraînés, pas d'adapter Llama réétiqueté.
Fallback DeltaNet PyTorch utilisé ; aucune dépendance de kernels compilés ajoutée.

## Installation reproductible, Linux CUDA

Python 3.12 et libarchive système nécessaires. uv 0.10.9 utilisé. Depuis le dépôt :

```bash
uv venv --python python3.12 .venv
uv pip sync --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 requirements-ml.lock
uv pip install --python .venv/bin/python --no-deps -e .
uv pip check --python .venv/bin/python
.venv/bin/python -m unittest discover -s tests/tslm -v
.venv/bin/python scripts/tslm/check_runtime.py
```

Ce lockfile a reconstruit `/home/hicham/pipe-v0/.venv-repro`, avec contrôle des
132 paquets, puis une inférence réelle hors ligne. Les versions sont figées ;
l'index CUDA fournit les wheels PyTorch `+cu128`. Ce n'est pas un environnement CPU/Mac.

## Refaire le parcours sur H100

Depuis `/home/hicham/pipe-v0/code`, avec `.venv/bin/python` désignant le Python du
venv parent (utiliser son chemin absolu ci-dessous). Réutiliser les sources est
idempotent ; **choisir un nouveau dossier pour chaque entraînement**.

```bash
/home/hicham/pipe-v0/.venv/bin/python scripts/tslm/download_base.py \
  --output /home/hicham/pipe-v0/artifacts/base-qwen3.5-4b
/home/hicham/pipe-v0/.venv/bin/python -m pipe.tslm.prepare \
  --data-dir /home/hicham/pipe-v0/data --manifest-dir manifests \
  --output /home/hicham/pipe-v0/artifacts/prepared
/home/hicham/pipe-v0/.venv/bin/python scripts/eval/verify_split_invariants.py \
  --data-root /home/hicham/pipe-v0/data/extracted --manifest manifests/split_v2.csv
/home/hicham/pipe-v0/.venv/bin/python scripts/tslm/check_batch.py \
  --base /home/hicham/pipe-v0/artifacts/base-qwen3.5-4b \
  --prepared /home/hicham/pipe-v0/artifacts/prepared
/home/hicham/pipe-v0/.venv/bin/python -m pipe.tslm.train \
  --config configs/tslm/v0.json --environment-lock requirements-ml.lock \
  --prepared /home/hicham/pipe-v0/artifacts/prepared \
  --data-root /home/hicham/pipe-v0/data/extracted \
  --base /home/hicham/pipe-v0/artifacts/base-qwen3.5-4b \
  --output /home/hicham/pipe-v0/artifacts/qwen-v0-NOUVEAU-RUN \
  --code-revision COMMIT_REEL_DU_CODE_TRANSFERE
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/hicham/pipe-v0/.venv/bin/python \
  scripts/tslm/check_reload.py \
  --checkpoint /home/hicham/pipe-v0/artifacts/qwen-v0-NOUVEAU-RUN \
  --output /home/hicham/pipe-v0/runs/reload-NOUVEAU-RUN.json
```

`COMMIT_REEL_DU_CODE_TRANSFERE` doit être remplacé par le résultat de `git rev-parse HEAD`
du checkout transféré. Pas de `.git` dans l'export distant ; ne pas inventer une révision.

## Transformation / apprentissage

Une seule fonction réutilisable : `pipe.tslm.preprocessing.preprocess_audio`.
Normalisation identique à Nevil : retrait de moyenne, RMS=1 par clip. Hann symétrique
256, hop 128 ; 61 fenêtres complètes, centres 16..976 ms, puis trois pas de padding
nul pour les patches SP. Pas de signal prétendu observé au-delà du clip.
Quatre bandes `[0,1000)`, `[1000,2000)`, `[2000,3000)`, `[3000,4000]` Hz ; énergie
unilatérale puis log1p, float32 `[4,64]`. Aucune statistique fit sur validation/test.
Tolérance du round-trip TimeF float32 : 1e-6. Pas de renormalisation par bande.

La propriété cible est la bande de plus grande énergie spectrale moyenne, calculée
sur les **61 pas réels après inversion de log1p**. Ce n'est ni une cause physique,
ni une importance causale pour la décision. Le modèle reçoit les séries, noms fixes
des bandes et instruction ; les mesures/labels/chemins/IDs ne sont pas injectés.
`answer` n'existe que pour la loss, avec padding cible masqué et EOS explicite.

## Livraison à Safoan

Bundle autonome sur la H100 : `/home/hicham/pipe-v0/artifacts/qwen-v0-smoke-001`.
Il contient `base/` (poids texte Qwen + tokenizer/config/chat template),
`temporal.pt`, `optimizer.pt`, configuration/provenance, licences, lockfile,
rapports, un WAV de validation original et `checksums.json`. Aucun poids dans Git.

Empreinte SHA-256 du **checksums.json final**, après ajout des licences/provenance :
`24f27afedfb3e26f853ff0029d8648e03ee08b46b50b906b1775ec69c96a46fe`.
Le premier rapport training/reload porte `5a84da98...` : même poids, avant les
fichiers de livraison ajoutés. `reload-delivery.json` contrôle le bundle final.

Accès via SSH autorisé existant ; pas d'URL publique ou de service GPU ouvert.
Pour récupérer le dossier depuis un compte autorisé :

```bash
scp -r hicham@89.169.123.193:/home/hicham/pipe-v0/artifacts/qwen-v0-smoke-001 ./
sha256sum qwen-v0-smoke-001/checksums.json
```

Sur le backend GPU, charger une fois puis appeler :

```python
from pipe.tslm.predict import Predictor, PredictionError

predictor = Predictor("/home/hicham/pipe-v0/artifacts/qwen-v0-smoke-001")
result = predictor.predict(wav_bytes)  # bytes originaux, pas un chemin fourni par le client
payload = result.model_dump(mode="json")
```

Compatible avec `Prediction` v0.1 de Safoan, sans modification de son schéma.
`sample_id = sha256(wav_bytes)[:24]`, hash complet aussi fourni. `score_type=none`,
aucune confiance inventée. `observations` = contrôle DSP ; `description` = texte
généré. Une divergence texte/mesure ne doit pas être masquée à l'affichage.
Une sortie invalide donne `prediction=null`, abstention `invalid_output`, texte
conservé et avertissement ; jamais un label de secours.

Erreurs `PredictionError.code` : `model_unavailable`, `unsupported_audio`,
`silent_audio`, `model_busy`, `gpu_out_of_memory`. Entrée strictement mono PCM16,
8 kHz, exactement une seconde, ≤128 Kio ; aucun recadrage/resampling silencieux.
Une inférence simultanée, 48 nouveaux tokens maximum et limite de génération
souple de 15 s (pas un timeout dur). Autres erreurs internes à traiter en 500 côté API.

Application de Safoan **non intégrée ici**. Prochaine action : brancher ce callable
sur son backend GPU/accès privé, puis V1 entraînée sur tout le développement,
évaluation commune après gel et revue G1 des interfaces. G4/G5 du hackathon ne sont pas achevés.

## Attribution

Wang Qi, Mei Zhongyi, Zhan Fan et Chen Jiongxi : dataset acoustique
[Zenodo 18631450](https://doi.org/10.5281/zenodo.18631450), CC BY 4.0 déclarée.
Le WAV du bundle n'est pas modifié ; ses représentations sont transformées comme décrit ci-dessus.
[Qwen 3.5-4B officiel](https://huggingface.co/Qwen/Qwen3.5-4B), licence Apache-2.0
et carte d'origine incluses. Sources de code OpenTSLM/TimeNet épinglées plus haut.
