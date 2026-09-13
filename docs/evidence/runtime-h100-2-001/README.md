# Second runtime H100 — 13 septembre 2026

Machine `ich@195.242.28.46`, environnement neuf `/home/ich/pipe-v0/.venv-repro`, snapshot `a18b22e1ead15ed741e9e264b931cba0093223cf` dans `code-a18b22e`. Aucun poids Qwen ou dataset copié à ce jalon ; aucun fit métier.

Bootstrap existant réutilisé, sans sudo ni modification du Python système : uv 0.10.9 copié depuis le poste, SHA-256 `8f8aa2a27b00bf3b35880b2e943bb8fd58714abe0981f8467b90e75faab41131`, lockfile `7b6429e67593cd945bb3c7d542cf160c9352064373d19112e865830327d1da2a`, empreintes vérifiées avant installation. `libarchive.so.13` déjà présent. Depuis le snapshot :

```bash
../tools/uv venv --python python3.12 ../.venv-repro
UV_CACHE_DIR=/home/ich/pipe-v0/uv-cache ../tools/uv pip sync --python ../.venv-repro/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 requirements-ml.lock
UV_CACHE_DIR=/home/ich/pipe-v0/uv-cache ../tools/uv pip install --python ../.venv-repro/bin/python --no-deps -e .
../tools/uv pip check --python ../.venv-repro/bin/python
../.venv-repro/bin/python -c 'import libarchive, opentslm, timenet, transformers; print("imports_ok")'
../.venv-repro/bin/python scripts/tslm/check_runtime.py
../.venv-repro/bin/python -m unittest discover -s tests/tslm -v
../.venv-repro/bin/python -m unittest discover -s tests/eval -v
```

Installation et contrôles terminés avec sortie 0 (handles 40659, 19133). 132 paquets compatibles ; Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8. Matmul/backward réels sur H100 80GB HBM3 : loss finie 249.751068, norme du gradient 8.807761. 163 tests TSLM et 51 tests évaluation passent sans skip. Logs bruts adjacents. Le lock fixe versions/révisions, pas l'identité binaire de toutes les roues ; ces contrôles ne sont pas une parité Qwen entre machines ni une mesure de vitesse d'entraînement.

## Socle train ensuite copié et vérifié

Transfert ciblé terminé avec sortie 0 (handle26958) : exactement43fichiers de la base Qwen gelée,598WAVtrain, deux caches train, préparation et deux rapports de parité. Liste issue de la préinscription de campagne et du manifeste d'audit train ; noms relatifs validés avant `tar --null --verbatim-files-from --files-from=-`, extraction dans une cible initialement sans ces trois arbres avec `--keep-old-files --no-same-owner`. Aucun WAV/cache val/test/externe copié, aucun ancien reçu réécrit.

Vérification stdlib terminée avec sortie0 (handle1166), script `e142f66`, SHA-256 `5b309b60c1e97d88d0b76d63e7b53b4431307bb55d3300860c64b52fac039ba9`, identique localement et sur le serveur :

```bash
python3 verify_assets.py --root /home/ich/pipe-v0 --code /home/ich/pipe-v0/code-a18b22e
```

`verify-assets.log` : inventaire exact646fichiers, références gelées/43SHAQwen/5SHAprovenance et598MD5source WAV conformes, aucun lien symbolique ni fichier supplémentaire. Agrégat SHA256 des assets `dc23c125f94c27b741b867ff6ee80b21475ba303dfb7600130be3fa4b1b46970`, des598WAV `2bf8c8b7974a33b93f52a46568eaf3e3b5124f56eb69b1071d22298f32c8fbe3`.

La deuxième machine dispose désormais du logiciel et du socle train vérifiés. Aucun chargement/forward Qwen réel ni fit métier n'y a encore été exécuté : cette parité reste à établir avant une campagne répartie. Les prochaines préinscriptions doivent désigner les chemins propres au nœud ; D3 reste sur la première machine, sans duplication de l'essai.
