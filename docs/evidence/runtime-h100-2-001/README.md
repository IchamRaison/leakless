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
