# Vérification de livraison — 13 septembre 2026

Code exécuté : `77c174f`, `/home/hicham/pipe-v0/code-language-api-004`, H100 de `hicham@89.169.123.193`.

```bash
PYTHONPATH=/home/hicham/pipe-v0/.venv-temporal-api/lib/python3.12/site-packages:src \
  /home/hicham/pipe-v0/.venv-repro/bin/python -m pytest tests/temporal tests/api tests/replay -q
```

Résultat observé : **39 passed, 2 warnings in 3.87s**. Avertissements de dépréciation Starlette/httpx et AnyIO, aucun skip ni échec.

`evaluation.json` : modèle brut16/25,9fallbacks,8/25 sur inversion, reload_max_diff0. Les compteurs ont été recalculés à partir des observations individuelles. Développement artificiel, pas de généralisation terrain validée.

`http-smoke-001/report.json` : 43fenêtres audio réelles en chronologie artificielle, notification31e haute, doublon HTTP idempotent, zéro envoi. Persistance reconnue par le modèle ; fin mal reconnue et explicitement remplacée par la règle. P95ingestion C1 :8,12ms, hors génération asynchrone.

`service-launch.json` : service loopback8020 distinct de8019 et du repli195, un travailleur modèle, aucun WhatsApp. Le processus est laissé disponible pour l'intégration ; pas de superviseur de redémarrage automatique configuré.
