# Consolidation des branches LeakLess

Icham

## Décision du 15 septembre 2026

Branche de référence : `main`. Fusion normale (sans squash ni réécriture) de `feat/c1-opentslm-qwen` à `48d7a0784a7e6885f21db2d5775e92a6ac1aa6f2` dans `main` à `812f925c0984f046151a1cb2545ba73ae16e6d76`.

La livraison retenue comprend C1 temporel, OpenTSLM/Qwen, l'adaptateur V2 et la démo. Les expériences divergentes (dont la suite V2/27B) ne sont pas promues par cette consolidation. Les archives conservent leurs commits exacts, récupérables sans reflog. Les mentions de branches dans les preuves et journaux historiques restent inchangées.

## Résolutions de fusion

- `src/pipe/api/main.py`, lifespan : intentions distinctes combinées. Chargement optionnel TSLM et initialisation temporelle coexistent avec le store et la boucle d'alertes de démo ; les deux sont nettoyés en sortie, y compris en cas d'exception.
- `DemoExperience.tsx`, appel ModelReadout : ajout du sample actif ; définition et avertissement issus du rapport officiel conservés.
- `InspectRecording.tsx`, avertissement : texte officiel conservé ; bouton V2 explicite pour l'enregistrement importé, aucun appel automatique.
- `ModelReadout.tsx` : coexistence plutôt que remplacement du rapport officiel par des chiffres live. Inférence V2 sur demande, identité de réponse vérifiée, abstention explicite et score brut distinct d'une probabilité terrain. Le rapport gelé ailleurs dans l'UI reste intact.
- Tests DemoExperience/InspectRecording : décision initiale NOT RUN et absence de requête automatique, pas de probabilité inventée.
- Quatre conflits du miroir vault (Passation, Tableau de bord, Décisions, Journal Icham) : source dédiée distante actuelle utilisée, jamais le vieux miroir. Le vault habituel ayant des modifications locales non commitées, un clone séparé sert à la synchronisation.

## Vérification locale

Runtime Python 3.12.13 ; dépendances temporelles épinglées et dépendances ML du lock, avec torch 2.8.0+cpu / torchvision 0.23.0+cpu à la place de CUDA. Aucun téléchargement des poids ni entraînement lancé.

- `PYTHONPATH=src:scripts/timenet HF_HUB_OFFLINE=1 WANDB_MODE=disabled .venv/bin/python -m pytest -q tests` : 228 tests réussis et 105 sous-tests, aucun skip. Inclut la régression de coexistence `tests/api/test_merged_lifespan.py`.
- Réseau réel des tests API : zéro appel, zéro Telegram.
- `npm --prefix frontend test -- --maxWorkers=1` : 99 tests réussis dans 23 fichiers.
- `npm --prefix frontend run build` : TypeScript, 5 tests d'intégrité du rapport gelé et build Vite réussis. Avertissement de chunk > 500 kB, non bloquant.
- Une exécution frontend parallèle a dépassé le timeout de 5 s de `App.regression-2.test.tsx` ; la suite complète séquentielle passe sans modifier les tests ni augmenter leur timeout.

Cette vérification porte sur l'intégration logicielle CPU, pas sur le service distant H100 ni une performance terrain. Poids et raccordement du Monitor au service temporel restent des sujets distincts ; aucun redéploiement GPU n'est revendiqué.

## Registre avant nettoyage

Chaque branche ci-dessous est conservée par un tag `archive/2026-09-15/<nom-branche>` pointant exactement sur le SHA indiqué, avant sa suppression de la liste des branches. « Intégrée » signifie ancêtre de la fusion retenue ; « Archive divergente » signifie que la pointe complète n'est pas fusionnée, même si des travaux antérieurs ont été repris.

| Branche | SHA exact | Traitement |
|---|---|---|
| `feat/c1-opentslm-qwen` | `48d7a0784a7e6885f21db2d5775e92a6ac1aa6f2` | Intégrée |
| `feat/c1-temporal-alerts` | `d29fe13e024ab7865fc49b49dfdb0320506daba0` | Intégrée |
| `feat/demo-official-tslm` | `8e403d25ace84a3890380d3dc084690e4fbb4de0` | Intégrée |
| `feat/demo-simulated-incident` | `9bc921aaa2d1e8cb45a06003fea2af105d8bfe4f` | Intégrée |
| `feat/demo-temporal-building` | `5beb344d21fe7436ee42be7d031b2a7886ed70f2` | Intégrée |
| `feat/icham-quality-eval` | `16a6df043771a8a2bb2b8c7b68d24b2f3f085b93` | Intégrée |
| `feat/icham-tslm` | `afa714d3768334fd98948299271de1f2a60a3aef` | Archive divergente |
| `feat/icham-v2-reliability` | `ec73e5f17fbdda64580862ee8f92cd38032af79b` | Archive divergente |
| `feat/safoan-app` | `4dd7b8868b8e44297e0c838edfe3aa055c9e6ad8` | Intégrée |
| `feat/sensor-replay` | `81c987f583972f68399db0f3e4ef9a7b73bd393c` | Archive divergente |
| `feat/vincent-baseline-v1` | `fc837f7b05c4ab7dd8a9eabd420daf65b49db9d4` | Archive divergente |
| `nevil/setup` | `1289095fccd5f7c7c0553e22894e1fdf378d9856` | Archive divergente |
| `nevil/temporal-evidence` | `3e4e73ab6e1944b42b0223e9bd9b3d500f49bc65` | Archive divergente |

## Retrouver une variante

```sh
git fetch origin --tags
git switch -c reprise-v2 archive/2026-09-15/feat/icham-v2-reliability
```

Ne pas relancer automatiquement les fits historiques ni réutiliser leurs confirmations comme données de sélection. Pour la livraison courante : `git switch main && git pull --ff-only`.
