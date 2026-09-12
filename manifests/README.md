# manifests/

**Le split figé. Identifiants et folds uniquement — jamais d'audio.**

## ✅ Actif — `split_v2`

| Fichier | Rôle |
|---|---|
| `split_v2.csv` | **Le contrat de split.** `clip_id`, `label`, `label_3c`, `group_id`, `fold`. Le seul fichier que le code d'entraînement lit. Ni chemin, ni device, ni pression, ni débit. |
| `split_v2_audit.csv` | **Audit uniquement.** Chemin, device, matériau, région, pression, débit, catégorie de bruit, fenêtre, répétition, md5. **Aucune colonne n'entre dans une entrée de modèle.** |
| `split_v2.meta.json` | Seed, procédure, contraintes d'acceptation, statistiques d'arêtes, sensibilité du seuil, `sha256`. |

```bash
python3 scripts/ingest/build_split_v2.py --data-root <hors dépôt>
python3 scripts/eval/verify_split_invariants.py --data-root <hors dépôt> --manifest manifests/split_v2.csv
```

Seed **20260912**, tirage accepté n° 7535. Les cinq invariants de fuite sont tenus, vérifiés par
reconstruction depuis l'audio et non par lecture du `group_id`. Détail, cause racine et réserves :
[`docs/SPLIT_V2_AUDIT.md`](../docs/SPLIT_V2_AUDIT.md).

## 🔴 Invalide — `split_v1`, conservé comme artefact historique

| Fichier | Statut |
|---|---|
| `split_v1.csv`, `split_v1_audit.csv`, `split_v1.meta.json` | **INVALIDES.** Ne pas entraîner dessus, ne pas comparer dessus. |

**Pourquoi** : 30 conditions physiques traversaient les folds (92 clips), plus 10 paires de
quasi-doublons et 5 conditions non-leak. Son contrôle `condition_overlap = 0` exigeait la
présence d'un champ absent pour 348 des 500 clips leak, et ne couvrait donc que 18 % d'entre eux.

Ces fichiers ne sont **jamais modifiés en place** : l'erreur doit rester inspectable.

> Les folds de `split_v2` sont les mêmes pour tous les modèles. Les changer après avoir vu un
> score invalide toute comparaison.
