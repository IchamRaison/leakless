# manifests/

**Le split figé. Identifiants et folds uniquement — jamais d'audio.**

| Fichier | Rôle |
|---|---|
| `split_v1.csv` | **Le contrat de split.** `clip_id`, `label`, `label_3c`, `group_id`, `device`, `fold`. Le seul fichier que le code d'entraînement lit. **Aucune métadonnée révélant l'étiquette.** |
| `split_v1_audit.csv` | **Audit uniquement.** Chemin, matériau, région, pression, débit, fenêtre, répétition, md5. Pression et débit ne sont renseignés que pour la classe *leak* — **aucune colonne de ce fichier n'entre dans une entrée de modèle.** |
| `split_v1.meta.json` | Seed, procédure, cibles, hachages `sha256`. |

Régénération à l'identique :

```bash
python3 scripts/ingest/build_groups.py --data-root <hors dépôt> --write-manifest manifests
```

Seed **20260912**. Vérifications d'overlap toutes à 0. Détail et réserves :
[`docs/SPLIT_AUDIT.md`](../docs/SPLIT_AUDIT.md).

> Ces folds sont les mêmes pour tous les modèles. Les changer après avoir vu un score invalide
> toute comparaison.
