# Diagnostic de parité V2

L'outil ne change ni modèle, ni prétraitement, ni cache. Il ne calcule aucune
métrique de classification et n'ouvre aucun signal/cache test. Les labels ne
sont jamais passés au modèle. Exécuter depuis le checkout V2 avec l'environnement
ML existant et `PYTHONPATH=src:scripts/timenet`.

## Préflight CPU

```sh
python scripts/tslm/diagnose_parity.py --mode preflight \
  --checkpoint /home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle \
  --prepared /home/hicham/pipe-v0/artifacts/prepared \
  --data-root /home/hicham/pipe-v0/data/extracted --manifests manifests \
  --timef-version /home/hicham/pipe-v0/artifacts/prepared/timef/leakless/acoustic-leak/2.0.0 \
  --output /chemin/neuf/preflight
```

Quatre entrées fixes : validation `c00c343da6afa`, `ce0594e503562`,
`c50c4f2d91963`, puis contrôle train `c0128b879694e`. `--all-validation` remplace
les trois validations par les 208, sans changer le contrôle train.

Comparaisons : normalisation directe float64, conversion float32 contrôlée,
waveform TimeF si fournie, puis tableaux quatre bandes directs/cache/TimeF et
hypothèse `band_series(normalized.astype(float32))`. TimeF est lu avec un filtre
explicite d'identifiants ; aucune lecture des tâches/labels ni vérification
globale des shards. Les MD5 audio et les identités de split/cache sont contrôlés.

## Traces du modèle

Même commande avec `--mode trace`, dans un autre dossier neuf. Le pilote root
coordonne son exécution H100 : ne pas lancer plusieurs jobs GPU concurremment.

- Trois répétitions alternées par chemin pour les quatre cas fixes ; une par
  chemin pour les autres validations lorsque `--all-validation` est demandé.
- Même tableau via `Predictor.score_series` et scoring modèle ; comparaison
  avec les adaptateurs WAV/waveform, sans reconstruire un modèle concurrent.
- Hooks sur encodeur, projecteur et décodeur entiers ; empreintes/différences
  des entrées, embeddings/masques, logits de classe, log-probabilités et sommes.
  Le résultat avec hooks est comparé au résultat sans hooks : une instrumentation
  qui change le score ne constitue pas une preuve causale propre.
- Tableaux intermédiaires NPZ seulement pour les quatre cas fixes ; les logits
  vocabulaire complets sont comparés en mémoire, pas conservés sur disque.
- `--base /chemin/base-qwen3.5-4b` remplace uniquement la source de chargement du
  décodeur par la base originale, avec les mêmes poids temporels. Le hash de son
  état doit rester celui du Qwen gelé ; c'est une expérience séparée, pas un fix.

Pour vérifier un nouveau processus, relancer exactement la commande dans un
autre dossier avec `--reference /ancien/diagnostic/report.json`. L'outil exige
mêmes identités de checkpoint, entrées et scoring, et deux PID sur le même hôte.
Tolérance absolue des scores : **1e-6**, non configurable. Un dépassement est
rapporté comme tel, sans arrondi ni valeur de secours ; le diagnostic n'est pas
un validateur de qualité et un processus terminé peut révéler des divergences.

`report.json` contient les provenances effectives, traces et comparaisons. Le
premier étage *exactement* différent n'est pas automatiquement le premier étage
responsable d'une différence de score significative. Une cause n'est établie
qu'après comparaison contrôlée. V1 et tous ses anciens exports restent intacts.

Contrôle CPU minimal :

```sh
python -m unittest discover -s tests/tslm -p test_diagnose_parity.py -v
```
