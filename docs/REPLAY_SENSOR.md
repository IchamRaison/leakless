# Simulateur de capteur local

Python 3.10+ et bibliothèque standard uniquement. Depuis la racine du worktree :

```bash
python3 -m unittest discover -s tests/replay -v
python3 tests/replay/test_replay_sensor.py --write-demo /tmp/pipe-replay-demo
python3 scripts/replay_sensor.py --scenario /tmp/pipe-replay-demo/scenario.json --output /tmp/pipe-replay-demo/run
```

Les deux dossiers de sortie doivent être neufs. Le générateur de test écrit un WAV synthétique constant et renseigne ses empreintes dans le scénario fourni. Aucun audio n'est embarqué dans Git. Ce scénario de huit secondes provoque trois coupures, un retard de 0,5 seconde et un doublon immédiat. Résultat attendu : cinq réceptions uniques, trois pertes, six émissions dont une retransmission ignorée. Aucun résultat de détection.

Pour des WAV autorisés, copier `examples/replay/scenario.json`, renseigner chemins relatifs au JSON et SHA-256 exacts (`sha256sum fichier.wav`), utiliser `source_mode: replay` et la provenance `train` ou `authorized_demo`. Cette provenance est une déclaration de l'opérateur, pas une vérification contre le split gelé. Ne jamais fournir le test. `development_fixture` exige `synthetic`. Offsets entiers en échantillons, croissants et sans chevauchement ; première disponibilité à `(offset + 8000) / 8000` secondes. Chaque WAV doit être mono PCM16, 8 kHz et exactement une seconde, au maximum 64 Kio avec ses en-têtes. Les fichiers sont validés avant création du run et revérifiés avant émission.

Incidents indexés à partir de zéro : `drop`, `delay` avec `seconds`, `duplicate` (retransmission immédiate uniquement), au plus un incident par position. `consumer_seconds` simule la durée de traitement local. Aucun aléa, réseau, modèle ou GPU. Le récepteur confirme uniquement les octets WAV en mémoire.

Le journal `events.jsonl` trace émission, traitement, réception, pertes, doublons et terminaison avec compteurs. `manifest.json` conserve chemins, provenance et transitions de la chronologie artificielle. Les lignes de transport ne contiennent ni chemin source, ni label, ni payload encodé. Deux captures identiques ont des séquences différentes ; la retransmission conserve session et séquence. Une relance crée un UUID de session neuf.

L'ordonnanceur utilise une origine monotone, UTC ne servant qu'aux traces. Il conserve au plus une fenêtre en traitement et la plus récente en attente. Les échéances accumulées lors d'un retard du processus sont traitées avant le consommateur, avec pertes explicites. Une arrivée retardée ne remplace pas une fenêtre plus récente déjà en attente. Les métadonnées du scénario sont en mémoire O(n), les payloads restent bornés. Le consommateur local est simulé dans la boucle ; raccorder un appel bloquant réel demandera un worker distinct et un timeout convenu avec Safoan.

Ctrl-C produit une fin `interrupted` et un code 130 ; les fenêtres en attente/en traitement sont signalées perdues, les positions futures comptées `unprocessed_positions`. Une erreur pendant le replay est journalisée `failed` et remonte au CLI. Une entrée invalide est refusée avant création du run. Le dossier neuf évite tout écrasement.

Preuve exécutée le 2026-09-12 sous Python 3.13 : deux tests regroupant les branches passent ; replay réel terminé, cinq réceptions, trois coupures et un doublon ignoré. Retard injecté mesuré à 0,500202 s ; maximum hors injection 0,002864 s sur ce seul run, aucune garantie temps réel. Traces locales : `/tmp/ehl-sensor-demo-20260912/run`. Ces fichiers temporaires peuvent disparaître ; les commandes ci-dessus reproduisent la preuve.

Étapes 1–4 du plan livrées. Étape 5 restante : convenir de l'interface applicative avec Safoan puis réaliser le raccordement et la vraie inférence. Ce simulateur ne valide ni acquisition continue, ni performance terrain, ni capacité du modèle à suivre 1 Hz.
