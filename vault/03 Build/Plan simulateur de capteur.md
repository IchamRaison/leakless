# Plan simulateur de capteur

Icham

## Statut et objectif

2026-09-12, conversation parallèle : Icham demande un chantier technique pendant le travail de l'agent ML, puis son plan dans le vault. **Plan rédigé ; aucune implémentation ni exécution autorisée par cette seule demande.** Ce document détaille la brique source/replay de [[Plan surveillance continue]] §4 B, sans remplacer ce plan ni déplacer le chantier V1/T0.

Objectif : disposer d'un petit programme local qui se comporte comme une source acoustique cadencée. Il émet automatiquement des fenêtres identifiées, permet de provoquer des défauts de transmission et laisse une trace vérifiable. Il doit fonctionner sans GPU, modèle, réseau ni application pour son premier jalon.

Résultat attendu : une source d'entrée testable que l'application pourra consommer lorsque son raccordement sera convenu. **Le simulateur ne détecte pas les fuites et ne décide pas des alertes.**

## Périmètre et isolation

- Première version : un appareil simulé, un scénario fini, WAV mono PCM16 à 8 kHz, fenêtres non chevauchantes de 8 000 échantillons. Pas de rééchantillonnage, mixage, normalisation ou concaténation audio cachés.
- Utiliser une liste explicite de WAV d'une seconde issus du train ou d'un lot démo autorisé hors test. Les signaux synthétiques servent aux tests unitaires et portent cette provenance. Ne pas sélectionner de clips selon les résultats du modèle.
- Un ordre de clips isolés constitue une chronologie **artificielle**, pas une acquisition continue. Les transitions entre fichiers sont conservées dans la provenance ; ne pas les transformer en annotation d'apparition de fuite.
- Travail futur dans une branche/worktree propre distincte du chantier ML. Périmètre proposé : un script `scripts/replay_sensor.py`, un test `tests/replay/test_replay_sensor.py` et un petit scénario JSON sans audio embarqué. Vérifier les chemins disponibles avant de les créer et réutiliser l'existant si trouvé.
- Ne pas modifier `src/pipe/tslm/`, ses configurations, ses dépendances, les manifestes/splits, le harness Nevil ou les scripts de training/export. Aucun appel à la H100 ni accès aux processus de l'agent principal.
- API/frontend et `Prediction` restent à Safoan. Ce plan ne lui attribue pas une nouvelle implémentation déjà convenue ; le raccordement applicatif nécessite une revue de son interface.

Hors première version : pilote matériel, microphone système, MQTT/WebSocket, service public, stockage distribué, multi-appareils, apprentissage, seuil de fuite et envoi de notifications. Un vrai adaptateur de capteur viendra après identification de son format et de son domaine physique, pas par simple remplacement promis à l'avance.

## Chaîne minimale

```text
Liste locale de WAV autorisés + scénario explicite
  → lecture/validation d'une fenêtre
  → cadence et incidents de transport simulés
  → récepteur local borné
  → journal de réception et de pertes

Plus tard, après accord sur l'interface :
  même enveloppe de fenêtre → backend existant → véritable inférence
```

Le récepteur local confirme uniquement la réception. Il ne renvoie ni classe, ni score, ni description de substitution. Aucun nouveau serveur n'est nécessaire pour prouver le fonctionnement de la source.

## 1 — Définir le scénario et l'enveloppe

Scénario JSON local proposé : version, identifiant opaque d'appareil, liste ordonnée d'entrées audio autorisées, offsets sur la chronologie de replay et incidents explicitement datés. Vérifier l'existence, le format et les empreintes des sources avant émission ; refuser un scénario invalide sans modifier les WAV.

Enveloppe minimale de chaque fenêtre, **proposition à faire valider avant intégration à Safoan** :

| Champ | Sens |
|---|---|
| `schema_version` | Version de l'enveloppe de transport |
| `device_id`, `session_id` | Appareil simulé opaque ; nouvelle session à chaque lancement |
| `sequence` | Numéro de position dans le flux, non réutilisé dans une session |
| `source_mode` | `replay` ou `development_fixture`, jamais `live` pour la source simulée |
| `source_offset_samples` | Début dans la chronologie artificielle du scénario, pas date d'acquisition réelle |
| `sample_rate_hz`, `n_samples`, `encoding` | 8 000 Hz, 8 000 échantillons, PCM16 little-endian mono |
| `scheduled_at`, `emitted_at` | Échéance de replay et émission réelle, dates UTC |
| `audio_sha256` | Empreinte du payload WAV transmis, distincte de l'identité temporelle |
| payload audio | Octets WAV d'une seconde ; pas de chemin à lire fourni au backend |

Le récepteur ajoute `received_at` et son résultat de transport. Les fichiers source et leur provenance détaillée restent dans le manifeste local du replay, pas dans les entrées du modèle. Aucune étiquette, pression, débit, matériau ou réponse attendue dans l'enveloppe.

Deux fenêtres peuvent contenir exactement les mêmes octets : leur hash audio est alors identique, mais leur couple `(session_id, sequence)` reste différent. Une retransmission de la même fenêtre conserve ce couple. Ne pas dédupliquer une chronologie sur le seul `sample_id`/hash audio de `Prediction`.

**Critère de passage :** scénario valide → fenêtres au format explicite ; source invalide → erreur claire ; aucun label ou chemin source transmis au consommateur modèle.

## 2 — Rejouer à cadence réelle

Bibliothèque standard Python en premier : lecture WAV, JSON, horloge monotone, journal et file bornée. Aucun import PyTorch/Qwen nécessaire. Réutiliser une validation audio existante seulement si elle est disponible sans charger le runtime GPU ; ne pas recopier le DSP du modèle.

- Planifier les échéances depuis une origine monotone et les offsets du scénario, pas `sleep(1)` après chaque traitement. Le temps du consommateur ne doit pas ralentir silencieusement l'horloge source.
- Émettre la première seconde après sa durée de capture simulée, puis une fenêtre par seconde. Ne pas exposer une fenêtre avant son échéance de disponibilité.
- Conserver UTC pour les journaux ; utiliser l'horloge monotone pour cadence et retards. Un changement d'heure système ne doit pas réordonner le flux.
- Une file d'attente limitée à une fenêtre, plus une éventuellement en cours. Si le consommateur est lent, garder la plus récente en attente, compter et journaliser celle remplacée. Ne pas rejouer une rafale de fenêtres anciennes pour masquer le retard.
- Fin de scénario et interruption utilisateur explicites ; pas de boucle infinie par défaut, pas de reprise silencieuse en début de fichier.

**Critère de passage :** avec une horloge de test, échéances exactes et ordre vérifiables ; en exécution réelle, retards mesurés et publiés, pas de garantie temps réel dur. Mémoire/file bornées indépendamment de la durée du scénario.

## 3 — Injecter les défauts utiles

Incidents déterministes par numéro de fenêtre, sans aléa nécessaire pour la V0 du simulateur :

| Scénario | Comportement attendu |
|---|---|
| Réception normale | Une livraison par position, contenu audio inchangé |
| Coupure de trois fenêtres | Aucune donnée pour ces positions ; temps et séquence avancent, trous visibles à la reprise |
| Livraison retardée | Offset/identité d'origine conservés ; retard observable, sans réétiqueter l'audio ancien comme frais |
| Doublon volontaire | Même identité de fenêtre ; récepteur de test le reconnaît sans compter une seconde capture |
| Consommateur plus lent que la source | File reste bornée, pertes explicites, aucun rattrapage non borné |
| Signal constant | Trame reçue avec contenu constant ; ni perte de transport ni preuve « sans fuite » |
| WAV invalide/tronqué | Erreur de format explicite, aucune fenêtre réparée silencieusement |
| Fin puis nouveau lancement | Fin signalée, nouvelle session ; aucun mélange avec les séquences précédentes |

La coupure de transport ne produit pas un WAV de silence. Le silence audio ne signifie pas perte de connexion. Le simulateur et son récepteur exposent ces faits ; l'application décidera ensuite comment afficher une surveillance dégradée. Aucun incident ne génère une alerte de fuite factice.

**Critère de passage :** un test court couvre ces branches avec des données/horloges contrôlées ; une seule commande de test, sans dépendance GPU et sans sommeil long dans la suite.

## 4 — Livrer la brique locale et sa preuve

CLI indicative, **non implémentée ni exécutable à ce stade** :

```bash
python3 scripts/replay_sensor.py --scenario /chemin/scenario.json --output /chemin/nouveau-run
python3 -m unittest discover -s tests/replay -v
```

Le premier mode est exclusivement local : récepteur de test et journal JSONL, pas d'URL distante par défaut. Sortie dans un dossier neuf, refus d'écrasement. Le journal contient événements d'émission/réception/perte/fin, horodatages, séquences, empreintes et compteurs ; pas d'audio encodé dans chaque ligne ni de secret.

Livraison prévue : script autonome, scénario de test, test exécutable et court compte rendu de commande réelle, version et résultat observé. Les traces runtime restent hors Git. Documenter ce qui a été vérifié : intégrité audio, cadence, pertes, doublons et reprise ; aucune AUC, F1 ou métrique métier.

**Acceptation locale :** sans modèle ni réseau, un scénario fini émet les fenêtres prévues, reproduit les incidents, termine proprement et laisse une trace permettant de rapprocher chaque émission/réception. Tests exécutés et résultats réels requis avant de déclarer cette étape terminée.

## 5 — Raccorder ensuite au backend existant

Après accord de Safoan, inspecter le code de son API réellement publié et choisir le transport existant ; les endpoints de [[Contrats techniques]] restent des cibles, pas une preuve qu'une route de streaming existe.

Conserver les métadonnées de transport dans l'orchestrateur. Le modèle ne reçoit que les octets/séries autorisés et son prompt fixe. Associer sa réponse à l'identité de fenêtre sans modifier le CSV d'évaluation Nevil. Distinguer provenance de l'audio (`replay`) et calcul réellement exécuté à la réception : une inférence réelle sur un replay n'est pas une acquisition live.

Prévoir timeout borné et traitement explicite de modèle indisponible/occupé, sans réponse fictive ni retries infinis. Tout appel au serveur GPU se coordonne avec le chantier principal ; il ne fait pas partie de la première preuve locale. La mesure de débit du modèle et la politique d'alerte restent dans [[Plan surveillance continue]], hors de ce simulateur.

**Acceptation d'intégration, distincte :** fenêtres réellement reçues par le backend convenu, vraie inférence identifiée quand disponible, erreurs visibles sinon ; pas de promesse que le modèle suit 1 Hz avant mesure.

## Passation et prochaine action

- Réalisé dans cette conversation : recherche Entire sans résultat sur le simulateur, lecture des notes de surveillance/contrats/coordination, rédaction de ce plan et liens de continuité.
- Non réalisé : code, tests, sélection de fichiers audio, nouvelle branche produit, lancement de replay, connexion à l'application ou au GPU.
- La recherche n'établit pas l'absence de travaux non publiés de Safoan ; vérifier son périmètre avant l'intégration.
- Prochaine action après demande d'implémentation : vérifier l'absence de doublon dans le code courant, isoler le chantier puis réaliser les étapes 1–4 sur CPU. Étape 5 conditionnée à l'interface convenue et à la disponibilité du service.
- Ce plan ne change ni les propriétaires ni l'état d'avancement de la campagne ML. Références : [[Plan surveillance continue]], [[Architecture]], [[Contrats techniques]], [[Coordination et passation agents]].
