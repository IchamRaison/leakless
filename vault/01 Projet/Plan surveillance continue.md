# Plan surveillance continue

Icham

## Statut et décision utilisateur

2026-09-12 : Icham précise que l'entrée vient d'un appareil qui écoute les tuyaux en permanence et signale automatiquement une fuite, **pas d'un technicien qui importe un extrait**. Cette clarification remplace le scénario produit de relecture manuelle. L'humain reçoit et examine les alertes.

Le cap produit est confirmé par cette demande ; **les choix techniques et jalons ci-dessous sont proposés, pas encore implémentés ni validés par l'équipe**. Ce tour autorise une refonte du plan et des notes, pas un nouvel entraînement, une intégration matérielle, une notification externe ou un déploiement.

Problème : surveiller un signal de canalisation et avertir lorsqu'un événement compatible avec une fuite persiste, avec des éléments vérifiables pour l'examiner. L'objectif final est bien la détection automatique ; tant que la fiabilité n'est pas validée, la sortie du prototype est une suspicion, pas la confirmation physique d'une fuite.

## 1. Deux niveaux de livraison

- **Prochain jalon hackathon proposé** : un canal, flux horodaté rejoué automatiquement, modèle réellement entraîné, suivi d'événement, alerte et historique dans l'application. Inférence calculée pendant le replay, source enregistrée explicitement signalée. Si un flux compatible d'appareil est disponible, le raccorder après vérification.
- **Validation produit** : appareil réel installé, longues acquisitions continues, défauts et événements annotés, évaluation sur sessions/installations réservées. L'embarqué, le fonctionnement hors réseau et la flotte multi-appareils ne sont pas démontrés par une H100.

Un assemblage de WAV isolés reste un scénario artificiel. Il peut tester le logiciel et la démonstration, jamais prouver un délai physique de détection ou un taux de fausses alertes sur le terrain. L'absence de chronologie réelle ne bloque pas le prototype, mais bloque ces revendications opérationnelles.

## 2. Ce que nous conservons

- V0 Qwen 3.5-4B/OpenTSLM-SP, checkpoint autonome, preprocessing unique et interface `Prediction` : [[V0 ML - exécution]]. Elle classe un extrait, pas un événement suivi dans le temps.
- TimeNet/TimeF pour préparation et traçabilité des données d'apprentissage ; même transformation numérique en inférence. Pas d'obligation ajoutée d'écrire un dataset TimeF à chaque fenêtre live.
- Split v2 et groupes existants, sans redistribution pour améliorer un score. Dataset actuel : 1 000 clips d'une seconde, pas une chronologie continue vérifiée.
- Moteur et contrôles Nevil `08562e3`, à relire/tester et intégrer ; baseline Vincent si livrée. Ses tests T1–T3 sont des perturbations de clips, pas une évaluation du cycle de vie des alertes.
- Audio/spectrogramme et affichage Safoan : ils deviennent les pièces justificatives d'une alerte. L'import manuel reste un outil de diagnostic, pas l'interaction principale.

## 3. Architecture proposée

```text
Appareil réel OU replay horodaté déclaré
  → validation du flux + contrôle de qualité
  → fenêtres causales + preprocessing partagé
  → score fuite issu du TSLM
  → historique récent + règle de déclenchement/fin
  → événement horodaté + alerte + compte rendu
  → tableau de surveillance et examen humain
```

La V0 actuelle traite une seconde PCM16 mono à 8 kHz. Proposition de départ : fenêtres d'une seconde, cadence à figer après mesure. Le flux brut et son gain/qualité restent contrôlés avant normalisation ; une fenêtre silencieuse ou manquante n'est pas une preuve d'absence de fuite.

Deux cadences proposées : score de classe régulier, texte lors de l'ouverture d'un événement ou sur demande. Le score provient du TSLM adapté au signal, pas d'une baseline ensuite racontée par Qwen. La durée, les horaires et les comptes de fenêtres viennent du logiciel ; une description d'un extrait par le TSLM ne doit pas être présentée comme une analyse de plusieurs minutes.

Le calcul de score sans génération longue reste à implémenter et chronométrer. Les latences V0 ~0,75–1,35 s sur un exemple ne garantissent pas un flux soutenu. Mesurer latence de bout en bout, p95, débit, file d'attente et mémoire ; un retard non borné ou des fenêtres perdues doivent dégrader explicitement l'état de surveillance. Le texte ne doit pas retarder l'émission de l'alerte.

Première exécution proposée sur le serveur existant, un canal ; aucune promesse de Qwen embarqué ni besoin déduit de H100 par appareil. Décider edge/cloud ensuite à partir des contraintes réseau, énergie, confidentialité et débit réel.

## 4. Étapes et critères de passage

### A — Cadrer le fonctionnement de l'appareil et le succès

Icham + personne du métier, avec Nevil/Safoan : identifier capteur, fixation/contact, grandeur mesurée, format/fréquence/gain, accès au flux, destination de l'alerte et conditions ordinaires de fonctionnement. Ne pas supposer qu'un microphone d'ambiance ou des ultrasons ont le même domaine que nos WAV. Rééchantillonnage éventuel documenté, anti-aliasing, compatibilité physique à vérifier.

Fixer ensemble délai de détection souhaité, charge de fausses alertes acceptable, durée de validation et comportement en perte de réseau. Aucun seuil chiffré de qualité ou délai garanti n'est encore choisi. Prototype : alerte dans le dashboard, pas de SMS ni d'action automatique sur une vanne.

**Sortie :** contrat de flux et critères métier écrits ; responsabilités et distinction replay/appareil réel explicites.

### B — Construire une boucle continue minimale

Safoan + Icham : ingestion/replay automatique, identifiant d'appareil et de session, numéro de fenêtre, heures source/réception, ordre et déduplication. Une fenêtre n'utilise que des observations déjà reçues, pas de contexte futur. Prévoir redémarrage, trous, données invalides et fin de replay. API privée, entrées validées, mémoire/file bornées ; conservation limitée aux données nécessaires et preuves d'événements, accès autorisé.

Ajouter un état de santé distinct de l'état d'alerte : démarrage, flux frais, surveillance dégradée ou indisponible. En cas de panne, ne pas afficher « pas de fuite » ni fermer implicitement l'incident. L'acquittement humain d'une notification ne signifie pas disparition du signal.

**Sortie :** source → traitement réel → timeline, avec arrêt du flux visible et provenance du replay. Commencer avec V0, sans attendre V1 ; pas de validation de qualité déduite.

### C — Entraîner et choisir un détecteur V1 sur fenêtres

Icham + Nevil : définir/tester le calcul de score de classe, sans confiance fabriquée. Raccorder l'export Nevil ; score non calibré indiqué comme tel. Ajouter un parcours validation seule au moteur commun avec son propriétaire. Aligner la tâche fuite/non-fuite et l'emploi du bruit externe avant toute comparaison : garder le v2 intact, nommer les vues/périmètres et comparer exactement les mêmes exemples.

Mesurer V0, puis entraîner sur tout le train retenu. Garder Qwen 3.5-4B, adapter d'abord encodeur/projecteur ; un petit nombre de configurations sélectionnées sur validation. LoRA, représentation plus riche ou modèle léger seulement si une limite observée le justifie. Comparer aux contrôles C0–C3 et à la baseline disponible, sans assimiler C3 à la Random Forest de Vincent.

**Sortie :** V1 rechargée, performances validation et coût d'inférence mesurés, export reproductible. Aucun tuning sur les résultats test déjà publiés de Nevil ; aucune qualité continue déduite de la classification des clips.

### D — Passer des scores aux événements

Icham + Nevil définissent, Safoan intègre : une règle causale versionnée de persistance et deux conditions distinctes d'ouverture/fin pour éviter les oscillations. États proposés : aucun événement actif → suspicion → alerte active → fin observée. Seuils et durées sont des paramètres à régler sur développement continu, pas une règle arbitraire présentée comme validée.

Un incident persistant produit un événement mis à jour, pas une nouvelle notification par fenêtre. Tester bruit bref, signal intermittent, suspicion persistante, retour durable au calme, fuite présente dès le démarrage, silence, coupure et reprise. Ne pas compter les fenêtres voisines comme des preuves indépendantes. Sans séquences annotées, ces essais prouvent uniquement la logique logicielle.

**Sortie :** événements cohérents/reproductibles avec source, horaires, versions, extraits justificatifs, état/raison et défauts de surveillance. La santé du flux reste distincte de la décision de fuite.

### E — Donner une utilité au langage et au dashboard

Safoan + Icham : appareil connecté et fraîcheur du signal, événement en cours, historique, extrait/spectrogramme, description et mesures distinctes. Une notification suffit pour ouvrir l'examen ; la description peut arriver après. Pas de cause physique, localisation précise, débit perdu ou réparation inventés. Le point d'installation du capteur est une métadonnée connue, pas une fuite localisée par le modèle.

Comparer le compte rendu TSLM à « baseline + mesures + gabarit » pour vérifier sa fidélité et sa compréhension par un destinataire. Une concordance texte/mesure ne démontre pas une explication causale de la décision.

**Sortie :** un responsable peut comprendre l'alerte et retrouver sa preuve ; aucune dépendance à un upload manuel pour la déclencher.

### F — Acquérir la preuve continue manquante

En parallèle dès A, Nevil + responsable du matériel à identifier : récupérer ou enregistrer sur un banc autorisé de longues séquences sans fuite, variations d'usage (vannes, pompes, débit), bruits, pertes de contact et fuites contrôlées avec début/fin annotés et incertitude indiquée. Inclure des sessions où la fuite existe déjà au démarrage. Ne pas créer de fuite sur une installation occupée pour le test.

Conserver identifiants de session/installation/capteur, étalonnage et horodatage. Diviser par acquisition/installation avant fenêtrage ; toute adaptation locale emploie uniquement un passé autorisé et une référence normale vérifiée. Garder des acquisitions réservées, sans mélange de fenêtres voisines entre développement/test. Le droit d'accès et la rétention des enregistrements restent à convenir, notamment si un capteur peut capter des conversations.

**Sortie :** corpus temporel avec durée connue et événements annotés. Les clips actuels restent utilisables pour leur question initiale ; ne pas leur inventer une continuité.

### G — Évaluer le système complet et livrer

Nevil, modèles Icham/Vincent : geler modèle, preprocessing et politique d'alerte, puis mesurer sur acquisitions réservées : rappel par événement, faux événements par appareil-heure réellement surveillée, délai depuis apparition annotée, répétitions, disponibilité/couverture et pertes de données. Rapporter durée totale, nombre d'événements, sessions et capteurs, avec limites d'incertitude. Les temps inconnus ne valent pas des heures négatives correctement surveillées.

Évaluer baseline et TSLM avec la même définition d'événement, les mêmes données et un protocole comparable de choix des seuils. Distinguer scores de classification, stress T1–T3, comportement de la règle d'alerte et évaluation sur continu réel. Aucune extrapolation « fausses alertes/jour » depuis un simple FPR par clip.

Équipe : intégrer le checkpoint figé, tester reprise/panne/reload, répéter la démo et publier preuves/licences/configurations. Si seules des séquences artificielles sont disponibles, présenter une démo de surveillance en replay et une évaluation séparée sur clips, pas un dispositif terrain validé. Vérifier deadline/format/acceptation du replay auprès des organisateurs ; les exigences TimeNet et entraînement TSLM restent maintenues.

## 5. Ordre de travail et répartition

1. A : capteur/flux et critères ; commencer F sans attendre la V1.
2. B et C en parallèle : Safoan construit la surveillance sur V0 ; Icham améliore le détecteur ; Nevil sécurise données/évaluation ; Vincent fournit le comparateur.
3. D puis E : suivi d'événement et compte rendu intégrés ; premiers réglages réels seulement lorsque F fournit du développement continu.
4. G : gel, évaluation finale, répétition et livraison ; poursuite terrain ensuite si les preuves restent insuffisantes.

Périmètre proposé pour le prochain bloc : contrat d'un canal, replay déclaré, score réel, première V1, événements et dashboard. Pas de nouveau boîtier, flotte distribuée, application mobile, achat matériel ni commande de vanne ajoutés par ce plan. Le responsable matériel, les objectifs chiffrés, le délai hackathon et le budget opérationnel restent à confirmer.

## Références et limites

État local prouvé : [[V0 ML - exécution]], [[Journal Icham#Vérification des nouveaux livrables Nevil]]. Le nouveau scénario vient explicitement d'Icham ; ce n'est pas une inférence depuis les ressources disponibles.

Référence méthodologique consultée : [DCASE 2022 — évaluation des événements sonores](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#evaluation) distingue classe et localisation temporelle, avec scénarios de réactivité et contrôle des faux positifs. Cela étaye la séparation événement/classification, pas la faisabilité de détection de fuites ni des valeurs de seuil pour PIPE. Aucun nouveau dataset DCASE adopté.
