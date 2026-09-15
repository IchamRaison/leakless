# Datasets utiles pour PIPE

Icham

Recherche initiale : 2026-09-12 ; revue : 2026-09-13.

Statut : sources documentaires réexaminées, aucun nouveau téléchargement de données dans cette revue. Depuis la recherche initiale, Aghashahi a été préparé et réservé dans [[V2 ML - exécution]] ; toujours sans score externe. Les autres pistes ne sont pas auditées localement ici.

## Verdict rapide

Ne pas mélanger de nouvelles données au train/V1 ou au test T0 déjà gelé. Le dataset Zenodo actuel reste le benchmark principal ; les sources ci-dessous servent à tester le transfert acoustique, la surveillance par événements ou l’utilité opérationnelle.

## Priorité immédiate

| Dataset | Ce qu'il apporte | Usage PIPE | Réserve |
| --- | --- | --- | --- |
| [Zenodo 18631450](https://zenodo.org/records/18631450) | 1 000 clips acoustiques d'une seconde, 500 fuite, 386 sans fuite et 114 bruits environnementaux ; source expérimentale Dongguan + bruits externes ; CC BY 4.0. | Benchmark actuel, démo et T0/T1–T3 déjà livrés. | Un seul site principal, clips courts, groupes v2 heuristiques ; ne prouve ni continuité ni terrain. |
| [Aghashahi — Mendeley](https://data.mendeley.com/datasets/tbrnp6vrnj/1) | 280 signaux étiquetés de30s, quatre types de fuite + sans fuite, deux topologies, bruit/débit variables ; hydrophones RAW8kHz, accélération/pression CSV ; CC BY4.0. | Réserve externe actuelle. Aussi candidat potentiel à un futur apprentissage, mais cela exigerait une réaffectation explicite et une autre confirmation. | Banc PVC unique ;280 signaux ≠280 expériences indépendantes. Grouper acquisitions et modalités liées. Fréquence51,2kHz des autres modalités non reconfirmée sur cette page. |
| [Hong Kong WDN — Mendeley](https://data.mendeley.com/datasets/hkn8mxcjyz/1) | Réseaux réels enterrés, environ90 sites de fuite surun an, loggers/hydrophones/MEMS ; fuites signalées puis sites réparés, métal/non-métal ; CC BY4.0. | Candidat le plus prometteur de cette shortlist pour élargir la diversité terrain, en futur train ou confirmation selon un rôle fixé avant usage. | Indépendance et appariement complet avant/après réparation non vérifiés ; fréquences, formats, sessions et tailles exactes à auditer. |

## Pour prouver la surveillance continue

| Dataset | Signal utile | Ce qu'il permet | Limite |
| --- | --- | --- | --- |
| [Yorkshire Water acoustic logger](https://www.data.gov.uk/dataset/60360ad6-7d48-4324-88ca-646aa0dce879/yorkshire-water-daily-acoustic-logger-data2) | Niveau moyen `lvl` et dispersion `spr` quotidiens par logger, alarmes et visites terrain ; quelques fichiers son d'exemple. | Tester une chronologie d'alerte, la priorisation et les faux positifs au niveau site/jour. | Pas un benchmark WAV prêt à entraîner ; la licence du dataset est indiquée « non définie » dans le catalogue, à confirmer par ressource. |
| [Wessex Water acoustic logger](https://marketplace.wessexwater.co.uk/dataset/leakage-acoustic-logger-data) | Tableur de 63,21 MB avec données logger, contexte d'actif/GIS, travaux et économies ; licence CC BY. | Définir la valeur métier d'une alerte et un modèle de priorisation des points d'intérêt ; comparer mesures acoustiques et contexte. | Vérifier si les colonnes logger sont des résumés et non des WAV. Les colonnes de résultat ne doivent pas devenir des entrées : la page recommande seulement B–J et O–AH comme inputs. |
| [Réseau d'eau + compteurs intelligents NTNU](https://zenodo.org/records/14001028) | Débit/pression de compteurs àune minute, campagne septembre2023, événements de fuite avec début/fin. [README primaire](https://zenodo.org/records/14001028/files/README.txt?download=1). | Replay et contexte hydraulique pour ouverture, persistance et fin d'événement ; pas de remplacement des entrées audio. | Horodatages des fuites approximatifs à±3min, exception de résolution/défauts possibles surSWM1 ; licence à vérifier avant redistribution. |
| [Réseau urbain slovaque multi-source](https://doi.org/10.5281/zenodo.15096167) | Un an horaire, 8 737 lignes, 18 scores d'anomalie et labels `fault_d7` autour de défauts confirmés ; CC BY 4.0. | Sanity check du lead/lag et de la logique d'alerte sur une longue chronologie. | Une seule zone ; scores déjà transformés, étiquette ±7 jours large ; pas de signal brut ni de vérité « absence de fuite ». |

## Benchmarks hydrauliques secondaires

- [BattLeDIM 2020](https://zenodo.org/records/4017659) : SCADA débit/pression et événements de fuite pour tester le détecteur d'événements sur un benchmark connu ; simulation/benchmark hydraulique, pas validation acoustique terrain.
- [LeakDB](https://github.com/KIOS-Research/LeakDB) : nombreux scénarios artificiels réalistes sur réseaux d'eau ; utile pour générer des cas et comparer une règle hydraulique, mais insuffisant seul pour une revendication terrain.
- [MIMII](https://arxiv.org/abs/1909.09347) et [IICA](https://zenodo.org/records/7551606) : sons industriels/air comprimé intéressants pour du bruit auxiliaire, mais domaines différents ; IICA annonce 5 592 fichiers de 30 s et 22,1 GB d'archives. Ne pas les prioriser pendant le hackathon.

## Revue du 13 septembre — diversité globale

Icham demande d'examiner la diversité de l'ensemble des données, pas seulement les sept groupes sans-fuite. La source [Zenodo actuelle](https://zenodo.org/records/18631450) situe les deux classes de tuyaux au même site principal à Dongguan. Matériaux, montages, conditions hydrauliques, mécanismes de fuite et contexte normal doivent tous entrer dans l'audit de couverture ; davantage de clips ne garantit pas davantage d'acquisitions indépendantes.

**Avis : bonne shortlist de sources complémentaires, pas un corpus audio prêt à fusionner.** Hong Kong élargit surtout les situations terrain ; Aghashahi les facteurs expérimentaux contrôlés. Yorkshire et Wessex servent plutôt à tester l'alerte et sa priorisation : résumés journaliers/visites pour le premier, données tabulaires et contexte pour le second. NTNU, le réseau slovaque et les benchmarks hydrauliques ne remplacent pas des signaux acoustiques. MIMII et IICA sont d'autres domaines industriels ; ne pas relabelliser automatiquement leurs anomalies comme fuites d'eau.

**Question d'Icham : réserver Aghashahi ne prive-t-il pas le train de bonnes données ?** Oui, il existe un coût d'opportunité ; son rôle n'est pas sacré. La qualité annoncée n'établit toutefois ni un gain d'apprentissage ni une couverture terrain suffisante. Proposition à discuter après les diagnostics : futur train enrichi avec Aghashahi et autre réserve réellement exploitable, éventuellement Hong Kong après audit. Alternative : réserver dès le départ des acquisitions réellement distinctes, avec une confirmation inter-sites séparée. Jamais les fenêtres ou capteurs de la même expérience des deux côtés.

**Aucune réaffectation validée.** La réserve Aghashahi actuelle (`84007801…`,3600 fenêtres primaires issues de120 enregistrements hydrophone) reste intacte et sans score. Pour la réaffecter, il faudrait annoncer un nouveau protocole et renoncer explicitement à l'utiliser comme confirmation indépendante de ce futur entraînement. La revue documentaire n'autorise ni téléchargement supplémentaire, ni mélange de données, ni fit. D1 et les autres contrastes justifiés restent à réaliser ; ne pas les déclarer épuisés.

## Ordre recommandé — proposition, pas lancement

1. Terminer D1 puis les contrôles de représentation/apprentissage justifiés dans [[Diagnostic causal - exécution]] ; ne pas substituer une nouvelle collecte à ces expériences.
2. Si l'élargissement est retenu ensuite, auditer Hong Kong et décider explicitement des rôles train/développement/confirmation avant les expériences. Aghashahi reste réservé tant qu'un nouveau protocole n'est pas validé ; ne pas refaire sa préparation déjà terminée.
3. Pour un futur corpus multi-source : auditer la couverture des deux classes par site/capteur/condition, les recouvrements et les acquisitions liées. Ne pas laisser la provenance devenir un substitut du label. Mesurer l'apport des nouvelles données face à un témoin sans ajout, même méthode et protocole.
4. Garder Yorkshire/Wessex et les séries hydrauliques pour les usages événement/priorisation effectivement implémentés, pas pour gonfler le nombre de données audio annoncé.

## Porte de validation avant téléchargement massif

Pour chaque source : licence des fichiers, formats/fréquences, labels et provenance, identifiants de session/site, dédoublonnage, possibilité d'un split par acquisition et présence d'un signal d'apparition/fin. Si une de ces réponses manque, la source reste une piste documentée et ne devient pas une preuve de généralisation.

## Sources et date de vérification

Pages de la shortlist revérifiées le2026-09-13 ; README NTNU consulté. Wessex : page directe403, description du domaine officiel retrouvée par recherche indexée. Les propriétés déclarées ne remplacent pas un audit de fichiers ; indépendance Hong Kong et correspondances multimodales non démontrées. Les archives de données n'ont pas été téléchargées dans cette revue. La préparation Aghashahi antérieure est tracée dans [[V2 ML - exécution]].
