# Datasets utiles pour PIPE

Icham

Date : 2026-09-12  
Statut : shortlist recherchée, non téléchargée et non auditée localement dans cette note.

## Verdict rapide

Ne pas mélanger de nouvelles données au train/V1 ou au test T0 déjà gelé. Le dataset Zenodo actuel reste le benchmark principal ; les sources ci-dessous servent à tester le transfert acoustique, la surveillance par événements ou l’utilité opérationnelle.

## Priorité immédiate

| Dataset | Ce qu'il apporte | Usage PIPE | Réserve |
| --- | --- | --- | --- |
| [Zenodo 18631450](https://zenodo.org/records/18631450) | 1 000 clips acoustiques d'une seconde, 500 fuite, 386 sans fuite et 114 bruits environnementaux ; source expérimentale Dongguan + bruits externes ; CC BY 4.0. | Benchmark actuel, démo et T0/T1–T3 déjà livrés. | Un seul site principal, clips courts, groupes v2 heuristiques ; ne prouve ni continuité ni terrain. |
| [Aghashahi — Mendeley](https://data.mendeley.com/datasets/tbrnp6vrnj/1) | 280 signaux entièrement étiquetés, 30 s, quatre types de fuite + sans fuite, deux topologies, bruit/débit variables ; hydrophones à 8 kHz, accéléromètres et pression à 51,2 kHz ; CC BY 4.0. | Meilleur hold-out acoustique externe immédiatement compatible avec la voie hydrophone 8 kHz ; test de transfert et de robustesse. | Banc PVC de laboratoire, petit volume ; regrouper par expérience/topologie, ne pas faire de split aléatoire par fenêtre. |
| [Hong Kong WDN — Mendeley](https://data.mendeley.com/datasets/hkn8mxcjyz/1) | Signaux de réseaux réels enterrés, environ 90 sites sur un an, noise loggers, hydrophones et accéléromètres MEMS ; fuites signalées vs sites réparés ; conduites métalliques et non métalliques ; CC BY 4.0. | Hold-out terrain indépendant, ou audit de généralisation par modalité et matériau. | Fréquences, formats, identifiants de session et taille exacte restent à vérifier avant tout code. |

## Pour prouver la surveillance continue

| Dataset | Signal utile | Ce qu'il permet | Limite |
| --- | --- | --- | --- |
| [Yorkshire Water acoustic logger](https://www.data.gov.uk/dataset/60360ad6-7d48-4324-88ca-646aa0dce879/yorkshire-water-daily-acoustic-logger-data2) | Niveau moyen `lvl` et dispersion `spr` quotidiens par logger, alarmes et visites terrain ; quelques fichiers son d'exemple. | Tester une chronologie d'alerte, la priorisation et les faux positifs au niveau site/jour. | Pas un benchmark WAV prêt à entraîner ; la licence du dataset est indiquée « non définie » dans le catalogue, à confirmer par ressource. |
| [Wessex Water acoustic logger](https://marketplace.wessexwater.co.uk/dataset/leakage-acoustic-logger-data) | Tableur de 63,21 MB avec données logger, contexte d'actif/GIS, travaux et économies ; licence CC BY. | Définir la valeur métier d'une alerte et un modèle de priorisation des points d'intérêt ; comparer mesures acoustiques et contexte. | Vérifier si les colonnes logger sont des résumés et non des WAV. Les colonnes de résultat ne doivent pas devenir des entrées : la page recommande seulement B–J et O–AH comme inputs. |
| [Réseau d'eau + compteurs intelligents NTNU](https://zenodo.org/records/14001028) | Mesures de débit à une minute, réseau réel isolé, 45 jonctions/48 conduites, quatre matériaux, essais de fuite contrôlée à 0,7–29 m³/h. | Replay d'événements et sidecar débit/pression pour tester ouverture, persistance, fin et délai. | Pas acoustique ; événements contrôlés, licence à vérifier avant redistribution. |
| [Réseau urbain slovaque multi-source](https://doi.org/10.5281/zenodo.15096167) | Un an horaire, 8 737 lignes, 18 scores d'anomalie et labels `fault_d7` autour de défauts confirmés ; CC BY 4.0. | Sanity check du lead/lag et de la logique d'alerte sur une longue chronologie. | Une seule zone ; scores déjà transformés, étiquette ±7 jours large ; pas de signal brut ni de vérité « absence de fuite ». |

## Benchmarks hydrauliques secondaires

- [BattLeDIM 2020](https://zenodo.org/records/4017659) : SCADA débit/pression et événements de fuite pour tester le détecteur d'événements sur un benchmark connu ; simulation/benchmark hydraulique, pas validation acoustique terrain.
- [LeakDB](https://github.com/KIOS-Research/LeakDB) : nombreux scénarios artificiels réalistes sur réseaux d'eau ; utile pour générer des cas et comparer une règle hydraulique, mais insuffisant seul pour une revendication terrain.
- [MIMII](https://arxiv.org/abs/1909.09347) et [IICA](https://zenodo.org/records/7551606) : sons industriels/air comprimé intéressants pour du bruit auxiliaire, mais domaines différents ; IICA annonce 5 592 fichiers de 30 s et 22,1 GB d'archives. Ne pas les prioriser pendant le hackathon.

## Ordre recommandé

1. Auditer puis réserver Aghashahi comme hold-out acoustique externe ; harmoniser seulement le sous-ensemble hydrophone 8 kHz.
2. Auditer Hong Kong pour un vrai test de transfert terrain, sans l'utiliser pour régler le checkpoint actuel.
3. Utiliser Yorkshire/Wessex pour la couche événement/priorisation, pas pour prétendre entraîner le classifieur audio.
4. Ajouter NTNU ou le dataset slovaque uniquement si l'équipe implémente effectivement le replay continu.

## Porte de validation avant téléchargement massif

Pour chaque source : licence des fichiers, formats/fréquences, labels et provenance, identifiants de session/site, dédoublonnage, possibilité d'un split par acquisition et présence d'un signal d'apparition/fin. Si une de ces réponses manque, la source reste une piste documentée et ne devient pas une preuve de généralisation.

## Sources et date de vérification

Pages consultées le 2026-09-12. Les caractéristiques et limites sont celles annoncées par les dépôts/catalogues ; aucun fichier n'a encore été téléchargé, exécuté ou intégré au modèle PIPE.
