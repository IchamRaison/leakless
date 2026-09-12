Status: team-approved direction, conditional on dataset audit
Date: 2026-09-12

WHY
Les équipes de maintenance ne peuvent pas surveiller et interpréter en permanence
les signaux produits par une infrastructure d’eau.

Le besoin n’est pas simplement de “détecter une fuite avec de l’IA”.

La décision utilisateur est :
“Ce signal inhabituel mérite-t-il une inspection, ou ressemble-t-il davantage
à une situation sans fuite / à du bruit environnemental ?”

Utilisateur cible :
Facilities manager / technicien de maintenance.

QUESTION EXPERIMENTALE
Can a time-series language model distinguish leak-associated acoustic recordings
from non-leak and environmental noise while explaining its decision using
measurable signal properties?

HOW
Hypothèse :
Des propriétés temporelles et acoustiques mesurables contiennent suffisamment
d’information pour distinguer leak / no-leak / environmental noise.

Pipeline :
WAV mesuré
→ TimeNet
→ propriétés mesurables du signal
→ descriptions textuelles fondées sur ces mesures
→ TSLM
→ classe / description
→ comparaison avec baseline

Même split groupé pour baseline et TSLM.
Pas de split aléatoire par fichier si plusieurs clips proviennent de la même
condition/session/device.

WHAT
Prototype software-only :

- entrée : enregistrement acoustique temporel ;
- sortie : leak / no-leak / environmental noise ;
- description fondée uniquement sur des propriétés mesurées ;
- comparaison baseline vs TSLM sur le même held-out split.

NON-CLAIMS

- aucun hardware LeakLess testé ;
- aucune validation terrain/client ;
- aucune localisation réelle démontrée ;
- aucune cause physique inventée ;
- aucune recommandation de réparation ;
- aucune supériorité du TSLM avant mesure.

GATE
On continue seulement si l’audit dataset confirme :

- labels exploitables ;
- groupes reconstructibles ;
- split held-out défendable ;
- leakage contrôlable.

QUESTION DE DEMO
Pourquoi utiliser un TSLM plutôt qu’un simple classifieur acoustique ?

La valeur du TSLM doit être mesurée, pas supposée.
