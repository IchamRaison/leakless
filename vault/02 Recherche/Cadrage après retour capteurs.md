# Cadrage après retour capteurs

Icham

> Note historique. Le plan courant est [[Plan directeur agents]] et les fiches dans 06 Agents.

## Information fournie par Icham

Le membre du groupe a fabriqué ses propres capteurs. Icham décrit leurs mesures comme « le debit d'eau en KW etc... ». Grandeur et unité exactes non confirmées : kW désigne une puissance, pas un débit volumique. Ne pas supposer une conversion ou un format.

Icham demande une proposition réaliste pour le hackathon, sans dépendre de l’intégration de ce matériel.

## Proposition à discuter

Un assistant d’analyse de relevés pour un technicien : importer un fichier de séries temporelles provenant d’un dataset ouvert adapté, identifier un état annoté lié à une fuite et produire un compte rendu court avec observations vérifiables. Démo par rejeu de données réservées, pas par installation de capteurs. Pas de localisation géographique ni de diagnostic physique certain.

Limiter initialement la tâche à une classification de fenêtres normal/fuite si les labels disponibles le permettent. Comparer une baseline simple au TSLM fine-tuné. Générer les cibles textuelles à partir des labels et mesures calculées, en déclarant leur nature générée ; ne pas les présenter comme expertise humaine. Évaluer la classification séparément du texte et des mesures citées.

## Condition de faisabilité non levée

Aucun dataset ouvert de mesures réelles adapté n’a encore été validé. LeakDB est simulé : utilisable comme preuve sur simulation, pas comme validation terrain ni comme satisfaction automatique de l’exigence d’entrées réelles. Rechercher en priorité les données avant de figer le produit ; si elles manquent, demander aux organisateurs si une démonstration simulée clairement déclarée est recevable, sinon pivoter.

Le capteur personnel et les données clients ne sont pas des prérequis du MVP. Le professionnel peut aider à sélectionner les questions utiles et relire les sorties. Aucun projet final choisi, aucun entraînement ou résultat produit.
