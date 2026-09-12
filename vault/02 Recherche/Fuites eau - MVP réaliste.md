# Fuites eau - MVP réaliste

Icham

> Note historique de cadrage. Pour exécuter le projet actuel, suivre [[Plan directeur agents]] et les fiches dans 06 Agents. Ne pas prendre les anciennes pistes pour des tâches actives.

Statut : proposition, non décidée et non implémentée.

## Problème proposé

Aider un technicien à décider si une évolution de débit et de pression mérite une investigation de fuite. Ne pas promettre de localiser un point de fuite, d’inférer une cause certaine ou de prouver l’absence de fuite.

Hypothèse : l’entreprise utilise ou peut exporter des mesures temporelles de débit/pression. À confirmer ; si elle travaille uniquement avec de l’acoustique, il faudra choisir une autre représentation et un dataset adapté.

## Périmètre hackathon

Importer une fenêtre de mesures avec unités, horodatage et métadonnées autorisées. Un TSLM fine-tuné produit un statut limité (normal/suspect ou abstention calibrée), un intervalle à inspecter si les labels le permettent et une synthèse courte. Les statistiques et valeurs citées sont vérifiées par du code déterministe. Les conseils éventuels viennent d’une checklist validée par le professionnel, pas d’une cause physique inventée par le modèle.

Commencer par la décision au niveau d’une fenêtre ; localisation temporelle seulement si l’entraînement et les annotations le permettent. Aucune localisation géographique dans le MVP.

## Données et blocage principal

Source consultée : https://github.com/KIOS-Research/LeakDB
Dataset lié : https://zenodo.org/records/13985057

Le README précise que LeakDB contient des scénarios de fuite créés artificiellement sur des réseaux de distribution d’eau. C’est une piste de développement contrôlé, pas une preuve terrain. Formats détaillés, licence et fichiers non encore inspectés. Le brief demande une démo avec entrées réelles : ne pas considérer LeakDB seul comme satisfaisant ce critère sans clarification des organisateurs.

Voie préférée : dataset public de mesures réelles avec labels exploitables. Des mesures de l’entreprise, fournies avec autorisation et sans données client sensibles, pourraient servir à une démonstration externe séparée ; elles ne remplacent pas l’exigence de données ouvertes pour l’entraînement. Leur existence n’est pas confirmée.

## Évaluation proposée

Baseline simple : seuil sur débit persistant ou anomalie par rapport au profil attendu, calibré sur le train/validation. Baseline apprise si nécessaire : classifieur sur statistiques et texte par gabarit. Comparer au TSLM avant/après fine-tuning.

Mesurer précision/rappel fuite, fausses alertes par durée observée et, si disponible, retard de détection. Ne pas juger la qualité sur un texte convaincant. Réserver les scénarios/sites/périodes avant de créer les fenêtres et exclure chevauchements et labels des entrées. Définir explicitement le régime de généralisation ; pas de revendication nouveau réseau si testé sur le même réseau.

## Démo

Rejouer un cas normal et un cas annoté, montrer les courbes, la décision, les éléments mesurables et le résultat réel. Afficher clairement simulé/mesuré, période observée et limites. Aucun résultat de modèle disponible à ce stade.

## Conditions de poursuite

1. Le professionnel confirme un besoin et le type de capteurs.
2. Des séries ouvertes utilisables et des labels sont accessibles.
3. Une baseline et un split crédibles sont définis.
4. Le test d’intégration TimeNet → entraînement court → inférence fonctionne avant toute interface ambitieuse.

Prochaine question à Icham : l’entreprise exploite-t-elle débit/pression, ou plutôt de l’écoute acoustique sur place ?
