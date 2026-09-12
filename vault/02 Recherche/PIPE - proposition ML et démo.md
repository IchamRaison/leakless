# PIPE - proposition ML et démo

Icham

> Proposition initiale conservée pour ses recherches et sources. Le plan courant est [[Plan directeur agents]]. Pour la démo, suivre [[Démo et pitch]] et [[Protocole évaluation]] : exemples de développement/démo autorisés, test final scellé réservé à l'évaluation. Les étapes et variantes ci-dessous ne remplacent pas ces règles.

## Statut et recommandation

Proposition après recherche, à valider par l'équipe. Aucun entraînement ni interface exécutés. Nom de travail PIPE, disponibilité de marque non recherchée.

Construire un atelier d'analyse acoustique de fuites : le jury écoute un signal réel, voit son spectrogramme, puis un TSLM entraîné classe fuite/non-fuite et décrit des propriétés mesurables. L'interface expose les preuves, les erreurs, et un test de robustesse interactif. Pas de carte de localisation inventée.

## Pourquoi cette direction

L'utilisateur demande un impact visuel et une contribution ML sérieuse. Une animation de réseau géographique ne serait pas justifiée par les données réelles trouvées. Le signal acoustique permet une démo sonore et visuelle sans matériel supplémentaire. L'audio est traité comme séries numériques temporelles, pas comme transcription et pas uniquement comme image pour un VLM.

## Données vérifiées

Source primaire : https://zenodo.org/records/18631450
API relue par l'agent principal : https://zenodo.org/api/records/18631450
Licence déclarée : CC BY 4.0.

La description annonce 1 000 clips d'une seconde : 500 fuite, 386 sans fuite, 114 bruits environnementaux. Les deux premières catégories viennent d'une base extérieure de formation à la détection de fuites à Dongguan. Certains bruits environnementaux proviennent d'un autre site public. Ce sont des mesures physiques sur une base expérimentale, pas une validation chez des clients.

Archive fuite téléchargée et listée par l'agent principal : 500 WAV, noms encodant matériau, région, pression, vitesse d'écoulement et appareil. Nombreux suffixes _1/_2 et captures hydrophone/logger sous mêmes conditions : risque de dépendance évident. Archives sans fuite et bruit encore à auditer directement avant split.

Fichiers :
- https://zenodo.org/api/records/18631450/files/leak%20acoustic%20data.rar/content
- https://zenodo.org/api/records/18631450/files/no%20leak%20acoustic%20data.rar/content
- https://zenodo.org/api/records/18631450/files/environmental%20noise.rar/content

Ne pas donner noms de fichiers, labels, identifiants de groupe ou métadonnées révélant la cible au modèle. Ne pas interpréter m/s comme débit volumique. Ne pas prétendre connaître les unités physiques de l'amplitude WAV sans calibration.

## Démo visée

1. Deux clips réservés au test, labels masqués, lecture audio et spectrogramme synchronisés. La lecture en boucle éventuelle est indiquée, pas présentée comme un enregistrement continu.
2. Le jury fait son choix, puis lance une vraie inférence du TSLM.
3. Afficher prédiction et courte description spectrale, confrontées au label après l'inférence.
4. Mode robustesse : mélange contrôlé de bruit inédit pour le modèle à un clip de test, avec intensité réglable. Mélange affiché comme perturbation synthétique d'un signal réel. Réexécuter le modèle ; abstention souhaitée mais non garantie et jamais scriptée.
5. Afficher baseline vs TSLM, erreurs et courbe précision/couverture. Les scores viennent de l'évaluation réelle, jamais du scénario de démonstration.

Le MVP n'est pas une localisation géographique ni une prédiction de rupture. Une carte de tuyaux serait au mieux illustrative et n'est pas nécessaire.

## Contribution ML minimale

- Connecteur TimeNet réel vers TimeF, avec signaux, provenance, annotations réservées aux cibles et tâches classification/réponse textuelle.
- Transformer les WAV en séries temporelles d'énergie par bandes fréquentielles, en gardant leur évolution dans le temps. Paramètres ajustés sur le train uniquement. Conserver les WAV originaux pour écoute et vérification.
- OpenTSLM-SP avec petit modèle de base accessible (Llama 3.2 1B ou Gemma supporté), entraînement effectif des composants temporels/projecteur, adaptation légère si nécessaire. Ne pas faire du TSLM un simple rédacteur recevant la réponse d'un autre classifieur.
- Cible : classe + propriétés de signal calculables. Générer les descriptions depuis mesures et labels du train, déclarer leur nature non experte. Pas de cause physique ou recommandation de réparation inventée.
- Baseline forte : caractéristiques spectrales + SVM/Random Forest ; si possible petit CNN. Même information disponible et mêmes splits.
- Comparer TSLM avant/après adaptation et baseline. Ablation métadonnées seules, puis caractéristiques agrégées vs séries temporelles pour tester la valeur de l'évolution temporelle. Une permutation temporelle peut ne pas affecter un signal stationnaire ; c'est une conclusion possible, pas un test censé réussir à tout prix.
- Mesurer macro-F1, rappel fuite, fausses alertes, exactitude des mesures textuelles, robustesse sous bruit. Abstention fondée sur score calibré sur validation, pas une confiance librement inventée dans le texte.

## Condition scientifique impérative

Pas de split aléatoire clip par clip. Grouper les captures liées à un même événement/conditions, même si plusieurs matériaux ou appareils le mesurent. Dédupliquer exactement puis rechercher les quasi-doublons. Réserver groupes et appareils lorsque support suffisant, en déclarant la portée réelle du test.

Les noms ne garantissent pas une reconstruction parfaite des sessions. Si l'indépendance des événements reste inconnue, la documenter, demander aux auteurs, et limiter les revendications. Un score élevé obtenu en reconnaissant une session n'est pas un résultat défendable.

## Alternatives étudiées

- Intra-Domestic Water Leaks : https://github.com/rzese/Intra-Domestic-Water-Leaks-Dataset . README relu par l'agent principal : consommation réelle résidentielle à pas 5 minutes, 40 428 journées annotées. Pas d'identifiants logement documentés dans la version inspectée : séparation par usager non établie. Le CSV signalé par la recherche contient 288 colonnes label compris alors qu'une journée à pas 5 min suggère 288 mesures ; format à clarifier avant usage. Moins adapté à un test hors-logement rigoureux sans métadonnées complémentaires.
- LeakDB : https://github.com/KIOS-Research/LeakDB . Simulé explicitement. Utile si les organisateurs acceptent ce cadre, pas de preuve terrain.
- BattLeDIM : https://github.com/KIOS-Research/BattLeDIM . Détection/localisation de réseau déjà étudiée ; ne pas revendiquer une nouveauté générale ni présenter les scénarios comme mesures réelles.

## Plan de validation avant engagement

1. Audit des trois archives et des groupes : signal exploitable, labels, indépendance et licence.
2. Baseline sur split groupé. Confirmer avec organisateurs que cette représentation acoustique en séries et le périmètre d'entraînement répondent au challenge.
3. Test TimeNet → chargement OpenTSLM-SP → quelques mises à jour → checkpoint → inférence. Mesurer temps et mémoire ; aucun budget/temps d'entraînement garanti à ce stade.
4. Seulement après ce test, lancer entraînement complet et frontend sonore/spectrogramme en parallèle.

Si le dataset ne permet pas une séparation défendable, chercher une source mieux documentée ou adopter explicitement une preuve sur simulation avec accord du jury. Ne pas maquiller ce blocage par une belle interface.

## Sources techniques

- https://github.com/OpenTSLM/OpenTSLM
- https://github.com/OpenTSLM/TimeNet
- https://docs.timenet.ai/concepts.html

TimeNet et entraînement TSLM restent obligatoires pour notre soumission. Une solution prompt-only est un prototype de secours, pas un livrable conforme. La détection acoustique par ML existe déjà ; aucune nouveauté algorithmique établie. La différenciation proposée est une expérience d'analyse vérifiable, la robustesse et le connecteur réutilisable.
