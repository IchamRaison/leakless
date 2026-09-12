# Plan de session Icham - première version TSLM

Icham

Statut au 2026-09-12 : plan validé explicitement par Icham (« je valide »), puis démarrage explicitement différé (« Commence pas »). La validation du plan n'autorise donc pas encore l'implémentation, l'installation ou une dépense. Ce document précise la fiche [[Agent Icham - ML]], sans remplacer les contrats d'équipe ni l'audit de Nevil ; les choix techniques conditionnels restent à finaliser.

## Objectif et deux livraisons distinctes

Objectif immédiat : conclure le cadrage ML, choisir une stratégie vérifiable et rendre les dépendances explicites. Après accord de démarrage, viser rapidement un modèle réellement adapté qui transforme les séries acoustiques en classe et texte court.

- V0 technique : un apprentissage de contrôle fonctionne, le checkpoint complet recharge dans un processus neuf et une vraie inférence respecte Prediction. Safoan peut intégrer le service. Cette étape ne prouve pas une qualité utile ni une généralisation.
- V1 candidate : entraînement sur le train autorisé, sélection sur validation groupée, descriptions contrôlées et résultats/latence/limites publiés. C'est la première version destinée à être jugée sur sa qualité, sans garantie préalable de battre la baseline.
- Version finale : modèle et réglages gelés, évaluation indépendante par Nevil sur le test réservé, intégration et reproduction vérifiées. Le test final n'est pas nécessaire pour commencer l'intégration de V0.

La session vise V0 puis V1 si les prérequis sont disponibles. Sans données ou accès, une preuve technique partielle doit être nommée comme telle ; aucun succès scientifique ni délai garanti.

## 1. Terminer les décisions structurantes avant le code

Acquis : priorité au texte utile ; résultat fonctionnel et défendable en cas de temps court ; première livraison rapide puis amélioration ; vitesse sans sacrifier la précision ; aucun plafond budgétaire actuellement fixé. Ces préférences ne fixent ni métrique minimale, ni crédits réels, ni machine.

Périmètre validé et détails restant à consigner ensemble :

- Sortie V1 : classe et une propriété acoustique vérifiable au premier jalon, puis enrichissement ; compte rendu en une passe, pas de chat libre. Périmètre accepté avec le plan, propriété exacte et schéma détaillé encore à convenir.
- Propriété : la choisir avec Nevil parmi celles mesurables sur la représentation réellement fournie au modèle. Définir unités, règle de calcul, seuils éventuels et tolérances. Ne pas demander au modèle de reconstruire une grandeur éliminée par normalisation.
- Classes : résoudre le contrat binaire versus la nouvelle direction à trois classes après l'audit ; recommandation du protocole courant, comparaison fuite/non-fuite du même site et challenge de bruits externes séparé. Pas de décision unilatérale pendant cette session.
- Qualité : définir comment arbitrer classification, exactitude descriptive et latence ; ne pas choisir sur la fluidité du texte seule. Établir les règles de sélection sur validation avant les essais comparatifs.
- Incertitude : pas de pourcentage généré présenté comme confiance. Recommandation de départ : erreur explicite si sortie invalide ; abstention sur ambiguïté uniquement si score/règle évalués avec Nevil, sinon pas de promesse de détection fiable des cas incertains.
- Moyens : choix opérationnel de la machine, allocation, persistance, durée maximale de run et arrêt à valider avant lancement payant.

Sortie : courte spécification ML convenue, critères de réussite, hypothèses non résolues et accord distinct pour commencer. Les faits techniques vérifiables sont recherchés par l'agent ; Icham arbitre les choix et ne doit pas deviner les API ou hyperparamètres.

## 2. Choisir et vérifier le chemin technique

Orientation proposée, déjà présente dans la fiche ML : OpenTSLM-SP, base supportée et accessible ; examiner Llama 3.2 1B comme candidat, pas choix définitif. Vérifier aussi les checkpoints temporels disponibles, ce qu'ils contiennent et ce qui doit être adapté à l'acoustique. Ne pas confondre poids du LLM et checkpoint TSLM complet.

Le README officiel https://github.com/OpenTSLM/OpenTSLM documente SP, des bases Llama/Gemma et des scripts d'adaptation, mais pas une performance PIPE. Lecture amont vérifiée le 2026-09-12 ; révision et accès effectifs à figer lors de l'exécution.

Proposition d'apprentissage initiale : adapter encodeur temporel et projecteur ; garder le LLM gelé si cette voie est supportée. Définir et contrôler les paramètres entraînables. Étudier LoRA ou une base différente seulement si les premiers résultats révèlent une limitation correspondante.

Après accord : vérifier l'environnement existant, créer uniquement ce qui manque, tester accès aux poids et runtime GPU, figer versions et mesurer mémoire/temps du premier batch. L'inventaire antérieur n'a pas identifié de GPU NVIDIA local ; l'accès SSH à une H100 80 Go distante est désormais vérifié, sans installation ni entraînement. Détails dans [[Journal Icham]].

Sortie : environnement reproductible, modèle chargeable, liste de paramètres entraînables et configuration de premier essai. Le nombre d'epochs, le batch et les durées restent à mesurer, pas à inventer maintenant.

## 3. Verrouiller le contrat avec les autres rôles

Nevil fournit audit/groupes/split, un vrai exemple de développement autorisé, round-trip TimeNet/TimeF, transformation partagée [C,T], versions/normalisation, labels et propriétés de contrôle. C représente les bandes fréquentielles et T leur évolution temporelle. L'audit ne doit pas être contourné pour produire un score.

Safoan fournit/convient de Prediction, signature d'appel, erreurs et limites d'entrée. Vincent utilise les mêmes données autorisées et le même split pour son comparateur. Leurs implémentations ne deviennent pas la propriété d'Icham.

Séparer explicitement :

- Entrées d'inférence : séries numériques et instructions/descriptions de canaux non révélatrices.
- Cibles d'apprentissage : classe et texte fabriqué depuis les propriétés déterministes, sans prétendre à une annotation experte.
- Contrôle : mesures DSP et labels utilisés par l'évaluateur, pas réponses à recopier dans le prompt.

Sortie : un exemple réel commun et contrats convenus à G1. Les modifications d'interfaces impliquent leurs consommateurs ; pas de migration implicite de classes ou de schéma.

## 4. Construire la boucle minimale et livrer V0

Après les accords nécessaires : adapter un exemple TimeNet au format exact OpenTSLM, vérifier formes/dtypes/padding/masques et séparation prompt/cible. Réutiliser le prétraitement de Nevil et les mécanismes amont plutôt que les dupliquer.

Exécuter un forward puis un petit apprentissage de contrôle sur un sous-ensemble train autorisé. Vérifier perte finie, gradients et changement des poids attendus. Un essai de mémorisation de quelques exemples est du debug, jamais un score de validation. Les essais synthétiques éventuels ne prouvent que la mécanique.

Sauvegarder tous les composants modifiés et configurations nécessaires, fermer le processus, recharger et inférer réellement sur un exemple de développement non utilisé dans cet essai. Publier aussi les sorties invalides et les erreurs ; ne pas les remplacer avec le vrai label.

Livraison V0 à Safoan : fonction de prédiction réelle, schéma/version, exigences runtime et gestion de l'indisponibilité. Étiqueter explicitement cette version comme intégration technique, qualité non validée. Aucun notebook vivant requis.

Sortie : preuves G2, checkpoint complet avec checksum, commande de reload/inférence et tests minimaux exécutés.

## 5. Entraîner et évaluer la première V1 candidate

Entraîner sur train, choisir les paramètres sur validation groupée, ne pas ouvrir le test final. Comparer au modèle initial si comparable et à la baseline de Vincent lorsqu'elle est disponible. Même split et mêmes entrées autorisées.

Mesurer avec Nevil : macro-F1, précision/rappel fuite, faux positifs, matrice de confusion, effectifs et invalidités ; exactitude de la propriété textuelle selon sa définition, assertions non supportées et taux de sorties exploitables ; latence réelle et mémoire. Distinguer démarrage du service et requête avec modèle déjà chargé.

Recommander aussi un contrôle baseline + mesures/gabarit pour juger la valeur ajoutée du texte appris ; cette extension du comparateur reste à convenir avec l'équipe. Une description exacte n'est ni une preuve de raisonnement causal ni une preuve d'utilité métier ; quelques relectures de clips autorisés par un technicien peuvent fournir un premier retour exploratoire si un participant est disponible.

Sortie : V1 candidate, tableau de résultats reproductibles et limites. Si aucune version n'a une qualité défendable, le constater et garder V0 uniquement comme preuve technique.

## 6. Améliorer en fonction des erreurs observées

Choisir un petit nombre d'expériences répondant à des causes précises : représentation temporelle, réglages d'apprentissage, qualité des cibles, capacité d'adaptation du modèle. Pour chaque essai : hypothèse, changement, coût observé et résultat sur validation.

Avec Nevil, utiliser les perturbations et ablations prévues pour tester bruit et apport de l'ordre temporel. Un modèle plus grand, LoRA ou davantage de propriétés ne sont pas des objectifs en soi. Les adopter s'ils corrigent une limite mesurée. Garder une version antérieure utilisable et ne pas retarder son intégration.

Optimiser ensuite la latence à qualité comparable ; sortie courte et chargement unique du modèle sont les premiers choix prévus. Pas de gain de qualité supposé à partir de la seule taille du modèle.

Sortie : version retenue et raisons factuelles, avec les expériences négatives conservées.

## 7. Intégrer, geler et transmettre

Avec Safoan : vraie requête clip → prétraitement → checkpoint adapté → Prediction → affichage, plus panne contrôlée et erreurs explicites. Audio visualisé et signal analysé doivent correspondre ; séparer mesures DSP, texte généré et label de démonstration. Une baseline ou un replay ne sont jamais étiquetés TSLM live.

Figer modèle, prétraitement, génération et seuils éventuels. Transmettre à Nevil pour l'évaluation finale réservée ; ne pas reprendre l'optimisation sur ce test. Livrer code/configurations, environnement, poids autorisés hors Git, checksums, prédictions, résultats, procédure de reproduction et besoins matériels mesurés.

Sortie : modèle intégrable/reproductible et niveau de preuve honnêtement nommé. Réserver le dernier tiers du temps effectivement disponible à l'évaluation/intégration, conformément au premier tour accepté ; estimer les durées après le premier batch, sans horaire fictif.

## Dépendances et conditions d'arrêt

- Données retardées : accès, lecture amont et préparation runtime peuvent avancer après accord ; pas d'entraînement métier ni score scientifique sans données/split autorisés.
- Accès GPU/poids absent : vérifier les alternatives supportées ; ne pas louer silencieusement une machine ni déduire un accès du voucher annoncé.
- Modèle qui n'apprend pas : examiner données, cibles, pertes, masques et gradients avant d'augmenter la capacité ou la durée.
- Texte faux : revoir propriété/cibles/apprentissage, publier le taux d'erreur ; pas de correction cachée par le vrai label ou les mesures pour embellir le score TSLM.
- Temps court : réduire essais et richesse du texte, préserver TimeNet, entraînement réel, reload, comparaison et protocole honnête.

Continuité : à chaque résultat significatif, mettre à jour [[Journal Icham]], [[Passation]] et les notes concernées, intégrer/pousser le vault source puis rafraîchir le miroir. Préserver les fichiers d'outillage non suivis et les changements d'autres participants. Aucun envoi de message ou lancement externe n'est effectué par la simple rédaction de ce plan.
