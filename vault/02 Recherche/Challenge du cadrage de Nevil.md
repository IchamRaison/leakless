# Challenge du cadrage de Nevil

Icham

2026-09-12. Analyse demandée par Icham à partir du texte de Nevil sur Why → How → What. Statut : recommandations à discuter, pas changement des contrats ni protocole adopté. Aucun audit d'archives, entretien utilisateur ou entraînement exécuté dans cette étape.

## Avis

Conserver le cadrage par l'utilisateur et sa décision. Le Golden Circle organise le récit ; il ne démontre ni le besoin ni la pertinence du TSLM. Distinguer trois preuves : reconnaître les classes du dataset, produire des descriptions exactes, aider une personne dans son travail. La première ne suffit pas à établir les deux autres.

TimeNet et entraînement TSLM sont des contraintes du challenge. Leur nécessité dans un futur produit reste une question expérimentale. Le présent échange ne vaut pas validation du nom LeakLess.

## 1. Définir le travail humain avant de promettre l'inspection

« Technicien acoustique » et « facilities manager » désignent deux utilisateurs possibles, dont le travail doit être précisé. Recommandation pour le MVP : un technicien qui dispose déjà d'un enregistrement et doit le relire/documenter. Faire décrire par le professionnel un cas récent : origine du signal, outil utilisé, hésitation, action suivante, temps passé et conséquence d'une erreur. Le besoin reste une hypothèse tant que cet échange n'a pas eu lieu.

La fiche Zenodo décrit des clips d'une seconde, avec des labels d'acquisition. Elle ne fournit pas dans sa description de cible « inspection nécessaire » ni de suivi avant apparition visible d'une fuite. Ces limites empêchent d'utiliser ce benchmark seul comme preuve d'alerte précoce ou de meilleure décision d'intervention. [Source du dataset](https://zenodo.org/records/18631450).

## 2. Les trois dossiers ne fixent pas automatiquement trois classes produit

La même source indique que la majorité des bruits environnementaux vient de dlmeasure.com et une petite partie du site expérimental. Il faut distinguer ces provenances dans l'audit. Hypothèse de risque à tester : un classifieur pourrait reconnaître le domaine d'enregistrement plutôt qu'une propriété de fuite. Un split groupé ne supprime pas à lui seul une confusion entre provenance et classe.

Un enregistrement de fuite peut aussi contenir du bruit : « bruit » n'est pas nécessairement un état physique exclusif. Conserver pour l'instant la tâche binaire fuite/non-fuite du site expérimental et une analyse séparée des bruits/perturbations, conformément à [[Protocole évaluation]]. Une variante trois classes doit être décidée explicitement après G0 et coordonnée avec les consommateurs de [[Contrats techniques]].

## 3. Corriger l'ambiguïté du flux texte → modèle

Le chemin d'inférence attendu reste WAV → TimeNet/TimeF → transformation numérique partagée → TSLM → classe et description. Les descriptions cibles construites depuis les mesures et labels servent à la supervision ; elles ne sont pas insérées dans le prompt d'inférence.

Injecter le label dans le texte est une fuite directe. Donner en entrée les mesures exactes ensuite notées comme texte généré rend ce test essentiellement copiable ; ce n'est pas automatiquement une fuite du label, mais cela ne prouve pas leur extraction depuis le signal. Les mesures DSP de contrôle restent identifiées séparément des assertions du modèle.

## 4. Comparer au bon système simple

L'exemple « baseline avec un score » contre « TSLM avec une phrase riche » ne permet pas d'attribuer une utilité au TSLM. Un gabarit peut aussi présenter les mêmes propriétés mesurées.

Comparaison proposée, avec même audio, mêmes visualisations et même accès aux données :

| Variante | Fonction de la comparaison |
| --- | --- |
| Random Forest + classe/score | Référence de classification déjà prévue |
| Même Random Forest + mesures DSP + compte rendu par gabarit | Isoler l'apport d'une restitution lisible sans TSLM |
| TSLM adapté + description générée, mesures DSP séparées | Tester l'apport supplémentaire du modèle temporel et du langage |

Le gabarit est une restitution de faits, jamais une prétendue explication interne de la Random Forest. Cette variante ne nécessite pas un nouvel entraînement de la baseline. Son ajout à l'application doit être convenu avec Safoan ; il reste proposé ici.

Comparer rappel fuite, faux positifs, macro-F1, sorties invalides, fidélité des valeurs aux mesures et latence. Fixer sur validation les critères/tolérances avant le test final. Une différence illustrative de 92 % contre 91 % ne se juge pas sans métrique précisée, effectifs indépendants et incertitude. Aucun de ces nombres n'est un résultat PIPE.

## 5. Une description exacte ne prouve pas le raisonnement du modèle

Mesurer l'énergie d'une bande et vérifier le nombre cité établit une exactitude factuelle. Cela ne démontre pas que cette bande a causé la prédiction. La distinction entre plausibilité et fidélité au mécanisme de décision est explicitée par [Jacovi et Goldberg, ACL 2020](https://aclanthology.org/2020.acl-main.386/).

Préférer « description acoustique vérifiable » à « explication de sa décision ». Toute référence « énergie élevée » doit préciser population de comparaison issue du train, normalisation, bande et seuil ; toute confiance/abstention doit suivre le protocole de validation. Un texte prudent n'est pas une calibration.

L'intérêt propre de l'ordre temporel reste également à tester par les ablations prévues. [OpenTSLM](https://arxiv.org/abs/2510.02410) présente des résultats sur activité humaine, sommeil et ECG ; cette référence n'établit pas la performance sur nos clips de canalisations.

## 6. Tester l'utilité avec une personne

Après une boucle technique réelle, proposer un petit essai exploratoire au professionnel : variante baseline enrichie contre TSLM, identité des modèles masquée, cas distincts comparables et ordre contrebalancé. Utiliser le lot de développement/démo autorisé, sans ouvrir les labels du test final pour régler le produit.

Mesurer temps de revue/rédaction, erreurs factuelles du compte rendu et capacité à repérer les contradictions effectivement présentes. Demander quelle action suivrait, tout en signalant l'absence de vérité terrain sur la décision d'inspection. Un retour positif ou un petit pilote ne constitue pas une validation terrain. Si aucun professionnel n'est disponible, publier seulement les résultats techniques et laisser l'utilité humaine non évaluée.

## Formulation proposée après retour de G0

Hypothèse produit : aider un technicien à relire et documenter un enregistrement acoustique de canalisation pour préparer une investigation.

Question expérimentale : sur des groupes d'acquisition réservés, un TSLM peut-il distinguer fuite/non-fuite et produire des descriptions acoustiques vérifiables ? Sa restitution aide-t-elle davantage à la revue qu'une Random Forest accompagnée des mêmes mesures et d'un texte déterministe ? Examiner séparément les bruits environnementaux.

## Prochaine action

Nevil poursuit G0 : audit des trois archives, provenances et groupes réellement défendables. Icham confronte le besoin au cas concret du professionnel, puis l'équipe fige le problème, les classes et les critères de comparaison avant l'entraînement destiné à produire les résultats. L'inventaire runtime et les tests mécaniques de chargement I0 peuvent avancer sans modifier ce cadrage. Aucun nouveau lancement GPU ni changement de chantier des coéquipiers effectué par cette analyse.

Liens : [[Plan directeur agents]], [[Protocole évaluation]], [[Contrats techniques]], [[Journal Icham]].
