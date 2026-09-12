# Protocole évaluation

Icham

Propriétaire : Nevil ; baseline fournie par Vincent ; TSLM par Icham. Statut : protocole à implémenter avant optimisation.

## Question testée

Dans le domaine expérimental accessible, le modèle distingue-t-il les clips fuite/non-fuite sur des groupes réservés, et reste-t-il fiable sous perturbation ? La réponse n'établit ni performance client, ni localisation, ni cause physique de la fuite.

## 1. Audit et split avant tout apprentissage

Télécharger les trois archives ; établir effectifs, sample rates, amplitudes, durées, anomalies et licences. Ne pas prendre une déclaration de l'archive pour une vérification complète. Dédoublonner par hash du signal décodé, pas seulement du fichier ; rechercher quasi-doublons/corrélations et captures concomitantes.

Les noms contiennent conditions, appareils et suffixes. Regrouper les captures d'un même événement supposé, y compris plusieurs capteurs/matériaux si le même événement les alimente. Documenter les heuristiques et leurs échecs. Des noms identiques en pression/vitesse ne prouvent pas une même session ; inversement des noms différents ne prouvent pas l'indépendance.

Construire les splits par groupe et vérifier que chaque classe est représentée. Choisir les proportions selon le nombre de groupes réellement indépendants ; ne pas forcer 70/15/15 si cela invalide les classes. Publier effectifs clips ET groupes. Mettre les fichiers ambigus en quarantaine ou limiter explicitement la revendication, décision avec Icham.

Le test principal recommandé oppose fuite et non-fuite du site expérimental. Les bruits environnementaux externes servent d'abord de challenge hors-domaine séparé ; ne pas gonfler la performance en ajoutant des négatifs faciles d'une autre provenance. Une variante qui les inclut doit être nommée séparément.

## 2. Développement et test final

Train pour fit, validation pour choix de paramètres et seuils, test final scellé jusqu'au gel. La baseline et le TSLM partagent IDs et cibles. Nevil contrôle les labels du test final ; partager uniquement un petit lot de démonstration explicitement séparé du test scellé pour construire l'UI.

Pour un petit nombre de groupes, considérer validation croisée groupée sur le développement, puis test final indépendant si possible. Évaluer un appareil/matériau non vu en test complémentaire seulement si le support le permet. Aucun score de généralisation à un nouveau site avec un seul site.

## 3. Comparateurs

- Majoritaire comme sanity check, pas comme seule baseline.
- Random Forest de Vincent sur features spectrales, parameters/seed enregistrés.
- TSLM initial avant adaptation, si capable de produire des sorties comparables ; invalidités comptées.
- TSLM adapté.
- Option si temps : CNN acoustique petit ; ne pas ajouter avant G2/G3.

Même prétraitement source et budget d'information. Hyperparamètres ajustés uniquement sur validation. Ne pas sélectionner la meilleure seed sur test.

## 4. Métriques

Macro-F1, précision et rappel fuite, matrice de confusion et taux de faux positifs parmi clips sans fuite. Inclure effectifs, invalidités et abstentions. Ne pas annoncer fausses alertes par jour sur des clips isolés sans chronologie continue. Ne pas annoncer délai d'apparition de fuite sans annotation d'onset.

Scores d'abstention : couverture et erreur parmi les prédictions retenues, avec performance globale incluant les non-réponses. Choisir seuil sur validation ; pas de confiance verbale auto-déclarée. Si comparaison entre modèles, tenir compte de couvertures différentes.

Rapporter incertitude par bootstrap de groupes ou autre méthode justifiée quand taille suffisante ; les clips corrélés ne sont pas des observations indépendantes. Publier limites si groupes trop peu nombreux.

Texte : taux de sorties valides, exactitude des propriétés numériques dans tolérances fixées avant scoring, assertions non supportées. Définir extraction et unités. Ne pas noter uniquement BLEU/ROUGE, longueur ou apparence convaincante.

## 5. Robustesse et ablations

Fixer sur validation la grille de bruit puis conserver un pool de bruit inédit pour test. Pour chaque perturbation : sample_id parent, noise_id, seed, SNR demandé/réalisé, input hash, modèle, prédiction et runtime. Reporter rappel/FPR/couverture par niveau. Le signal perturbé est une augmentation synthétique, clairement identifiée.

Tester agrégats vs évolution temporelle pour étudier si la dimension temporelle apporte une information utile. Une permutation de l'ordre peut ne pas changer un signal stationnaire ; ce résultat est recevable. Ne pas prétendre que la heatmap d'attention explique causalement la décision.

Audit shortcut : entraînement/prediction sans noms ni métadonnées sources, sondage facultatif métadonnées seules pour révéler confusions, tests de changement de gain pour voir si simple amplitude suffit. Distinguer ablation d'analyse et version du modèle soumise.

## 6. Artefacts d'un run

run_id, commit, dataset version, manifest hash, split hash, config hash, model hash, preprocessing version, seeds, environnement, commandes, logs, predictions.jsonl et metrics.json. Chaque score affiché renvoie à un run. Relancer le scoring depuis les prédictions sauvegardées doit reproduire les métriques.

Ne jamais remplacer une sortie ratée par une bonne prédiction inventée. Exemples visuels issus d'un lot démo autorisé, pas cherry-picking caché sur le test final. Montrer au moins une limite réelle dans la présentation si disponible, sans inventer un échec pour le spectacle.

## Acceptation

- [ ] Doublons/groupes/splits audités et limites approuvées.
- [ ] Baseline et TSLM évalués sur mêmes exemples.
- [ ] Test non utilisé pour réglages ; seuils gelés.
- [ ] Métriques reproductibles, effectifs et échecs inclus.
- [ ] Source expérimentale et perturbations synthétiques visibles.
- [ ] Aucun résultat clinique/terrain/localisation extrapolé.
