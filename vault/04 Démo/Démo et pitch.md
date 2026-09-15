# Démo et pitch

Icham

Plan cible ; V0 ML exécutée/rechargée, application non intégrée. Durée officielle à confirmer. Propriétaire technique Safoan, défense ML Icham/Nevil, baseline Vincent. Le pitch reste partagé.

## Nouveau scénario proposé — surveillance automatique

[[Plan surveillance continue]] remplace l'import manuel comme parcours principal : un canal est surveillé, un événement apparaît, l'alerte s'ouvre avec preuves audio/spectrogramme, puis sa persistance/fin et une panne de flux sont visibles. Utiliser un replay déclaré et une inférence réellement exécutée ; une séquence assemblée est un scénario artificiel, pas une capture terrain. Les règles d'alerte ne doivent jamais lire le label caché ou l'horloge d'un scénario pour fabriquer une détection. Le texte peut arriver après l'alerte et reste distinct des mesures.

Montrer séparément résultats de classification et démonstration de fonctionnement continu ; pas de délai réel ni fausses alertes/jour sans acquisitions adaptées. En cas de résultat faible ou d'absence d'alerte, le montrer. Le parcours d'import ci-dessous reste disponible pour investigation et comparaison, pas comme justification du nouveau cas d'usage.

## Histoire

Un technicien doit interpréter un court enregistrement acoustique de canalisation. PIPE lui donne une prédiction, des observations mesurables et une comparaison claire ; aucune promesse de localisation géographique ou de diagnostic client garanti.

## Scénario principal

1. Faire écouter deux clips réels de développement/démo autorisés avec labels masqués. Afficher source expérimentale et durée réelle.
2. Le jury propose son choix ; lancer l'inférence réelle du TSLM. Montrer son et spectrogramme liés au même input_hash.
3. Afficher classe et description, puis révéler le label. Les observations DSP sont distinctes du texte généré.
4. Si P1 validé, ajouter bruit contrôlé et réexécuter le modèle. Indiquer perturbation synthétique ; ne pas scénariser la confiance ni garantir l'abstention.
5. Afficher comparaison globale au test réservé, support et limites. Le modèle peut perdre contre la baseline : résultat présenté honnêtement.
6. Montrer preuve TimeNet, configuration d'entraînement et checkpoint ; expliquer une limite réelle et la suite vers validation terrain.

## Visuels autorisés

Waveform, spectrogramme avec axes, scores correctement libellés, matrice de confusion, performance sous bruit, version/run. Pas de réseau géographique fictif présenté comme localisation, pas de causalité dérivée d'une attention map non validée, pas de compteur d'eau économisée sans mesure.

## Repli

Replay de véritables sorties enregistrées, avec badge REPLAY, date/run/hash et explication de la panne live. Vidéo après une exécution réussie. Si aucun modèle n'a été entraîné, dire que le critère d'entraînement n'est pas rempli ; aucune vidéo ou API de langage ne peut le remplacer.

## Checklist livraison

- [ ] Entrées réelles traçables/licenciées et labels séparés des prompts.
- [ ] Démo de bout en bout testée, erreurs gérées, audio audible.
- [ ] Config/data manifest/split/revisions et commandes de reproduction.
- [ ] Checkpoint complet téléchargé/rechargé et licence respectée.
- [ ] Baseline et évaluation courte avec support, limites et erreurs.
- [ ] Connexion TimeNet effectivement exercée.
- [ ] Résultats cohérents avec la version soumise.
- [ ] Aucun secret/donnée client à l'écran ou dans les artefacts.
- [ ] Format/deadline confirmés, lien de soumission lu après envoi.
