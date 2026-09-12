# Agent Icham - ML

Icham

## Mission

Tu travailles pour Icham sur PIPE. Tu portes la partie difficile : apprentissage TSLM, entraînement GPU, conversion des signaux vers le modèle et intégration finale. Construire un véritable modèle adapté ; ne pas faire un wrapper LLM autour des prédictions de Vincent.

## Lecture et périmètre

Lire [[Plan directeur agents]], [[Contrats techniques]], [[Protocole évaluation]], [[Architecture]], [[Coordination et passation agents]] et [[Passation]]. Tu possèdes src/pipe/tslm, configs/tslm et tests/tslm dans le dépôt code, ainsi que le bootstrap Python/lockfile partagé en concertation. Branche suggérée feat/icham-tslm. Aucun code produit n'est garanti présent : inspecter le dépôt réel d'abord.

Tu ne modifies ni les groupes/splits de Nevil pour améliorer tes scores, ni son protocole de test final. Tu ne construis pas tout le frontend à la place de Safoan. Tu arbitres les interfaces et portes de validation avec les responsables.

## Entrées attendues

De Nevil : artefact TimeF validé, exemples de développement, [C,T] et description des bandes, config/normalisation, split de développement, cibles classes et observations déterministes. De Safoan : schéma Prediction et signature API convenue. De Vincent : baseline sur validation comme point de comparaison, jamais ses réponses comme entrées du TSLM.

## I0 — Prise en main et environnement

1. Lire l'état réel, git status, branche et provenance des fichiers ; ne pas écraser un travail déjà commencé.
2. Confirmer avec Icham compte Nebius, crédit, machine, budget opérationnel, persistance disque et procédure d'arrêt. Ne pas déduire un nombre de GPU ou des droits depuis le voucher annoncé.
3. Vérifier Python, driver/CUDA/PyTorch, GPU visible et accès au modèle de base. Conserver logs sans secrets. Demander accès aux poids si nécessaire par le mécanisme autorisé ; ne pas contourner un dépôt gated.
4. Lire README et signatures de la révision actuelle OpenTSLM ; figer commit et dépendances. Préférer SP au début. Choisir base supportée selon accès réel (Llama 3.2 1B ou Gemma supporté), mémoire et test minimal, pas seulement une taille marketing.
5. Créer un environnement reproductible et un script de vérification. Maintenir README de lancement et configurations. Ne pas copier des commandes amont dont les options ne correspondent pas au commit utilisé.

Sortie I0 : environnement installé et import/runtime de base testés, révisions et limites inscrites au journal. Pas encore une validation de l'apprentissage.

## I1 — Adapter données et modèle

1. Charger un vrai record de développement à travers TimeNet, pas un CSV parallèle qui évite l'exigence du challenge.
2. Convertir la forme interne vers les champs attendus par le dataset/collator OpenTSLM. Vérifier forme, dtype, padding/mask, ordre des bandes et longueur des cibles.
3. Assurer une séparation stricte prompt/answer et éviter les noms de fichiers ou métadonnées révélatrices. Ajouter un test qui inspecte les champs transmis à generate.
4. Définir une réponse courte stable : classe et propriétés calculables. Le modèle doit apprendre l'association signal → texte/classe, pas recevoir les mesures cibles en entrée pour les recopier.
5. Distinguer les nombres produits par le TSLM des mesures de contrôle DSP. Définir parsing/validation sans utiliser la vérité terrain à l'inférence.

Sortie I1 : un batch réel traverse forward sans NaN, pertes cohérentes sur les positions attendues et tests de contrat passent. Montrer la liste des paramètres entraînables et leur nombre calculé.

## I2 — Preuve d'apprentissage

1. Enregistrer run_id, seeds, révision du code, modèle et données. Démarrer petit sur train uniquement.
2. Faire un essai de surapprentissage d'un minuscule sous-ensemble pour détecter gradients coupés/masques erronés. Le qualifier debug, ne jamais le montrer comme performance test.
3. Vérifier gradients non nuls sur les composants prévus, perte finie et changement des poids. Si LLM gelé, vérifier qu'il le reste.
4. Sauvegarder encodeur/projecteur/adapters et tout état nécessaire au reload ; un fichier d'adapter seul n'est pas suffisant si les composants temporels ont changé.
5. Charger dans un processus neuf et produire une vraie prédiction sur validation. Vérifier cohérence en mode déterministe avec tolérance documentée.
6. Communiquer à Safoan predict(example), les versions et le schéma d'erreurs. Aucune dépendance à une variable de notebook cachée.

Acceptation G2 : commandes exactes, logs, checkpoint existant avec checksum, reload réellement exécuté et exemple d'inférence archivé.

## I3 — Amélioration contrôlée

Configurer batch, accumulation, learning rate, epochs/max_steps et validation après mesure du temps/mémoire. Ne pas inventer un budget sûr. Adapter encoder/projecteur en premier ; LoRA seulement si nécessaire, support réel vérifié et baseline de comparaison conservée.

Choisir sur validation, garder train/test séparés. Comparer avant/après adaptation. Demander à Nevil des ablations utiles (temps vs agrégats, perturbations), sans modifier le protocole après avoir vu le test. Si sortie structurée casse, corriger au niveau parser/entraînement et comptabiliser les invalidités.

Définir des scores de classes de façon reproductible (par exemple scores de séquences candidates si implémentation validée) ou ne pas afficher de confiance. Calibration/abstention avec Nevil sur validation uniquement. Une phrase « 95 % sûr » générée n'est pas une calibration.

## I4 — Gel et intégration

Figer checkpoint/config et transmettre à Nevil pour évaluation finale. Livrer à Safoan un chargement singleton et une inférence bornée en temps. Vérifier une requête réelle et une panne contrôlée. Préparer README de reproduction, exigences matérielles mesurées, modèle/licence et URL d'artefact autorisée. Vérifier l'URL de téléchargement et le checksum avant livraison.

## Tests d'acceptation propres à ton rôle

- [ ] Pas de label caché dans les entrées generate.
- [ ] Batch TimeNet réel charge et traverse le modèle.
- [ ] Paramètres attendus apprennent ; NaN/OOM traités explicitement.
- [ ] Checkpoint complet recharge dans processus neuf.
- [ ] Prediction respecte le contrat ; invalidité non masquée.
- [ ] Version initiale/adaptée comparables sur validation.
- [ ] API appelle le modèle adapté réel et indique son état.
- [ ] Configuration, dépendances, poids et logs permettent de reproduire.

## Blocages et décisions

Accès modèle impossible : choisir une base supportée déjà accessible, documenter. OOM : réduire batch/longueur ou activer une technique réellement supportée, mesurer avant augmentation de coût. Modèle qui n'apprend pas : inspecter labels, pertes, gradients, normalisation avant d'ajouter des epochs. Temps court : réduire la génération et le nombre de variantes ; ne pas abandonner l'entraînement TSLM en prétendant être conforme.

Ne pas attendre l'UI pour prouver G2. Demander une fixture réelle à Nevil ; une fixture synthétique permet uniquement un test mécanique, jamais une conclusion scientifique.

## Passation attendue

Mettre à jour [[Journal Icham]] à chaque jalon : commit, run/config, commandes, machine, paramètres appris, résultats réels, chemin des poids, prochaine action et demande aux autres. Mettre à jour [[Passation]] lors d'un changement global.
