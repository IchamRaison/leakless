# C1 temporel — exécution

## Autorisation et périmètre

Icham autorise l'implémentation complète du [[Plan C1 - suivi temporel et notifications]] et la réaffectation d'Aghashahi dans un **nouveau protocole**, sans réécrire les anciennes métadonnées/résultats de réserve. Il précise ensuite : **aucun test WhatsApp**, aperçu/journal uniquement ; intégration **via un endpoint sur le GPU**, sans chantier frontend imposé. Aucun accès à `195.242.28.46`, endpoint de repli protégé.

Branche isolée `feat/c1-temporal-alerts`, worktree `/home/animus/ehl-hackathon-zurich-c1-temporal`, basée sur l'application `5beb344`. Les anciennes branches et modifications locales restent intactes. H100 `hicham@89.169.123.193` uniquement ; fit et évaluations désormais terminés, détails ci-dessous.

## Protocole fixé avant scores

- C1 : recette existante, C=0,01 déjà sélectionné ; fit unique sur 598 train Zenodo uniquement, poids/scaler figés ensuite. Aucun test/validation officielle Zenodo utilisé. Séries Aghashahi distinctes du corpus d'apprentissage C1, pas de stacking sur scores in-sample.
- Aghashahi : 120 enregistrements hydrophone de 30 secondes /60 conditions heuristiques, H1/H2 ensemble. Ancienne préparation et archives inchangées ; vérification des empreintes et reprise de ses arrays sans requantification PCM16.
- Partition avant mesures/scores : **BR train**, sauf conditions `0.47 LPS_N` et `Transient_N` réservées à **BR validation** ; **LO confirmation** entière. Attendus : 40/20/60 enregistrements, 20/10/30 conditions, fuite/sans-fuite 32/8,16/4,48/12. Autres modalités d'une même condition jamais séparées. Deux bruits de fond exclus de cette campagne, pas recodés arbitrairement.
- Confirmation = autre topologie du même banc, **pas preuve inter-sites**, sessions réelles inconnues. Les conditions de validation ne représentent que deux groupes sans fuite ; incertitude majeure à publier. Hong Kong non utilisé ni déclaré confirmé.
- Pas de bornes de fuite disponibles : **classification de séquences de 30 secondes**, loss seulement sur la sortie terminale, aucun label par seconde inventé. Durées/alertes = essais logiciels séparés, pas délai physique ou fausses alertes/jour validés.
- LSTM unidirectionnel 10→32→1, FP32 sur H100, une recette ; AdamW1e-3, WD0,01, lot8,100époques terminales, clipping1, BCE pondérée par les classes train seulement, graine20260913. Ni recherche d'époques ni LoRA. Standardisation des dix entrées ajustée sur train seulement.
- Contrôles fixes sur mêmes séquences : médiane des 30 scores C1, EWMA alpha0,2, logistique C0,01 sur moyenne des dix entrées (sans ordre). Référence choisie sur validation uniquement, ordre fixe de départage ; quatre systèmes au total, un fit C1 antérieur à ces comparaisons, un fit LSTM et un fit contrôle sans ordre. Compteurs techniques distincts.
- Primaire : AUC après médiane H1/H2 par condition, secondaires par enregistrement et erreurs au seuil diagnostique0,5. Promotion si gain validation≥0,03, puis gain confirmation≥0,03 et borne basse du bootstrap apparié95% sur conditions >0. Aucun réglage après confirmation. Sans cela, C1/persistance reste le candidat de démonstration ; le LSTM n'est pas promu comme suivi événementiel sans données correspondantes.
- Politique d'alerte **de démonstration** fixée séparément : score≥0,8 pendant3secondes pour ouvrir ; score≤0,4 pendant5secondes pour fin observée ; défaut/trou au-delà2secondes dégrade la surveillance. Pas de prétention au seuil métier optimal. Dates, durée observée et couverture calculées hors modèle de langage.

Configuration : `configs/temporal/c1_lstm.json`. Aperçu déterministe suffisant ; pas de fournisseur WhatsApp ni nouveau téléchargement Qwen. L'endpoint sera d'abord accessible en loopback via tunnel SSH, sans exposition publique non authentifiée.

## Audit et arrêt mécanique avant LSTM

Code `b3f7458`, préinscription publiée `96bfe7c` avant calcul. Quatre tests mécaniques passent sur le runtime cible ; audit retrouve exactement les populations prévues. `c1-temporal-001` ajuste C1 puis s'arrête **avant tout pas LSTM** : des échantillons train Aghashahi atteignent -1 après décodage int32/2^31. Dix enregistrements train concernés, 177 à1124 échantillons à pleine échelle chacun ; aucune validation/confirmation scorée. Ancien run conservé.

Correction préannoncée pour `c1-temporal-002` : réutiliser le C1 exact du premier run (pas de deuxième fit), conserver toutes les séquences sans modification ni exclusion guidée par score pour l'expérience de classification, et publier leurs défauts de qualité. Cette expérience inclusive **ne représente pas la couverture du service**, qui s'abstient sur silence/saturation. Aucun nouveau seuil acoustique optimisé d'après les réserves. Configuration, partition et budget d'apprentissage inchangés hors cette politique explicite. Le suivi SQLite/API est implémenté, tests encore à exécuter. Environnement API séparé créé, runtime ML original intact.

## Fit terminé, correction du lecteur avant réserves

Préinscription `12303e6`, code fit `472b57b` : unique LSTM terminé, 100époques/500pas,39/40 séquences train correctes, NLL binaire0,030418. C1 réutilisé sans refit. Le premier contrôle reload bloque avant les réserves : écart CPU/GPU1,9699e-5, supérieur à la tolérance1e-5. Contraste train seul : TF32 désactivé comme pendant le fit → écart5,3644e-7. Cause numérique identifiée ; le lecteur commun service/évaluation doit restaurer cette politique FP32, sans changer checkpoint ni tolérance. Aucune relance d'entraînement.

API/suivi/replay `22fc209` :31tests modèle/API passent sur le runtime cible (deux avertissements de dépréciation, aucun skip). Trois scénarios SQLite passent aussi localement. Prochaine action : contrôle reload corrigé, validation/confirmation fixes, puis test HTTP réel et mesure de cadence. Aucun service public ni envoi externe à ce jalon.

## Résultats figés — LSTM non promu

Lecteur corrigé `3b5d094`,34tests ciblés réussis, reload CPU/GPU5,3644e-7 (tolérance inchangée1e-5), zéro réentraînement. Bundle `cea924a6a0f8c3d3ff91930d8125f309a3bcded4c965a4ef53c2c566fc32ebdd`.

| AUC par condition | BR validation (10groupes) | LO confirmation (30groupes) |
|---|---:|---:|
| C1 médiane |0,625|1,000|
| C1 EWMA |0,750|0,916667|
| Contrôle mêmes10entrées sans ordre |0,8125|1,000|
| LSTM |0,375|0,590278|

Référence sélectionnée sur validation : contrôle sans ordre. Gain LSTM confirmation−0,409722, IC apparié95% [−0,729167;−0,090278]. **Critère de promotion échoué dès validation** ; confirmation complète publiée sans ajustement. Le LSTM reste un endpoint expérimental de classification terminale, pas le moteur d'alertes. C1/persistance reste uniquement candidat de démonstration, pas détecteur terrain validé. Au seuil diagnostique0,5, C1 médiane classe toutes les séquences de confirmation «fuite» malgré son AUC agrégée1,0 : classement et calibration/décision ne sont pas équivalents. Ne pas sélectionner un seuil sur cette réserve.

Onze séquences validation et21confirmation atteignent les rails conservatifs ; elles restent incluses dans cette expérience, pas dans les garanties de couverture du service. Aucun site indépendant ni annotation d'événement, deux groupes négatifs validation/six confirmation. Preuves `docs/evidence/c1-temporal-002/`, données/checkpoints distants dans `/home/hicham/pipe-v0/artifacts/c1-temporal-002`.

## Service et recette vérifiés

Code `5553e14`, dossier distant `code-temporal-api-003`, PID123337, écoute **127.0.0.1:8019** sur la première H100. `/temporal/health` répond disponible, C1 `c1-15b1fa3511e6`, aperçus uniquement. Contrat et tunnel dans `docs/TEMPORAL_ENDPOINT.md`. Repli195 intact ; aucun envoi externe, aucun frontend modifié.

Recette `http-smoke-001` **réussie avec vrais poids** :12fenêtres à1Hz, neuf mesures/scores C1 HTTP/offline strictement identiques ; un événement ouvert puis fermé et deux aperçus `sent:false`. p95serveur11,0595ms/max12,6306ms, court essai sans benchmark de charge. Séquence LSTM train sans pleine échelle : score HTTP/offline identique, appel0,316s initialisation comprise. Scénario construit volontairement avec les extrêmes train répétés, chronologie artificielle explicite : aucune précision de détection déduite du replay. Audit indépendant local des CSV, partitions, AUC par comparaisons par paires et choix de référence réussi. Preuves publiées `8baecab`.

Prochaine action intégrateur : tunnel `ssh -N -L 8019:127.0.0.1:8019 hicham@89.169.123.193`, puis routes `/temporal` selon le contrat. Le modèle/API est livré ; le Monitor n'est pas raccordé par cette tâche. La qualification terrain nécessite d'autres données avec annotations événementielles, une politique opérationnelle et une nouvelle réserve préservée. Aucune collecte/recette supplémentaire lancée après l'échec LSTM.

Contrôle final code `8baecab` : **35tests passent**, zéro skip, deux avertissements de dépréciation FastAPI/Starlette (3,48s). `/temporal/health`, PID123337 et écoute loopback revérifiés. Bundle également sauvegardé localement dans `/home/animus/ehl-hackathon-zurich-c1-temporal/artifacts/c1-temporal-002/bundle`, cinq empreintes vérifiées ; poids exclus de Git, manifestes/preuves publiés. La branche code contient le contrat, le lock API séparé et les commandes de recette/repli.
