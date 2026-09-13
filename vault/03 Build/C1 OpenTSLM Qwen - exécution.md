# C1 → OpenTSLM → Qwen : évolution et message

## État actuel — implémenté, entraîné, testé et disponible

Publication évaluateurs : Icham demande README et push immédiat, sans nouveau packaging/déploiement. README racine : branche, tunnel8020, replay43s sur le runtime existant et contrat d'intégration. Dépôt code désormais `IchamRaison/leakless` (ancien nom redirigé), privé : accès GitHub/SSH à fournir par l'équipe. Poids sur H100 ; clone non autonome, aucune release de poids publiée. Preuves/limites conservées. Prochaine action : transmettre le lien de branche et les accès autorisés, puis raccorder le backend.

Livraison `2318dee`, branche `feat/c1-opentslm-qwen`. Service code `77c174f`, PID129483 au lancement, première H100 `hicham@89.169.123.193`, **127.0.0.1:8020**, répertoire `/home/hicham/pipe-v0/code-language-api-004`. Aucun WhatsApp, frontend ou repli195 touché ; ancien8019 préservé. Pas de superviseur de redémarrage automatique.

Checkpoint retenu : `artifacts/temporal-language-lora-001`, Qwen3.5-4B + OpenTSLM-SP + LoRA rang8. **16/25descriptions brutes correctes**, contre3/25 sans LoRA ;9fallbacks,8/25 sur inversion temporelle. Rechargement neuf exact (écart0), deux recettes seulement. Les25scénarios sont des chronologies artificielles de développement, pas des acquisitions indépendantes. Les règles font25/25 par construction : aucune supériorité du TSLM ni qualité terrain démontrée. Choix de formulation contrôlée, pas rédaction libre ; secondes calculées par horodatages/compteurs, pas par Qwen.

**39tests passent**, commande/preuve dans `docs/evidence/temporal-language-lora-001/verification.md`. Recette HTTP43secondes à1Hz avec vrais WAV/poids C1 : 3basses,35hautes,5basses ; message logique Nevil à la31e haute, rien à30, doublon idempotent, deux aperçus prêts, zéro envoi. Qwen reconnaît la persistance ; il rate la fin, donc `template_fallback` explicite. P95ingestion C18,12ms sur ce court essai, hors calcul asynchrone Qwen. `high_observed_seconds:35` compte les hautes ; `observed_seconds:40` inclut les5basses de clôture et ne décrit pas la durée physique de fuite.

Métadonnées SHA `d29bf1227d389801c35b33d47d1227b369c26a628207155d8ddc4e47ba5f3295`, version `opentslm-qwen-dynamics-d29bf1227d38`. Poids/normalisation/métadonnées sauvegardés localement hors Git dans `/home/animus/ehl-hackathon-zurich-c1-temporal/artifacts/temporal-language-lora-001`, empreintes identiques ; base Qwen encore requise sur le GPU. Preuves publiées : `docs/evidence/temporal-language-lora-001/`, contrat `docs/TEMPORAL_LANGUAGE_ENDPOINT.md`.

**Prochaine action : raccordement backend/Monitor, pas nouveau fit.** Tunnel `ssh -N -L 8020:127.0.0.1:8020 hicham@89.169.123.193`, puis routes `/temporal/sessions`, fenêtres et lecture `previews`. Attendre `status:ready`, dédupliquer par `(event_id,kind)` ; l'envoi reste externe et non testé. Ne pas exposer directement le port : proxy HTTPS/authentification nécessaire hors tunnel. Ne pas relancer les deux fits terminés.

## Demande actuelle

Icham corrige le périmètre : conserver C1 pour détecter, utiliser **OpenTSLM** (pas un LSTM de classification acoustique) et un petit Qwen pour restituer l'évolution ; préparer un message destiné à Nevil quand le signal reste actif **plus de30secondes**. Partie IA/endpoint seulement : aucun envoi WhatsApp, frontend ou modification du repli195. Nouvelle branche `feat/c1-opentslm-qwen`, composants C1/API déjà testés réutilisés.

## Historique — implémentation prévue, avant apprentissage

- Historique causal jusqu'à64fenêtres1s, dix valeurs (neuf mesures et score C1), remise à zéro sur trou/qualité/reprise. Horodatages/compteurs déterministes calculent durée observée et disponibilité ; jamais d'origine physique de fuite inventée.
- OpenTSLM-SP existant + **Qwen3.5-4B déjà disponible**, Qwen gelé, encodeur/projecteur entraînés pour une tâche distincte de restitution temporelle. Choix parmi cinq formulations contrôlées : calme, bref, intermittent, persistant, retour au calme. Les durées et décisions d'envoi ne sont pas confiées à une génération libre.
- Supervision de démonstration sur chronologies artificielles de valeurs/mesures C1 train. Ce ne sont pas de vraies apparitions de fuite : labels de dynamique issus de règles explicites, benchmark de règles attendu parfait par construction. Aucun réemploi des labels/acquisitions de confirmation Aghashahi ni du test officiel. L'évaluation de scénarios réservés ne démontrera pas de qualité terrain ni de supériorité du TSLM.
- Une recette initiale bornée, graine et budget publiés avant fit ; rechargement dans un processus neuf, comparaison règles/TSLM et perturbations temporelles. Si formulation incohérente avec les faits, retour explicite au gabarit, pas de texte modèle présenté comme fiable.
- Nouveau service isolé et nouvelle base ; ancien endpoint8019 conservé. Notification proposée une fois par événement après31fenêtres hautes consécutives, déduplication durable ; texte et destinataire logique Nevil retournés au backend, `sent:false`.

État : implémentation en cours, aucun nouvel entraînement à ce jalon. Les résultats du petit LSTM précédent restent conservés et ne décrivent pas ce nouveau module.

## Historique — campagne lancée

Code `3346c33`, préinscription de préparation publiée `a468d26` avant fit ; `configs/temporal/language.json`. Quarante scénarios train (8par formulation),25réservés (5par formulation), mesures réelles des598train C1 et ordre artificiel. Base Qwen3.5-4B/révision851bf6e… vérifiée par empreintes, OpenTSLM2968f4b… déjà installé. Une recette400pas, encodeur/projecteur3e-4, NLL sur les cinq tokens de choix Qwen, Qwen gelé. H100 première machine, environ15Go utilisés au début ; aucune nouvelle campagne acoustique ou utilisation d'Aghashahi.

Service `b3f593e` implémenté : contexte64×10, message à31hautes consécutives seulement, file durable SQLite et un travailleur modèle hors chemin critique C1, destinataire logique Nevil, `sent:false`. Cinq scénarios de suivi passent localement, dont interruption avant30s et déduplication après redémarrage. Régression API complète et résultat ML encore en cours. Formulations contrôlées, pas de rédaction libre : Qwen choisit une description à partir des séries ; durées/gabarit d'alerte restent déterministes. Prochaine action : terminer le fit/reload, publier performance brute et fréquence de fallback, puis recette HTTP réelle de plus de30s sur nouveau port8020 sans remplacer8019.

## Historique — second essai borné, annoncé sur faiblesse train

39tests logiciels passent. Au pas345 du premier fit, NLL train1,82695 (référence uniforme ln5≈1,609) ; l'apprentissage encodeur/projecteur seul reste insuffisant, pas de supériorité affirmée à partir d'un batch. Premier fit conservé/terminé sans adaptation en cours. Deuxième recette annoncée : **LoRA rang8 sur q_proj/v_proj de Qwen, LR1e-4**, mêmes400pas, données/normalisation/initialisation/encodeur/projecteurLR inchangés. Base Qwen gelée hors adaptateurs, empreinte des paramètres gelés vérifiée avant/après. `n_configs_compared:2`. Ce choix répond au comportement train, pas à une nouvelle observation physique réservée. Comparaison sur les mêmes25scénarios synthétiques de développement, désormais sélection exploratoire et non confirmation indépendante ; pas de troisième recherche automatique.

Premier résultat complet :3/25descriptions brutes correctes,22fallbacks,0/25inversions décrites correctement ; reload exact. Preuves `5f68057`. Second fit400pas terminé (349,53s de boucle),917504paramètres LoRA ; reload/évaluation en cours. Pour le service, retenir le checkpoint au plus grand nombre de descriptions brutes correctes sur ces25scénarios de développement ; égalité → modèle sans LoRA. Le gabarit sûr ne compte pas comme une réussite du modèle. Pas d'autres hyperparamètres ajustés, pas de nouvelle réserve physique consultée.

Le premier reload LoRA échoue avant les scénarios : clés sauvegardées depuis le wrapper (`llm.`), lecteur amont OpenTSLM attendant les noms internes. Correctif `77c174f` : normalisation du préfixe et correspondance exhaustive des clés avant copie ; poids et métadonnées restent inchangés, aucun refit. Métadonnées LoRA SHA `d29bf1227d389801c35b33d47d1227b369c26a628207155d8ddc4e47ba5f3295`. Relecture dans un processus neuf en cours.
