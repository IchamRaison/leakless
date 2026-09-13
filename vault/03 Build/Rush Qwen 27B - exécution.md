# Rush Qwen 27B — exécution

## Mandat du 13 septembre 2026

Icham autorise une heure totale : 09:17:26–10:17:26 Paris ; arrêt propre des mises à jour à 10:02:26. Le nouveau goal porte sur ce rush, sans déclarer achevé le cap produit plus large. Première action : téléchargement officiel anonyme de `Qwen/Qwen3.8-27B`, révision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, sur les deux H100 libres vérifiées à la reprise.

## Recette fixe, avant toute observation

- BF16 uniquement, aucun remplacement par FP8/autre modèle. Base gelée ; encodeur/projecteur neufs et LoRA entraînables. Dimension cachée 5120.
- LoRA rang8, alpha16, dropout0, biais non entraînés ; cibles `q_proj`, `v_proj`, `in_proj_qkv`, `out_proj` à vérifier sur les modules chargés.
- AdamW : encodeur2e-4, projecteur1e-4, LoRA1e-4, weight decay0,01, clippingglobal1, lot effectif8, graine20260912.
- Même TimeNet, preprocessing canonique, série(4,64), supervision réponse complète + EOS. Journaliser séparément classe/description et gradients, sans changer la loss.
- Checkpointing non réentrant, Qwen en mode train mais poids originaux gelés, cache désactivé ; thinking désactivé.
- Les598 clips train et les3folds internes existants uniquement. Validation officielle/test/externe fermés. Une recette ; A/C1/D3 sont références descriptives, pas contrôle causal de la taille seule.

## Ordre et critères

1. Jusqu'à minute20 : télécharger/vérifier la même base, code/environnement/données ; adapter les helpers et exécuter les tests. Microbatch4 puis2 puis1 : premier qui passe avec≥10% de VRAM libre au pic. Vrai backward fini dans les3composants, base inchangée. Débit avec logger, sauvegarde/reload neuf et parité entre machines≤1e-6. Compter les mises à jour techniques puis jeter leurs poids.
2. Avant tout résultat réservé, fixer1/2/4époques : plus grand budget dont le débit mesuré majoré de25% permet les3fits avant minute45. Sinon1époque et couverture potentiellement partielle. H100-1 fold0 ; H100-2 fold1 puisfold2. Chaque fit neuf, checkpoint terminal seulement, arrêt propre en frontière de pas à minute45.
3. Minutes45–60 : recharger en processus neufs ; évaluer chaque checkpoint sur ses clips réservés via le harness existant. AUC groupe primaire ; AUCclip/NLL/Brier/erreurs au seuil diagnostique0,5 secondaires. Export continu sans seuil. Score inchangé : softmax des sommes `leak;` / `no_leak;`, sans description/EOS, non calibré.
4. Publier configurations, prédictions, résultats, empreintes, compteurs et limites. Si tous les folds finissent : moyenne des3AUCgroupe et références. Sinon résultats individuels, aucune moyenne partielle. Prototype désigné à l'avance : fold0. Ni refit598, ni test final, ni promotion comme modèle validé.

Échec du contrôle technique : documenter et ne pas masquer par un changement de modèle/précision/représentation. Qualité et fin des3fits non garanties.

## État réel

Téléchargements terminés sur les deux nœuds :31 fichiers chacun, reçus strictement identiques SHA256 `66cc72538886a2addb286dd3c7b553efb9da6e1328b4bbcc8733cb832103a994`, empreintes de tous les fichiers calculées. Code initial `dae7e22` publié : LoRA opt-in, mode train compatible checkpointing, logger microbatch/LoRA et chargement explicite des adaptateurs sans fusion ; runner séparant fit et évaluation. Tests TSLM sur H100-2 :172 réussis sans skip, dont transparence du logger avec microbatches4/2/1. Contrôle technique MB4 sur H100-1 lancé ; pas encore de fit de comparaison ni de qualité27B mesurée. Anciennes preuves/splits inchangés.

Sources : [modèle officiel](https://huggingface.co/Qwen/Qwen3.8-27B), `configs/tslm/qwen27b_lora.json`. Historique antérieur inchangé : [[Diagnostic causal - exécution]].
