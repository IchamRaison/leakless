# LeakLess — du signal acoustique à l'alerte contextualisée

Prototype du hackathon EHL Zurich : écouter les canalisations, suivre une activité suspecte dans le temps et préparer un message exploitable.

**Chaîne IA disponible : C1 → historique temporel → OpenTSLM-SP + Qwen → aperçu de notification.** Aucun réentraînement nécessaire pour utiliser le service déployé.

## La chaîne

- **C1** produit neuf descripteurs acoustiques et un score continu par fenêtre audio d'une seconde.
- **Le suivi causal** conserve les horodatages, la persistance et les interruptions.
- **OpenTSLM-SP + Qwen3.5-4B avec LoRA** reçoit l'historique numérique et choisit une description temporelle contrôlée.
- **L'API** prépare un message destiné logiquement à Nevil après plus de 30 secondes consécutives avec score C1 ≥ 0,8 : la 31e fenêtre haute.

Les durées viennent des observations, pas du LLM. Le message décrit un « signal compatible avec une fuite observé depuis… », pas une date certaine d'apparition physique. Une description incohérente avec les faits mesurés est remplacée par un gabarit de secours explicitement identifié.

## Évaluateurs : utiliser la chaîne complète

La branche de référence est **`main`** : elle réunit la démo, l'adaptateur V2 et la chaîne C1/OpenTSLM/Qwen. Le dépôt est privé : demander à l'équipe l'accès GitHub et une clé SSH autorisée pour la démonstration hébergée. Aucun secret n'est inclus. Les anciennes branches divergentes sont conservées sous des tags `archive/2026-09-15/` ; voir [le registre de consolidation](docs/BRANCH_CONSOLIDATION.md).

```bash
git clone --branch main --single-branch \
  git@github.com:IchamRaison/leakless.git
cd leakless
```

### 1. Accéder au service chargé sur la H100

Garder ce tunnel ouvert dans un terminal local :

```bash
ssh -N -L 8020:127.0.0.1:8020 hicham@89.169.123.193
```

Dans un second terminal :

```bash
curl --fail http://127.0.0.1:8020/temporal/health
```

Vérifier `available:true` et `temporal_language:"opentslm-qwen-dynamics-d29bf1227d38"`. Le port **8020** sert cette chaîne ; 8019 est une ancienne version. Le service nécessite que la machine et le processus restent actifs. S'il est indisponible, contacter l'équipe ; ne pas relancer plusieurs serveurs sur le même port.

### 2. Exécuter le replay audio de bout en bout

Cette commande utilise le runtime, les poids et les WAV déjà présents sur la H100. Elle dure environ 43 secondes, puis attend les descriptions. Chaque exécution crée un dossier de résultats neuf.

```bash
ssh hicham@89.169.123.193 'cd /home/hicham/pipe-v0/code-language-api-004 &&
  demo_dir=$(mktemp -d /tmp/leakless-demo.XXXXXX) &&
  PYTHONPATH=/home/hicham/pipe-v0/.venv-temporal-api/lib/python3.12/site-packages:src \
  /home/hicham/pipe-v0/.venv-repro/bin/python scripts/temporal/smoke_language.py \
    --source /home/hicham/pipe-v0/artifacts/c1-temporal-002/http-smoke-001/scenario.json \
    --output "$demo_dir/result" \
    --endpoint http://127.0.0.1:8020'
```

Le rapport JSON affiche le message, les faits temporels, la provenance de la description et les contrôles de doublons. `status:"passed"` indique une recette logicielle réussie. Les WAV sont réels, mais leur répétition/chronologie est **artificielle** : ce replay démontre le fonctionnement de la chaîne, pas une performance terrain.

### 3. Raccorder son client

| Action | Route |
|---|---|
| Créer une session | `POST /temporal/sessions` avec `{"source_mode":"replay"}` |
| Envoyer une fenêtre | `POST /temporal/sessions/{id}/windows` |
| Lire l'état et les messages | `GET /temporal/sessions/{id}` |
| Décrire l'historique courant | `POST /temporal/sessions/{id}/describe` |
| Terminer l'observation | `POST /temporal/sessions/{id}/end` |

Fenêtres : WAV mono PCM16/32, 8 kHz, exactement 1 seconde ; multipart `file`, `sequence` (départ 0), `source_end_at` (fin source ISO8601 avec fuseau). Envoyer à 1 Hz, première fenêtre après une seconde, horloges synchronisées ; pas de rafale de fichiers historiques. Aucun label ni score attendu dans la requête.

Attendre `previews[].status:"ready"`, puis dédupliquer par `(event_id,kind)`. Les réponses indiquent `sent:false` et `delivery:"preview_only"` : **aucun WhatsApp n'est envoyé par ce module**. L'envoi et le raccordement au Monitor appartiennent au backend. La description à la demande exige au moins 31 fenêtres valides récentes ; modèle occupé : réponse 409.

Contrat, authentification, erreurs et commande de lancement : [guide de l'endpoint OpenTSLM/Qwen](docs/TEMPORAL_LANGUAGE_ENDPOINT.md).

## Ce qui est inclus et vérifié

Code IA/API, configurations, scripts d'entraînement/replay, tests, empreintes, rapports et notes de continuité. **39 tests logiciels et un replay HTTP complet vérifiés** : [preuve de vérification](docs/evidence/temporal-language-lora-001/verification.md), [rapport de replay](docs/evidence/temporal-language-lora-001/http-smoke-001/report.json).

Les poids Qwen, les adaptateurs binaires et le corpus complet ne sont pas distribués par un simple clone. Ils sont déjà installés sur la H100 pour la démonstration ci-dessus. Le [contrat](docs/TEMPORAL_LANGUAGE_ENDPOINT.md) fournit chemins et empreintes ; `requirements-ml.lock` et `requirements-temporal-api.txt` décrivent le runtime. Une installation autonome ailleurs nécessite de récupérer les artefacts avec l'équipe.

Le prototype distingue score acoustique, faits temporels et langage à vocabulaire fermé. La fiabilité terrain et la supériorité du TSLM sur des règles ne sont pas démontrées. Les [résultats détaillés](docs/evidence/temporal-language-lora-001/evaluation.json) restent accessibles pour examiner la portée de l'expérience.

## Autres composants

- [Studio audio de Safoan](docs/APPLICATION.md) : import WAV, lecture, waveform et spectrogramme ; raccordement au nouveau service séparé de cette livraison IA.
- [Endpoint C1 antérieur](docs/TEMPORAL_ENDPOINT.md), [expérience V0](docs/TSLM_V0.md), [sorties V2](docs/TSLM_V2_OUTPUT.md) : historiques distincts du service 8020.
- [Vault Obsidian](vault/Accueil.md) : architecture, décisions et passation ; [dépôt de notes partagé](https://github.com/IchamRaison/ehl-hackathon-zurich-vault).

Entire est optionnel pour contribuer : installer sa CLI puis `entire enable` pour les hooks locaux. Il n'est pas nécessaire pour appeler l'API ou exécuter la démonstration.
