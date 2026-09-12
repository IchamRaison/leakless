# Journal Icham

Icham

## État actuel

2026-09-12 : entretien `grill-me` demandé explicitement sur le périmètre ML. Premier tour en attente des réponses d'Icham ; aucune recommandation encore adoptée. Inventaire local en lecture seule effectué ; aucun environnement installé, modèle chargé ou entraînement lancé.

## Dernière passation

- Entretien : quatre décisions initiales, priorité de la contribution (classification/robustesse/restitution), ambition de nouveauté, échéance d'une version intégrable, budget de calcul. Les décisions dépendantes (base exacte, adaptation, sorties et déploiement) viendront après réponses et vérification des faits nécessaires.
- Recherche de faits déléguée au sous-agent `ml_runtime_facts` conformément à `/home/animus/.codex/skills/grilling/SKILL.md`. Commandes en lecture seule : `python3 --version`, `importlib.util.find_spec`, inspection des préfixes Python et des dossiers `.venv`/`venv`, `free -h`, `command -v nvidia-smi`, `lspci`, inventaire `rg --files` et lectures Git.
- Constat local : Python `/usr/bin/python3` 3.13.5 ; aucun venv actif/projet à la racine ; torch et transformers absents de cet interpréteur, NumPy présent. RAM 7,6 GiB dont environ 1,1 GiB disponibles au relevé. `nvidia-smi` absent ; seul contrôleur graphique listé Intel HD Graphics 520. Aucun accès GPU distant déduit de ces observations.
- Dépôt code inspecté à `3385f5a`, propre ; pas de code TSLM, manifeste Python, notebook, WAV ou checkpoint dans le checkout. Références Git déjà connues seulement, sans nouvelle vérification du distant du code ; `origin/nevil/setup` reste connu à `c47dc96` avec trois scripts non implémentés. Compte/crédit/quotas Nebius non inspectés.
- Prochaine action : réponses au premier tour, puis exploration des décisions débloquées. L'implémentation attend la compréhension partagée demandée par le skill ; la continuité documentaire du vault reste assurée selon la règle d'Icham.

## Installation du skill — étape précédente

- Source indiquée par Icham : https://www.aihero.dev/skills-grill-me ; dépôt amont https://github.com/mattpocock/skills, révision figée `3cca18b368ae95cdbdebbff572ccafa662551015`.
- Installation effectuée via le skill système `skill-installer` : `python3 /home/animus/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py --repo mattpocock/skills --ref 3cca18b368ae95cdbdebbff572ccafa662551015 --path skills/productivity/grill-me skills/productivity/grilling --dest /home/animus/.codex/skills`.
- Résultat : `/home/animus/.codex/skills/grill-me` et `/home/animus/.codex/skills/grilling` créés. La version actuelle de grill-me appelle grilling ; les deux sont nécessaires. Les deux `SKILL.md` et les deux `agents/openai.yaml` correspondent aux quatre empreintes Git de l'arbre amont, vérifiées par `git hash-object` et l'API GitHub.
- Installation globale à ce poste, pas distribuée aux coéquipiers par le vault. Disponible au prochain tour Codex ; invocation explicite de grill-me, aucun entretien commencé par l'installation. Aucun composant ML installé dans cette étape.

## Cadrage précédent

- Clarification structurelle : distinguer surveillance permanente et relecture d'un extrait ; proposer un technicien comme utilisateur principal et une aide à la qualification/documentation. Recommandation à confronter au professionnel, pas décision produit actée. Contribution distante `0a70a95` lue et conservée : [[PROBLEM STATEMENT - LeakLess Temporal AI]] se déclare approuvée en équipe sous réserve de G0 ; écart trois classes/contrats binaires signalé sans migration implicite.
- Vault distant intégré jusqu'à `f6eed23`, y compris la reprise de Safoan, avant cette édition. Source Zenodo relue en ligne : majorité des bruits venant de dlmeasure.com, minorité du site expérimental ; audit physique des archives non refait. Références OpenTSLM et Jacovi/Goldberg consultées pour distinguer résultats publiés, exactitude descriptive et fidélité d'une explication.
- Résultat de l'analyse : recommandation de conserver le protocole binaire courant, clarifier la séparation prompt/cibles et comparer la valeur du langage à une baseline enrichie par les mêmes mesures. Aucun message envoyé aux coéquipiers, aucune adoption de trois classes ou d'un nouveau nom.
- Prochaine action : examiner G0 de Nevil, confronter le problème à un cas réel du professionnel et arbitrer les propositions avant entraînement destiné aux résultats ; I0 runtime reste indépendant.

## Preuves de la prise en main précédente

- Références inspectées : vault `main` à `07708b13e57bdb62be9c9dce3d3cee398ba9e6af` ; code `main` à `3ea769d7ed2631dd96e08aa3853584a497841b0a`.
- Commandes exécutées : `git pull --ff-only` dans les deux dépôts (déjà à jour), `git status --short --branch`, `git ls-files` ; lecture des notes ; `pdftotext -layout` et rendu `pdftoppm -scale-to 1440 -png` du PDF, puis inspection des 22 pages.
- Résultat : `main` du code contient le bootstrap et les notes, sans modules produit. La branche distante `nevil/setup` à `c47dc961791ded28fc697b00b29738838f6252eb` contient de la documentation et trois scripts levant `NotImplementedError`, constaté par `git diff --stat main..origin/nevil/setup` et `git show`. Son README annonce une CLI TimeNet opérationnelle sur le poste de Nevil ; non reproduit ici, branche non intégrée.
- Artefacts ML / tests fonctionnels : aucun produit pendant cette lecture. Le PDF ne précise ni deadline de soumission ni durée du pitch ; accès et budget GPU restent à vérifier.
- Interfaces : aucun changement. Les contrats v0.1 restent proposés, G1 non validé. Le nom LeakLess reste une proposition sans validation finale enregistrée.
- Prochaine action : démarrer I0 de [[Agent Icham - ML]] par l'inventaire Python/GPU et la lecture de la révision OpenTSLM ; établir les accès/base disponibles. Compte, crédit et plafond opérationnel à confirmer avant création d'une instance payante. Dépendance I1 : exemple réel de développement et transformation TimeNet fournis par Nevil.

## Historique

### 2026-09-12 — Challenge du cadrage de Nevil

Icham transmet l'analyse Why/How/What de Nevil et demande une critique. Relecture des contrats et du protocole, vérification des sources publiques et rédaction des objections/test proposé. L'audit G0 reste prioritaire ; aucune performance, utilité terrain ou validation de nouvelle interface annoncée. Contributions distantes de Safoan conservées.

### 2026-09-12 — Lecture complète et reprise du rôle Icham

Lecture demandée par Icham, identification explicite du rôle et distinction du plan courant acoustique avec les anciennes pistes débit/pression. Lecture visuelle du PDF complétée ; mentions obsolètes du brief et statut historique de la proposition initiale corrigés. Le vault dédié reste la référence ; son miroir dans le dépôt code doit être rafraîchi explicitement après publication des notes.
