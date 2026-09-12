# Idées

## Statut

Piste privilégiée par Icham : aide à la détection de fuites d’eau, avec un membre du groupe qui possède une entreprise du secteur. Besoin précis et dataset encore non validés ; aucun entraînement lancé. Le dataset de banc hydraulique ci-dessous ne doit pas être confondu avec des données de canalisations d’eau.

## 1. Assistant de diagnostic hydraulique

Utilisateur : technicien de maintenance. Problème proposé à valider : identifier le composant dégradé à partir de courbes de pression, débit et température.

Données vérifiées via l’API UCI : banc hydraulique, cycles de 60 secondes, labels de refroidisseur, valve, fuite de pompe et accumulateur. Capteurs à plusieurs fréquences. Cible MVP : un seul composant, par exemple fuite interne de pompe, avec niveau et résumé factuel des signaux.

Baseline proposée : caractéristiques statistiques + Random Forest, réponse textuelle par gabarit. TSLM : signaux → état structuré et description courte ; annotations textuelles construites depuis les labels et mesures, pas présentées comme rapports humains. Évaluer macro-F1, confusion et exactitude des affirmations mesurables.

Risques : banc expérimental unique, manque de rapports experts, dépendance entre cycles. Vérifier l’ordre des acquisitions et les groupes avant de choisir un split ; ne pas prétendre généraliser à une nouvelle machine. Ne pas transformer un diagnostic d’état en prédiction de panne future. Licence et téléchargement des fichiers encore à vérifier.

Source consultée : https://archive.ics.uci.edu/api/dataset?id=447
Page : https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

## 2. Brouillon de compte rendu ECG contrôlé

Utilisateur : médecin relisant un ECG. Problème proposé : préparer une synthèse vérifiable, avec abstention et validation humaine obligatoire.

PTB-XL 1.0.3 : ECG 12 dérivations de 10 secondes, rapports et labels SCP, accès ouvert CC BY 4.0. Splits officiels par patient : folds 1–8 train, 9 validation, 10 test. Les rapports proviennent de cardiologues ou de l’appareil, avec des niveaux de validation variables.

MVP : sous-ensemble d’anomalies défini à l’avance, prédictions structurées et texte court. Baseline : classifieur + gabarit, et checkpoint TSLM avant fine-tuning. Évaluer labels et affirmations non supportées plutôt que la seule ressemblance textuelle.

Risques : OpenTSLM traite déjà ECG-QA, donc nouveauté à démontrer. Auditer l’exposition du checkpoint à PTB-XL/ECG-QA avant toute revendication de test inédit. Ne pas utiliser le rapport ou les codes cibles en entrée. Prototype de recherche, aucune utilisation diagnostique autonome.

Sources consultées : https://physionet.org/content/ptb-xl/1.0.3/ ; https://github.com/OpenTSLM/OpenTSLM

## 3. Assistant de triage de télémétrie spatiale

Utilisateur : opérateur surveillant des canaux. Problème proposé : trouver et résumer les intervalles anormaux pour guider l’inspection.

Source Telemanom consultée : SMAP/MSL, télémétrie réelle et intervalles annotés. MVP : localisation d’anomalie et description de sa forme ; aucune cause physique inventée.

Baseline proposée : détecteur statistique ou prévision + seuil ; métriques par événement, faux positifs et retard de détection.

Risques majeurs : identités des canaux et commandes anonymisées ; pas de rapports de cause racine. README indiquant une mise à l’échelle utilisant les extrema du test, incompatible avec une revendication stricte de preprocessing sans fuite. Téléchargement proposé via Kaggle avec authentification. Licence des données et possibilité de repartir de données non contaminées à vérifier. Piste non prioritaire en l’état.

Source consultée : https://github.com/khundman/telemanom

## Conclusions techniques vérifiées

TimeNet standardise et charge les données, mais ne réalise pas l’entraînement. OpenTSLM propose des checkpoints et des scripts de fine-tuning ; l’accès aux modèles de base Llama/Gemma peut être soumis à autorisation. Aucune durée d’entraînement ni performance n’a été mesurée pour notre projet.

Sources : https://github.com/OpenTSLM/TimeNet ; https://github.com/OpenTSLM/OpenTSLM

## Prochaine décision

Cadrer la piste eau avec le membre du groupe : types de réseaux/fuites, instruments et enregistrements, décision métier difficile. Ne promettre ni localisation ni prédiction sans données et annotations adaptées.
