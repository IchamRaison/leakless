# Restitution cohérente — interface opt-in

Implémentation logicielle de la phase 3 V2, **pas un nouveau modèle entraîné**.
`Predictor.predict`, `Prediction` v0.1 et les livraisons V1 restent inchangés.
Aucun endpoint applicatif n'est remplacé ou déployé par ce module.

Statut : sept tests CPU sur faux backend passent. L'inférence réelle et le lot
d'audit attendent la parité numérique et le dispatcher canonique
`predict.preprocess_for_model(waveform, sample_rate, metadata)` fourni par le
correctif V2. Aucune tolérance ni performance nouvelle n'est revendiquée ici.

## Utilisation

```python
from pipe.tslm.coherent import CoherentPredictor

service = CoherentPredictor(
    checkpoint="/chemin/bundle-selectionne",
    decision="/chemin/decision.json",
    validation_evidence="/chemin/validation-evidence.json",
    device="cuda",
)
result = service.predict(wav_bytes)
```

Le backend `Predictor` est chargé une fois et reste privé. Les requêtes du wrapper
sont sérialisées sans modifier les verrous V1 ; une requête concurrente reçoit
l'erreur existante `model_busy`. Le score et la génération utilisent ce même
backend. Le DSP emploie son dispatcher canonique et ses métadonnées : pas de
normalisation parallèle ni de version V1 sélectionnée implicitement.

## Politique numérique V2

Le bundle V2 déclare `single_clip_acoustic_encoding: true` et le scoring
`class-continuation-logprob-sum-softmax-single-clip-v2`. Chaque clip passe
séparément dans l'encodeur/projecteur ; les interfaces acceptent toujours les
lots et rendent un score par clip. Le graphe de gradients est conservé pour
l'apprentissage. V1 garde sa politique historique par défaut.

La trace contrôlée `docs/evidence/tslm-v2/batch-trace-001/report.json` montre une
première différence à la sortie de l'encodeur, amplifiée jusqu'à `0.06245874`
sur le score. L'intervention par clip rétablit l'égalité exacte dans les vingt
contextes tracés. Cela ne remplace pas le gate complet : 208 validations et
un train, trois interfaces, lots 1/2/4 dans les deux ordres, puis processus neuf,
tolérance absolue inchangée `1e-6`. Aucun entraînement avant son PASS complet.

## Préparer la décision après la parité

Le producteur du seuil doit utiliser les prédictions **de validation du modèle
final exact**, avec la règle préannoncée. Ne pas transférer le seuil d'un modèle
de fold ni recopier l'affichage arrondi d'un rapport. Ce module ne choisit ni ne
réajuste le seuil.

Le reçu `validation-evidence.json` contient exactement :

```python
from pipe.tslm.coherent import checkpoint_identity

receipt = {
    "schema_version": "pipe-threshold-evidence-v1",
    "fit_fold": "val",
    "threshold": selected_threshold,  # float complet du calcul de validation
    "threshold_repr": repr(float(selected_threshold)),
    "model_identity": checkpoint_identity(checkpoint),
    "rule": declared_validation_rule,
    "split_sha256": manifest_digest,
    "validation_predictions_sha256": validation_predictions_digest,
    "threshold_method_sha256": threshold_implementation_digest,
}
```

`checkpoint_identity` lie le checksum du bundle, ses poids temporels,
`config_hash`, `model_version`, `preprocessing_version`, le fichier
`scoring_spec.json` et les SHA des sources modèle/prédiction/prétraitement/
connecteur effectivement présentes dans l'installation. Tout module numérique
supplémentaire devra être inclus explicitement dans cette liste avant utilisation.

Après enregistrement du reçu par le pipeline de validation :

```python
import json
from pipe.tslm.coherent import decision_artifact

artifact = decision_artifact(checkpoint, evidence_path, "decision-v2-001")
with open(new_decision_path, "x") as stream:
    json.dump(artifact, stream, indent=2, allow_nan=False)
    stream.write("\n")
```

Le helper renvoie un dictionnaire, sans entraîner, inférer, écrire ou modifier le
bundle. L'artefact lie le reçu par SHA et versionne aussi le code de restitution.
Au chargement, il doit correspondre exactement à l'identité courante et au reçu.
Champs absents/supplémentaires, clés JSON répétées, seuil booléen/non fini,
provenance absente et reçu provenant du test sont refusés. Aucun seuil par défaut.

Ces empreintes assurent l'intégrité et l'association des artefacts ; **ce ne sont
pas des signatures cryptographiques ni une preuve que le producteur a réellement
ajusté le seuil sur validation**. Cette preuve appartient au pipeline et à ses
traces. Le wrapper ne recalcule pas les métriques et ne lit aucun label à l'inférence.

## Nouveau résultat `pipe-coherent-v1`

La classe de premier niveau est exclusivement `probability_leak >= threshold`.
`probability_leak` reste un **score brut relatif non calibré**, pas une fréquence
de fuite terrain. Le seuil complet, sa version et le SHA de l'artefact décision
accompagnent chaque résultat.

- `prediction` : seule classe destinée à l'affichage.
- `dominant_band_hz` : bande mesurée par DSP.
- `description` : texte validé contre cette mesure ou gabarit factuel DSP.
- `description_source`, `fallback_used`, `fallback_reasons` : provenance explicite.
- `audit` : retours exacts de `generate` avant nettoyage, texte brut, payload V1,
  erreurs et concordances brutes. **Ne pas afficher sa classe comme une seconde
  décision** ; ces champs servent au diagnostic et à la mesure de qualité brute.

Le texte brut doit respecter le format, la bande DSP et la décision issue du score.
Sinon, seul le gabarit factuel est affiché, avec les raisons du remplacement. Un
`UnicodeError` identifié pendant le décodage textuel peut aussi donner ce secours,
uniquement après score valide et mesure DSP réussie ; son erreur reste dans l'audit.

Entrée absente/invalide/constante, score non fini, modèle indisponible et panne GPU
ne donnent **aucun résultat de classe**, même si un score avait été obtenu avant
une panne ultérieure. Les erreurs existantes sont propagées ; les exceptions de
prédiction portent `coherent_audit` pour conserver les traces déjà disponibles.
Une `RuntimeError` de génération n'est pas assimilée à un simple défaut de texte.

## Vérification et limites

```bash
python3 -m unittest discover -s tests/tslm -p test_coherent.py -v
```

Les fixtures couvrent seuil exact et frontière inclusive, classe concurrente,
mauvaise bande, sortie invalide, décodage défaillant, erreurs de score/GPU et
artefacts incompatibles. Le critère futur sur le lot audité sera zéro contradiction
**affichée**, en conservant séparément la qualité du texte brut et tous les échecs.

Un gabarit juste par construction ne transforme pas les erreurs du modèle en
réussites. Cette couche n'améliore pas, à elle seule, rappel, fausses alertes,
calibration ou généralisation. Les latences incluent score et génération ; aucun
débit de surveillance continue n'est garanti. L'application devra adopter ce
nouveau contrat explicitement après concertation.
