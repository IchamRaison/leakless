# Restitution cohérente — interface opt-in

Implémentation logicielle de la phase 3 V2, **pas un nouveau modèle entraîné**.
`Predictor.predict`, `Prediction` v0.1 et les livraisons V1 restent inchangés.
Aucun endpoint applicatif n'est remplacé ou déployé par ce module.

Statut de ce module : douze tests CPU sur faux backend passent dans le runtime
`b750f5d`, y compris l'entrée waveform et l'audit des poids en mémoire. Ils ne remplacent pas l'inférence réelle ni le lot
d'audit après la porte de parité. Le dispatcher canonique
`predict.preprocess_for_model(waveform, sample_rate, metadata)` est réutilisé.
Aucune tolérance ni performance nouvelle n'est revendiquée ici.

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

Pour le holdout numérique préparé, sans conversion de PCM32 en PCM16 :

```python
import numpy as np

recording = np.load(path_to_prepared_npy, allow_pickle=False, mmap_mode="r")
waveform = recording[start_sample:start_sample + 8000]  # offset du manifeste gelé
result = service.predict_waveform(waveform, sample_rate=8000)
```

`predict_waveform` reçoit les valeurs brutes, pas un chemin, un ID ou des labels.
La frontière numérique existante exige 8000 échantillons réels et finis à 8 kHz ;
elle refuse les signaux constants. Une copie privée protège la cohérence entre
score, mesure et génération ; son dtype et ses valeurs ne sont pas modifiés par
le wrapper. La normalisation appartient toujours au preprocessing du backend.
Le texte utilise le même `model_input_for_model`, collator `normalize=False` et
`model.generate(max_new_tokens=metadata["max_new_tokens"], max_time=15.0)` que la
voie WAV, sans modifier les quatre sources numériques liées au gate.

La voie WAV conserve son `audit.raw_api_payload` V1. Pour waveform, ce champ vaut
**null**, puisqu'aucune API WAV n'a été appelée : aucun payload V1 n'est fabriqué.
Le retour brut exact de la génération, ses erreurs et les comparaisons au
score/DSP restent dans le même audit et suivent les mêmes règles de restitution.
Une sortie textuelle invalide utilise le même gabarit explicite ; la raison
`raw_api_abstained`, propre au payload WAV, n'existe pas pour cette voie.

`input_sha256` reste le SHA des octets pour WAV. Pour waveform, il est défini par
`pipe-waveform-f64le-v1` : SHA256 du JSON compact trié
`{"format":"pipe-waveform-f64le-v1","sample_rate":8000,"shape":[8000]}`, suivi d'un
octet nul et des valeurs contiguës float64 little-endian. Cette canonicalisation
concerne **le hachage uniquement**, pas le signal transmis au modèle. L'audit
`input_identity` déclare schéma, fréquence, forme et dtype d'origine. Ce SHA
n'est ni celui d'un faux WAV ni celui du conteneur `.npy` : l'adaptateur de données
conserve séparément les empreintes des fichiers sources.

Le backend `Predictor` est chargé une fois et reste privé. Les requêtes du wrapper
sont sérialisées sans modifier les verrous V1 ; une requête concurrente reçoit
l'erreur existante `model_busy`. Le score et la génération utilisent ce même
backend. Le DSP emploie son dispatcher canonique et ses métadonnées : pas de
normalisation parallèle ni de version V1 sélectionnée implicitement.

`service.state_hashes()` expose uniquement les empreintes des états encodeur,
projecteur et Qwen sous le même verrou. L'export peut ainsi vérifier les poids en
mémoire avant/après l'audit sans charger un second modèle ni accéder au backend privé.

## Politique numérique V2

Le bundle V2 déclare `single_clip_acoustic_encoding: true` et le scoring
`class-continuation-logprob-sum-softmax-single-clip-v2`. Chaque clip passe
séparément dans l'encodeur/projecteur ; les interfaces acceptent toujours les
lots et rendent un score par clip. Le graphe de gradients est conservé pour
l'apprentissage. V1 garde sa politique historique par défaut.

Variante C opt-in : scoring `class-continuation-logprob-sum-softmax-single-clip-c1text-v2`,
`amplitude_evidence: true`, version `c1-amplitude-text-6sig-v1`. Les quatre séries
sont inchangées ; neuf mesures C1 sont ajoutées au prompt, jamais des labels ou
une prédiction de baseline. Les voies WAV/waveform recalculent ces mesures après
la même normalisation et conversion float32 que TimeF. `score_series` exige les
mesures séparées pour C et les refuse pour A. Cela ne démontre pas encore un
bénéfice du langage ni de l'accès aux séries par rapport à un gabarit.

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
connecteur et `harness/features.py` effectivement présents dans l'installation. Tout module numérique
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

Les fixtures waveform contrôlent le passage exact de valeurs float64 dépassant
PCM16, l'absence de décodage WAV, les sorties partagées avec le chemin WAV sur un
faux backend, le SHA indépendant de l'endianness et les erreurs/fallbacks. Elles
ne constituent pas une nouvelle démonstration GPU de parité. L'ajout change le
SHA de restitution : régénérer l'artefact de décision versionné après gel du code,
sans réajuster silencieusement le seuil.

Un gabarit juste par construction ne transforme pas les erreurs du modèle en
réussites. Cette couche n'améliore pas, à elle seule, rappel, fausses alertes,
calibration ou généralisation. Les latences incluent score et génération ; aucun
débit de surveillance continue n'est garanti. L'application devra adopter ce
nouveau contrat explicitement après concertation.
