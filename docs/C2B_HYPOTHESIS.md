# C2B_HYPOTHESIS — définitions gelées avant toute évaluation

> **Ce document et le code qu'il décrit sont commités AVANT le premier calcul
> d'AUC ou de F1 de C2b sur validation ou test.** L'historique Git en est la
> preuve : ce commit ne contient aucun résultat. C'est la seule garantie
> crédible qu'aucun descripteur n'a été choisi en regardant un score.

---

## 1. La question, et une seule

> **Une représentation temporelle simple apporte-t-elle de l'information
> au-delà de C0, C1, C2 et C3 ?**

Pas « quelles features marchent le mieux ». Pas de pêche aux descripteurs.

## 2. Pourquoi cet échelon manquait

C0, C1, C2 et C3 sont **tous invariants à l'ordre des échantillons**. L'échelle
sautait donc d'un contrôle invariant à l'ordre directement au TSLM. Si le TSLM
dépassait C1, rien n'aurait permis de distinguer :

- « la modélisation temporelle apporte quelque chose », de
- « c'est simplement un meilleur extracteur de descripteurs ».

C2b ferme ce trou. Et il déplace le barreau : si C2b dépasse C1, l'information
temporelle est réelle **et bon marché**, ce qui est un résultat en soi.

## 3. Hypothèse, et sa motivation physique

> Une fuite est un écoulement turbulent continu : un bruit large bande
> stationnaire, faiblement modulé. Les sources qui composent la classe non-leak
> — moteurs, perceuses, chiens, pluie, alarmes — ont une structure rythmique.
> La **périodicité de l'enveloppe** devrait donc porter de l'information que ni
> la distribution d'amplitude (C1) ni le spectre agrégé (C2) ne peuvent voir.

Motivation formée sur le **fold train uniquement**, avant tout regard sur val ou
test. Pic d'autocorrélation d'enveloppe, médianes du train : *leak* 0,164 contre
*non-leak* 0,385.

## 4. Le piège évité — Wiener-Khintchine

L'autocorrélation de la **forme d'onde** est la transformée de Fourier inverse
de |FFT|². Elle ne contient donc rien de plus que le spectre de magnitude de C2 :
ce n'est pas un descripteur temporel, c'est C2 déguisé.

Seule l'**enveloppe**, une transformation non linéaire, brise cette équivalence.
C'est pourquoi la famille A opère sur l'enveloppe et jamais sur le signal.

## 5. Les six descripteurs, gelés

Audio **normalisé en RMS**, comme C1, C2, C3 et le jeu TimeF. Tous invariants au
gain — vérifié par test : multiplier l'entrée par 137 ne change aucune valeur.

| # | Descripteur | Famille | Définition |
|---|---|---|---|
| 1 | `env_ac_peak` | A | maximum de l'autocorrélation **non biaisée** de l'enveloppe sur la région gelée |
| 2 | `mod_peak_hz_log` | B | log₁₀ de la fréquence du pic dominant du spectre de modulation, même région |
| 3 | `mod_1_4` | B | énergie relative du spectre de modulation, 1-4 Hz |
| 4 | `mod_4_16` | B | énergie relative, 4-16 Hz |
| 5 | `mod_16_64` | B | énergie relative, 16-64 Hz |
| 6 | `spectral_flux` | C | variation L1 moyenne image à image du vecteur de bandes, chaque image normalisée en L1 |

**Six, pas un de plus.** Le cahier des charges autorisait 4 à 6.

### Paramètres gelés

| | Valeur | Justification, posée d'avance |
|---|---|---|
| Région de recherche | lags 32 à 2000 = **4 à 250 Hz** | sous 4 Hz, un clip d'une seconde offre moins de quatre cycles ; au-dessus de 250 Hz l'enveloppe redressée n'est plus fiable à 8 kHz |
| Bandes de modulation | **1-4, 4-16, 16-64 Hz** | grille logarithmique à facteur 4, usage courant en analyse de modulation |
| Enveloppe | `|x|`, moyenne retirée | le redressement suffit à briser Wiener-Khintchine, sans dépendance supplémentaire |
| Fenêtres du flux | **256 échantillons (32 ms), recouvrement 50 %** | 62 images par clip ; bandes identiques à celles de C2, réutilisées et non réinventées |
| Normalisation des bandes | par leur somme sur 4-250 Hz | une **forme**, pas un niveau : le niveau appartient à C0 |

### Deux corrections d'estimateur, décidées avant évaluation

1. **Autocorrélation non biaisée.** `np.correlate` renvoie l'estimateur biaisé,
   qui décroît en `(N − lag)/N` et favorise mécaniquement les modulations
   rapides. Mesuré sur signal de synthèse : une modulation à 7 Hz (lag 1143)
   perdait contre un pic de bruit au lag 58. On divise par le nombre de termes
   réellement sommés.
2. **La fréquence vient du spectre de modulation, pas de l'autocorrélation.**
   L'autocorrélation pique à chaque multiple de la période : en tirer une
   fréquence donne l'erreur d'octave classique. Mesuré : 55 Hz lu comme 27,3 Hz.
   Le spectre de modulation a une résolution de 1 Hz et son fondamental domine.
   Après correction, l'erreur de récupération est nulle à 7, 20, 55 et 120 Hz.

## 6. Ce qui est délibérément absent

Ni RMS, ni facteur de crête, ni descripteur spectral statique. C0, C1 et C2 les
couvrent déjà, et les réintroduire rendrait C2b incomparable aux autres échelons.

## 7. Modèle et protocole — identiques aux autres échelons

| | |
|---|---|
| Modèle | régression logistique, rien d'autre |
| Split | `manifests/split_v2.csv`, `sha256 7a8716a352844342…` |
| Standardisation | ajustée sur le **train** uniquement |
| Hyperparamètre `C` | choisi sur la **validation**, au niveau cluster |
| Seuil | argmax du macro-F1 cluster-level sur la **validation** |
| Évaluation | le moteur unique, `evaluate_predictions.py` |
| Réglage sur le test | **aucun** |

## 8. Redondance interne, dite franchement

Les familles A et B sont des paires de Fourier : le spectre de modulation est la
FFT de l'enveloppe, l'autocorrélation d'enveloppe en est le module au carré
retransformé. A donne une statistique de maximum, B des intégrales de bande.
Elles sont **corrélées, pas redondantes** — et ce n'est pas un ensemble de six
descripteurs indépendants. Il ne faut pas le présenter comme tel.

## 9. Ce que les stress tests diront, et ne diront pas

C2b sera évalué sur T0, T1, T2 et T3 **sans réentraînement** : le modèle est
ajusté sur T0/train et appliqué tel quel.

Attendu, et à vérifier plutôt qu'à supposer :

| | Effet attendu sur C2b | Pourquoi |
|---|---|---|
| T1 inversion | **faible** | l'autocorrélation et le flux sont presque invariants au renversement ; seule une asymétrie temporelle bougerait |
| T2 permutation de blocs | **fort** | la périodicité au-delà de 31,25 ms est détruite |
| T3 phase randomisée | **fort** | l'enveloppe est entièrement refaite, alors que `\|FFT\|` est préservé |

⚠️ **T1 est un stress faible pour des descripteurs temporels peu profonds.** Un
score inchangé sous T1 ne prouve pas l'absence de sensibilité temporelle. Et C2
lui-même n'est pas exactement invariant sous T3 : `_spectrum` applique une
fenêtre de Hann, et T3 ne préserve le `\|FFT\|` que du signal non fenêtré (écart
mesuré jusqu'à 21 % sur une bande étroite). Un petit déplacement de score sous
T3 n'est donc pas une preuve.

## 10. Verdicts autorisés

Avec 41 clusters indépendants en test, seuls trois énoncés sont permis :

- *evidence compatible with improvement*
- *inconclusive*
- *evidence compatible with degradation*

Aucune affirmation de significativité statistique.

## 11. Tests unitaires qui accompagnent ce gel

`tests/run_tests.py` — dix tests, aucun ne touchant val ou test :

`c2b_has_exactly_six_features` · `c2b_is_gain_invariant` ·
`c2b_is_order_sensitive` · `c2b_sees_what_c2_structurally_cannot` ·
`c2b_recovers_a_known_modulation_frequency` · `c2b_modulation_bands_are_proportions` ·
`c2b_search_region_matches_declared_bounds` ·
`c2b_periodicity_strength_is_low_on_white_noise` · `c2b_is_deterministic` ·
`c2b_is_in_the_ladder`


---

# RÉSULTATS — ajoutés au commit B, après le gel

> Les définitions ci-dessus ont été commitées en `355f074`, qui ne contient
> aucun chiffre. Tout ce qui suit a été mesuré ensuite.

## Échelle complète — test, 41 clusters (30 leak / 11 non-leak)

| run | clip AUC | clip PR-AUC | clip F1 | cluster AUC | cluster F1 | Brier |
|---|---|---|---|---|---|---|
| C1 enveloppe | **0,902** | 0,878 | **0,856** | **0,927** | 0,849 | **0,140** |
| **C2b temporel** | 0,847 | 0,829 | 0,763 | 0,915 | 0,814 | 0,160 |
| C3 baseline | 0,824 | 0,792 | 0,758 | 0,900 | **0,856** | 0,173 |
| C2 spectral | 0,710 | 0,711 | 0,680 | 0,779 | 0,776 | 0,214 |

## Comparaisons appariées — mêmes tirages de clusters

| comparaison | métrique | Δ observé | IC95 | lecture |
|---|---|---|---|---|
| C2b − C1 | clip AUC | −0,055 | [−0,164, +0,058] | inconclusive |
| C2b − C1 | cluster AUC | −0,012 | [−0,157, +0,157] | inconclusive |
| C2b − C2 | clip AUC | +0,137 | [−0,004, +0,269] | inconclusive |
| C2b − C2 | cluster AUC | +0,136 | [−0,061, +0,368] | inconclusive |
| C2b − C3 | clip AUC | +0,023 | [−0,154, +0,123] | inconclusive |
| C2b − C3 | cluster AUC | +0,015 | [−0,152, +0,225] | inconclusive |

## Stress temporel — même définition figée, jamais ajustée sur les données stressées

> **Ce qui est réellement fait.** Il n'existe **aucun checkpoint sérialisé** de
> C2b. La définition du modèle est figée (`355f074`) ; la régression logistique
> est réajustée de façon déterministe sur T0/train, avec les mêmes
> hyperparamètres (`C = 0,1`, choisi sur T0/val) et les mêmes données, dans le
> même processus que l'évaluation sous stress. Elle n'est **jamais** ajustée sur
> T1, T2 ou T3 : les transformations ne s'appliquent qu'aux descripteurs
> d'évaluation. `retrained: false` signifie exactement *not retrained on stressed
> data*. L'identité du modèle est vérifiée par une empreinte des paramètres
> ajustés (`model_fingerprint`), identique dans les quatre runs.

> ### ⚠️ Résultats T2 et T3 régénérés après un correctif de reproductibilité
>
> Les premiers chiffres T2/T3 ont été produits avec un `clip_rng` qui dérivait sa
> graine de `hash()`, salé par processus : les transformations n'étaient pas
> reproductibles d'une exécution à l'autre. Corrigé en `b23601a` (dérivation
> SHA-256 canonique). **T0 et T1 sont inchangés au bit près** — ils n'utilisent
> pas le générateur. Les anciens T2/T3 sont **SUPERSEDED**.

Métriques **indépendantes du seuil** uniquement, sur le test (194 clips, 41
clusters). Corrélation et Δp : clip à clip, orientés stress − T0.

| | clip AUC | cluster AUC | corrélation des probabilités avec T0 | Δp médiane [Q1, Q3] | statut |
|---|---|---|---|---|---|
| T0 original | 0,847 | 0,915 | 1,000 | — | inchangé |
| T1 inversion | 0,849 | 0,906 | **0,992** | +0,001 [−0,014, +0,017] | inchangé |
| ~~T2 (ancien, RNG instable)~~ | ~~0,819~~ | ~~0,845~~ | ~~0,748~~ | — | 🔴 SUPERSEDED |
| **T2 permutation de blocs** | **0,751** | **0,785** | **0,729** | +0,048 [−0,045, +0,196] | régénéré |
| ~~T3 (ancien, RNG instable)~~ | ~~0,726~~ | ~~0,679~~ | ~~0,724~~ | — | 🔴 SUPERSEDED |
| **T3 phase randomisée** | **0,726** | **0,697** | **0,705** | +0,022 [−0,077, +0,144] | régénéré |

Sur T3 l'AUC clip tombe au même arrondi à trois décimales (0,72619 contre
0,726297) : c'est une coïncidence, les 1000 prédictions diffèrent.

| comparaison appariée stress − T0 | Δ clip AUC | IC95 | lecture |
|---|---|---|---|
| T1 − T0 | +0,002 | [−0,011, +0,026] | inconclusive |
| ~~T2 − T0 (ancien)~~ | ~~−0,028~~ | ~~[−0,113, +0,024]~~ | ~~inconclusive~~ 🔴 |
| **T2 − T0** | **−0,097** | **[−0,192, −0,031]** | **compatible with degradation under stress** |
| **T3 − T0** | **−0,121** | **[−0,316, −0,059]** | **compatible with degradation under stress** |

> **Après remplacement du RNG non reproductible, l'effet T2 régénéré est mesurable
> sous le protocole de stress prédéfini. Les résultats T2/T3 antérieurs sont
> SUPERSEDED.**

> ℹ️ **Le seuil des runs stressés.** Le moteur applique la même règle à tous les
> runs : seuil recalculé sur le fold `val` *du run*. Pour un run stressé, ce fold
> est lui aussi transformé (seuil 0,448 sous T2 contre 0,496 sous T0). Les
> métadonnées le déclarent désormais exactement. **Aucune métrique dépendante du
> seuil (macro-F1, exactitude) ne sert à une conclusion de sensibilité
> temporelle** : les conclusions ci-dessus reposent sur l'AUC, la corrélation des
> probabilités et la distribution de Δp.

Le §9, figé en `355f074`, prévoyait qualitativement un effet faible sous T1 et
fort sous T2 et T3, sans seuil chiffré. Les mesures sont dans ce sens ; ce n'est
pas un test d'hypothèse.

## La réponse à la question unique

> ### « Does shallow temporal structure add information beyond the strongest static/acquisition control C1? »
>
> ## **INCONCLUSIVE**

Toutes les estimations ponctuelles favorisent C1 — clip AUC −0,055, cluster AUC
−0,012, macro-F1 −0,093 — mais chaque intervalle de confiance contient zéro.
Avec 41 clusters indépendants, et 11 seulement du côté non-leak, aucun de ces
écarts n'est distinguable du bruit. **Nous ne pouvons pas dire que la structure
temporelle simple apporte quelque chose au-delà de C1, ni qu'elle n'apporte rien.**

Ce qui est en revanche établi :

1. **Les prédictions de C2b sont sensibles aux perturbations d'ordre temporel et
   de phase.** T2 et T3 montrent que les prédictions de C2b (corrélation avec T0
   de 0,729 et 0,705) et sa performance de discrimination (Δ clip AUC −0,097 et
   −0,121, IC95 appariés sous zéro) changent quand l'organisation temporelle est
   perturbée. T3 conserve `|FFT|` à 3e-16 près : les prédictions de C2b ne sont
   donc pas une fonction du seul spectre d'amplitude. **Ces transformations ne
   sont pas des augmentations physiques démontrées comme préservant l'étiquette :
   elles n'établissent pas la pertinence physique causale de l'information
   temporelle.**
2. **C1 reste le barreau à franchir**, et il n'a pas bougé. Un TSLM qui
   dépasserait C2b sans dépasser C1 n'aurait rien démontré.
3. **C2b et C3 sont indiscernables** (Δ clip AUC +0,023, IC95 traversant zéro).
   La baseline historique n'apportait donc rien de plus qu'une représentation
   temporelle à six descripteurs.
