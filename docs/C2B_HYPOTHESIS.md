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
