# Protocole interne figé — évaluation finale sur les tests

**Date : 31 octobre 2026.** Rédigé et commité **avant** toute ouverture des tests pour les runs
finaux. Décision de l'auteur du 31/10 (« oui pour tout ») : périmètre figé, recadrage de
l'article, ouverture unique du test.

## 1. Code et données figés

- Code : `code_sha256 = d80bc7ce063551bc01533296775fdada56634d8b6a53360eb604e4941ba46856`
  (commit `ce74935` et suivants : seuls des scripts shell, la documentation et ce fichier peuvent
  changer ; aucun `.py` ni `.yaml` de `laya_slots/`, `experiments/`, `configs/`). Chaque run final
  porte ce hash dans `metrics.json` ; un run avec un autre hash n'est pas un run final.
- Données : manifestes `data/processed/<jeu>/<langue>/MANIFEST.json` inchangés depuis la phase 1
  (`data_sha256` dans chaque run).
- Modèles : révisions de `MODELES.lock.json` (Laya `7b928d8`, Qwen3.8-27B `1d4bf0f`,
  Gemma-4-31B-it `842da37`, Qwen3.8-Flash-Next-FP8 `236dfdf`, mDeBERTa-v3-base `a048466`, déjà dans
  le lock parce que GLiNER en dépend) ; les poids du tagger viennent de la conversion safetensors
  `refs/pr/5` (`faf660b`).
- Checkpoints : ceux de `checkpoints/{c2,c3v2,c3,tagger}/` entraînés au plus tard le 31/10,
  choisis sur le dev. **Aucun réentraînement** après l'ouverture du test.

## 2. Grille principale (modèles monolingues)

Jeux × langues : MultiATIS++, MTOP, Multi3WOZ × EN, FR (6 couples). Split `test`.

| Condition | Graines | Candidats |
|---|---|---|
| C0 tagger mDeBERTa-v3 | 13, 14, 15 | réels seulement (sortie du tagger) |
| C1 Laya zero-shot | 13, 14, 15 | oracle (grille) et réels |
| C2 Laya fine-tuné | 13, 14, 15 | oracle (grille) et réels |
| C3 Laya FT + augmentation v2 | 13–17 | oracle (grille) et réels |
| C4 Qwen3.8-27B, Gemma-4-31B-it, Qwen3.8-Flash-Next-FP8 | 13, 14, 15 | oracle (grille) et réels |

Grille oracle : `hard_no_overlap`, k = 5, p_absent = 0. Candidats réels : `configs/candidats/reels.yaml`
(lexique, GLiNER, ontologie, historique), **sans le tagger** (décision du 31/10).
Réglages Multi3WOZ : `requests.intent=domain requests.filled=state_before` (automatiques dans le Makefile).

## 3. Analyses (sur le test, mêmes checkpoints)

- Balayages k ∈ {2, 5, 10, 20} × p_absent ∈ {0, 0,2, 0,5}, MultiATIS++ EN : C1 (13), C2 (13–15),
  C3 v2 (13–17), C3 v1 (13–17, ablation) ; C3 v1 aussi avec candidats réels.
- Ablation `hard` (MultiATIS++ EN, grille) : C1 (13), C2 entraîné en `hard` (13–15), C3 v2 évalué en
  `hard` (13–15).
- Transfert (QR3), candidats oracle et réels, évalués sur le test FR : EN → FR (C2 13–15, C3 v2
  13–17, 3 jeux) et EN+FR → FR et → EN (checkpoints bilingues existants seulement : C2 13–15,
  Multi3WOZ 13–14 ; C3 v2 MultiATIS++ et MTOP 13–17, Multi3WOZ 13–15).
- Décision (QR4) : `experiments.decision --dev <run dev> --test <run test>` pour chaque juge avec
  candidats réels, graine 13 : température, isotone et seuils ajustés sur le dev, appliqués tels
  quels au test ; risques 0,5 %, 1 %, 2 % et 5 % (2 % et 5 % dans le texte).
- Latence : mesurée sur le dev (`scripts/latence.sh`, 31/10) ; pas de nouvelle mesure sur le test.

Runs dev complémentaires lancés avec la série (aucune décision n'en dépend, ils servent seulement
à la calibration et au tableau dev) : C1 et les 3 LLM sur MultiATIS++ FR, graine 13, oracle et réels.

## 4. Statistiques

- Intervalles à 95 % : bootstrap par grappes (énoncés ; dialogues pour Multi3WOZ), 1 000 tirages.
- Comparaisons appariées : test de permutation par grappes, 10 000 permutations, bilatéral ;
  graine 13 contre graine 13 (puis 14–14, 15–15 en complément).
- Familles de Holm (métriques : F1 des slots, taux de faux positifs, précision du choix) :
  1. C2 contre chacun des 3 LLM : 6 couples × 2 candidats × 3 LLM × 3 métriques = 108 tests ;
  2. C3 contre C2 : 6 × 2 × 3 = 36 tests ;
  3. C0 contre le meilleur juge Laya (C2 ou C3, choisi sur le **dev**) avec candidats réels :
     6 × 3 = 18 tests.
- Rien n'est retiré a posteriori : tous les runs finaux sont rapportés, y compris défavorables.

## 5. Règles d'exécution

- Les runs finaux s'écrivent dans `results/final-*`. Les runs antérieurs sur le test
  (exploration de la phase 0) reçoivent un fichier `OBSOLETE` et ne sont pas utilisés.
- Un run qui échoue pour une raison technique (mémoire, coupure) est relancé à l'identique et
  l'incident est noté dans le journal du README. Aucun réglage n'est modifié en fonction d'un
  résultat du test.
- Ordre : grille Laya et C0, analyses, puis les LLM ; script `scripts/test_final.sh`.
