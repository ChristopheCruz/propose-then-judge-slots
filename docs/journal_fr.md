# Laya comme juge de slots

**L'extracteur propose, le juge tranche.** Ce projet de recherche étudie si un petit modèle de décision calibré, [Laya](https://huggingface.co/convaiinnovations/laya) (environ 400 M de paramètres), peut remplir les slots d'un système de dialogue en choisissant parmi des candidats générés dynamiquement, à une fraction du coût d'un LLM. L'étude porte sur l'anglais et le français, sur trois benchmarks publics : MultiATIS++, MTOP et Multi3WOZ.

> Projet en cours. Les développements sont décrits dans [État d'avancement](#état-davancement) ; aucun résultat n'est encore publié.

## Motivation

Pour remplir les slots d'un assistant (« un vol de **Lyon** à **Paris** **jeudi matin** »), les approches récentes s'appuient sur des LLM génératifs. Elles sont efficaces, mais chères, lentes, sujettes à l'hallucination, et leurs scores de confiance sont rarement exploitables.

Nous proposons de séparer deux rôles :

1. **Proposer** : des générateurs bon marché (tagger, GLiNER, lexique, ontologie ou base de données, historique du dialogue) produisent des valeurs candidates pour chaque slot.
2. **Trancher** : un juge léger choisit une valeur parmi ces candidats, ou répond « non précisé ». Il fournit aussi une probabilité calibrée, qui permet de décider s'il faut agir ou demander une précision.

Le juge ne peut répondre que par un candidat ou par une abstention : il n'invente pas de valeur.

![Protocole expérimental](figures/protocole_experimental.png)

## Questions de recherche

| | Question |
|---|---|
| **QR1 — Efficacité** | Laya fine-tuné approche-t-il un LLM-juge open-weight en F1 de slots, pour un coût et une latence nettement inférieurs ? |
| **QR2 — Décomposition** | Quelle part de l'erreur vient du rappel des candidats, et quelle part du choix ? Comment varient-elles avec le nombre de candidats *k* et le taux de valeur correcte absente ? |
| **QR3 — Généralisation** | Le juge tient-il sur des intentions, slots ou domaines jamais vus, et en transfert de l'anglais vers le français ? |
| **QR4 — Abstention** | La décision en deux temps (le slot est-il exprimé ? puis lequel ?) et la calibration améliorent-elles la courbe risque-couverture ? |
| **QR5 — Entraînement** | L'augmentation par options aléatoires (ordre, nombre, distracteurs, valeur absente) rend-elle le juge plus robuste quand le contexte change ? |

### Hypothèses

- **H1.** Avec les candidats oracle, Laya fine-tuné reste à moins de 3 points de F1 du LLM-juge, avec une latence au moins 10 fois plus faible.
- **H2.** De bout en bout, le rappel des candidats est la première source d'erreur.
- **H3.** L'augmentation par options aléatoires réduit l'écart entre schémas vus et non vus.
- **H4.** La décision en deux temps réduit l'AURC par rapport à une option « non précisé » unique.

Les seuils de H1 seront figés dans un protocole interne daté, avant la grille d'expériences principale.

## Protocole

### Benchmarks

| Jeu | Langues | Tours | Rôle dans l'étude |
|---|---|---|---|
| MultiATIS++ | EN, FR | un seul | slots plats, un domaine ; point de départ |
| MTOP | EN, FR | un seul | multi-domaines, nombreuses intentions ; schémas non vus |
| Multi3WOZ | EN, FR | multi-tours | état de dialogue, reprise des slots, contexte dynamique |

Les trois jeux annotent les slots comme des extraits de texte. Leur conversion au format « choisir parmi des candidats » fait partie de la contribution : elle est décrite et testée dans `laya_slots/data/`.

### Conditions comparées

| Code | Condition | Ce qu'elle mesure |
|---|---|---|
| C0 | Tagger seul (encodeur multilingue en étiquetage BIO, style JointBERT) | baseline classique d'extraction |
| C1 | Laya zero-shot | plancher du juge, sans adaptation |
| C2 | Laya fine-tuné | effet du fine-tuning |
| C3 | Laya fine-tuné + augmentation par options aléatoires | contribution principale |
| C4 | LLM-juge open-weight (vLLM), zero-shot et few-shot | référence haute pour le même rôle |
| C5 | LLM extracteur génératif (sortie JSON) | alternative sans candidats |

Chaque condition est évaluée avec deux sortes de candidats :
- **oracle** : la bonne valeur plus des distracteurs, pour mesurer le juge seul ;
- **réels** : la sortie des générateurs, pour mesurer de bout en bout.

Axes de variation :
- *k* ∈ {2, 5, 10, 20} ;
- taux de valeur absente ∈ {0, 20, 50 %} ;
- distracteurs faciles ou difficiles ;
- transfert EN→FR ;
- domaines laissés de côté.

### Métriques

- **Officielles** : F1 de slots, exactitude d'intention et de trame ; Joint Goal Accuracy pour Multi3WOZ.
- **Décomposées** : rappel des candidats @k, exactitude du choix quand la bonne valeur est présente.
- **Calibration et abstention** : ECE, Brier, courbe risque-couverture (AURC).
- **Coût** : latence p50/p95, débit, mémoire, heures-GPU.
- **Statistiques** : intervalles de confiance à 95 % par bootstrap sur les dialogues, tests de permutation appariés avec correction de Holm.

## Organisation du dépôt

| Chemin | Contenu |
|---|---|
| `laya_slots/schema.py`, `interfaces.py` | Format unifié (`Example`, `Candidate`, `JudgeRequest`, `Judgment`) et protocoles communs |
| `laya_slots/data/` | Chargeurs MultiATIS++, MTOP, Multi3WOZ vers le format unifié ; préparation et manifestes hachés |
| `laya_slots/normalize/` | Normalisation des valeurs (dates, nombres…) avant comparaison |
| `laya_slots/candidates/` | Générateurs : oracle, distracteurs, lexique, ontologie, historique, GLiNER, union, top-k par embeddings ; mesure du rappel |
| `laya_slots/judges/` | Juges : Laya (une passe ou deux temps), LLM-juge (vLLM ou serveur compatible OpenAI), baselines |
| `laya_slots/evaluation/` | Métriques |
| `laya_slots/train/` | Fine-tuning de Laya : items construits comme à l'inférence, boucle d'entraînement, sauvegarde au format Laya |
| `laya_slots/decision/` | Calibration (température, isotone), décision à trois zones, courbes risque-couverture (`experiments/decision.py`) |
| `experiments/` | Lancement d'un run (`run.py`), fine-tuning (`train.py`), configs OmegaConf, tableaux CSV/LaTeX, téléchargement des modèles |
| `configs/` | Pilote, socle commun, candidats oracle/réels, conditions C0–C5, entraînement (`configs/train/`) |
| `scripts/` | `preparer.sh` (installation sur la machine), `labo.sh` (runs sur le serveur GPU par SSH) |
| `notebooks/` | Exemples Colab de Laya (triage de tickets, intentions et slots en contexte dynamique) |
| `docs/plan_de_recherche.docx` | Plan de recherche complet |
| `CONTRAT.md` | Contrat de développement : interfaces, conventions, répartition des répertoires |

Chaque run écrit son `config.yaml` résolu, `predictions.jsonl` (une ligne par requête au juge) et `metrics.json` (métriques, commit, hachage de la config et des données).

## Démarrage rapide

Prérequis : Linux, Python 3.11–3.12, [uv](https://docs.astral.sh/uv/) et, pour les expériences, des GPU NVIDIA. Le projet est développé sur 2×H200.

```bash
make preparer        # 1er appel : crée .env (à compléter : HF_HOME, LLM_MODEL…), puis s'arrête
make preparer        # environnement (uv.lock), modèles -> $HF_HOME, données, tests CPU et Laya réel
make verifier-gpu
make pilote
make c1 DATASET=multiatis LANGUE=en ARGS="data.limit=200"
make c4-dev ARGS="data.limit=200"      # LLM de mise au point sur une carte
make c4                                 # LLM de référence sur les deux H200
make entrainer-c2 && make c2            # fine-tuning de Laya, puis évaluation (idem c3)
make tableaux                           # agrège results/ en CSV et LaTeX
```

- `make aide` liste toutes les cibles. Les étapes se relancent une à une : `make installer`, `make modeles`, `make donnees`, `make verifier`.
- Les révisions des modèles téléchargés sont écrites dans `MODELES.lock.json` (versionné). Les runs se font hors ligne (`HF_HUB_OFFLINE=1`).
- Toute clé de config se surcharge en ligne de commande : `ARGS="data.limit=200 judge.batch_size=64"`.
- Les tests (`make tests`) tournent sur CPU, sans réseau ni modèle. Les tests avec les vrais modèles sont marqués `integration` (`make tests-integration`).

### Données

- **MTOP** et **Multi3WOZ** sont téléchargés et convertis automatiquement par `make donnees`.
- **MultiATIS++** est distribué par le LDC (LDC2021T04) et n'est pas redistribué ici. Placer les TSV de la copie LDC sous `data/raw/multiatis/` avant `make donnees`.

### Runs sur le serveur distant

`scripts/labo.sh` lance les expériences sur le serveur GPU par SSH, chacune sur un instantané figé du code, et sauvegarde les résultats sur GitHub :

```bash
./scripts/labo.sh init     # crée labo.env (serveur, dépôt GitHub, commande d'expérience)
./scripts/labo.sh setup    # prépare le serveur
./scripts/labo.sh run configs/pilote.yaml --gpus 0
```

Pour passer par le Makefile, mettre dans `labo.env` : `EXP_CMD='make run CONFIG="$CONFIG" SORTIE="$RUN_DIR" GPUS="$CUDA_VISIBLE_DEVICES"'`. `./scripts/labo.sh aide` détaille les autres commandes.

## État d'avancement

**Fait :**
- format unifié et conversion des trois jeux en EN et FR ;
- normalisation des valeurs ;
- générateurs de candidats et mesure du rappel ;
- juges Laya, LLM-juge et baselines ;
- pipeline d'expérience, métriques de base, tableaux ;
- CI sur CPU ;
- scripts d'installation et de lancement ;
- fine-tuning de Laya (C2) et augmentation par options aléatoires (C3) : code en place, premier essai C2 concluant le 22 octobre 2026.
- intervalles de confiance bootstrap par grappes (dialogues ou énoncés), tests de permutation appariés et correction de Holm (`make intervalles`, `make comparer`).

**À venir :**
- tagger BIO (C0) et extracteur LLM (C5) ;
- tests de robustesse ;
- C3, balayages k × p_absent, autres jeux et langues, 3 à 5 graines ;
- LLM de référence Qwen3-235B (C4) sur les deux H200.

**Premiers runs GPU (22 octobre 2026) :** les runs de l'après-midi sont invalidés par un bug de configuration des candidats oracle. Après correction, première comparaison valide sur tout le test MultiATIS++ EN (oracle k = 5, une graine) : Laya fine-tuné (C2) F1 0,919, Qwen3-32B zero-shot 0,276, Laya zero-shot 0,012. Détails dans le journal.

| Jalon | Date visée |
|---|---|
| Pipeline de bout en bout (MultiATIS++ EN, oracle, Laya et LLM-juge) | 13 novembre 2026 |
| Données gelées et protocole interne daté | 20 novembre 2026 |
| Grille C0–C5 complète sur les trois jeux | 8 janvier 2027 |
| Soumission (ACL Rolling Review) | mars 2027 |

## Journal d'expérimentation

Ce journal conserve, run après run, ce qu'il faut pour rédiger l'article : cadre, versions, décisions de protocole, résultats, constats et incidents. Les dates sont absolues. Un résultat n'est cité comme résultat que s'il est marqué **valide**.

> **État au 23 octobre 2026 : premiers résultats valides de la grille principale** (`hard_no_overlap`) et de l'ablation `hard` sur MultiATIS++ EN : C1, C2, C3, C4-dev, tout le test, une graine (§ 4). Balayages k × p_absent : H3 soutenue avec l'oracle (motif reproduit sur chaque graine). Candidats réels (26/10) : H3 non soutenue avec C3 v1 ; **C3 v2** (confusion entre slots) fait jeu égal avec C2 sur candidats réels et garde l'avantage sur les balayages (dev). C3 v2 est retenue pour la grille. Il manque les autres graines, les candidats réels, les autres jeux et langues, et la référence 235B. Le LLM est encore le modèle de mise au point (Qwen3-32B), pas la référence 235B. Les runs du 22/10 après-midi restent invalidés ou biaisés par le bug `candidates.fit: false`.

### 1. Cadre matériel et logiciel

**Matériel**

| Élément | Valeur |
|---|---|
| Machine | `caire-gpu2` |
| GPU | 2 × NVIDIA H200 NVL, 143 771 Mio chacune |
| Pilote NVIDIA | 595.71.05 |
| Disque | un disque de 2,2 To |
| Cache Hugging Face | `/home/ubuntu/hf-cache` (`HF_HOME`) |

- Le code est développé sur un poste local (dépôt git local, pas de GitHub). Il est copié sur la machine par `rsync`, puis installé par `make preparer` (uv).
- Pas de CUDA toolkit système (pas de `nvcc`). Le runtime CUDA 13.0 vient des paquets pip.

**Logiciel** (versions de `uv.lock`)

| Paquet | Version |
|---|---|
| Python | 3.12.15 |
| torch | 2.13.0 |
| vLLM | 0.30.0 |
| laya | 0.3.22 |
| transformers | 5.16.1 |
| gliner | 0.2.29 |
| sentence-transformers | 6.1.0 |
| flashinfer-python | 0.6.18.post1 |
| omegaconf | 2.3.1 |
| Runtime CUDA | 13.0 (pip) |

### 2. Modèles (révisions)

Révisions Hugging Face de `MODELES.lock.json` (généré le 22 octobre 2026 à 14:36 UTC).

| Modèle | Révision | Taille | Rôle | Statut au 22/10 |
|---|---|---|---|---|
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | `e156cb4efae4` | 236 Go | LLM de référence C4/C5, sur 2 cartes | téléchargé, pas encore lancé |
| `Qwen/Qwen3-32B` | `9216db5781bf` | 65 Go | mise au point C4-dev, 1 carte (61 Gio chargé) | utilisé |
| `convaiinnovations/laya` | `7b928d828b7b` | 1,5 Go | juge C1 (checkpoint `multilingual`) | utilisé |
| `urchade/gliner_multi-v2.1` | `443d26d654e0` | 2,3 Go | générateur de candidats | pas encore utilisé |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c211226b` | 0,5 Go | top-k par embeddings | pas encore utilisé |
| `microsoft/mdeberta-v3-base` | `a0484667b223` | 0,0 Go (sic) | encodeur prévu pour C0 | à vérifier |

### 3. Protocole des runs du 22 octobre 2026 et décisions

**Données et requêtes**

| Paramètre | Valeur |
|---|---|
| Jeu | MultiATIS++ EN, split test |
| Sous-ensemble | 200 premiers exemples (`data.limit=200`) |
| Graine | 13 |
| Intention | gold |
| Requêtes | `requests.slots=intent` : tous les slots autorisés par l'intention |
| Volume | ≈ 62 requêtes par énoncé, 12 355 requêtes |
| Paires gold | 746 ; 727 requêtes portent sur un slot exprimé |
| Slots non exprimés | 11 628 requêtes, soit 94 % |

**Candidats oracle**
- k = 5, p_absent = 0.
- Distracteurs `hard` : stratégies `train_values`, `ontology`, `other_slots`, n-grammes de l'énoncé (≤ 3 mots), `history` ; sélection par similarité de trigrammes avec la gold.
- Distracteurs `easy` : valeurs aléatoires prises dans tous les slots.

**Juges**

| Condition | Réglages |
|---|---|
| Pilote | `StringMatchJudge` (correspondance de chaînes), lot 64 |
| C1 | Laya, mode `single`, checkpoint `multilingual`, ordre des candidats mélangé, lot 128, `cuda` |
| C4-dev | Qwen3-32B via vLLM, backend logprobs, options par lettres, zero-shot, `enable_thinking=false`, `max_model_len=4096`, lot 256, ordre mélangé |

**Métriques**
- Principale : F1 micro des paires (slot, valeur), NONE exclu.
- Décomposition : exactitude du choix quand la gold est présente (« choix|présent ») ; taux de faux positifs (FP) sur les slots non exprimés.

**Décisions de protocole prises**
- Avec 94 % de requêtes non exprimées, le F1 mesure surtout l'abstention. On rapporte donc toujours côte à côte F1, choix|présent et taux de FP.
- `metrics.json` contient un bloc `settings` (limit, generators, k, p_absent, distractors, judge, judge_model…). `make tableaux` regroupe par condition, jeu, langue, modèle, distracteurs, k, p_absent et limite, et ignore les runs en échec.
- Nouveau préréglage `hard_no_overlap` : il exclut les distracteurs qui contiennent la gold ou y sont contenus (en mots entiers), comme « on april » ou « april 1 » pour « april ».
- **Décision du 23 octobre 2026 : `hard_no_overlap` pour la grille principale, `hard` en ablation.** Raison : choisir une variante de frontière de la gold est une erreur de découpage, pas de compréhension. Avec les candidats oracle, ces variantes sont fabriquées par nous ; avec les candidats réels, elles apparaîtront d'elles-mêmes si un générateur découpe mal. L'ablation `hard` mesure la sensibilité de chaque juge aux erreurs de frontière. Mise en œuvre : `configs/candidats/oracle.yaml` passe à `hard_no_overlap` ; `make … DISTR=hard` lance l'ablation (checkpoints C2/C3 suffixés `-hard`).

### 4. Chronologie des runs et résultats

#### 22 octobre 2026 — premiers runs GPU (invalidés)

Code : commit `143f163` sur la machine (arbre modifié, `git_dirty=True`). Résultats copiés dans le commit local `ce4e476`.

**Bug découvert le soir même.** `configs/candidats/oracle.yaml` et `configs/pilote.yaml` avaient `candidates.fit: false`. Le générateur oracle n'était pas ajusté sur le train : les réservoirs de distracteurs `train_values` et `random` étaient vides.
- **`easy` : invalide.** En moyenne 0,06 candidat par requête au lieu de 5 (aucun pour les slots non exprimés, la seule gold pour les exprimés). Le juge choisissait entre la gold et « non précisé » ; l'abstention était triviale.
- **`hard` : biaisé.** Les listes avaient bien 5 candidats, mais sans la stratégie `train_values`. Les distracteurs venaient plus souvent de l'énoncé.
- **Correction** : `fit: true` dans les deux configs, et garde-fou : le sampler lève une erreur si une stratégie exige l'ajustement et qu'il n'a pas eu lieu.
- **Tous les runs doivent être relancés.**

Effet du bug, mesuré en local avec le juge par correspondance de chaînes (pilote) :

| Distracteurs | fit | F1 | choix\|présent | FP |
|---|---|---|---|---|
| easy | non (bug) | 0,931 | 0,879 | 0 |
| easy | oui | 0,705 | 0,868 | 0,036 |
| hard | non (bug) | 0,005 | 0,044 | 0,998 |
| hard | oui | 0,015 | 0,132 | 0,989 |

**Runs du 22/10, pour mémoire seulement (à refaire, ne pas citer)**

| Run | Juge | Distracteurs | Ex. | F1 | P | R | choix\|présent | FP | p95/req. | Statut |
|---|---|---|---|---|---|---|---|---|---|---|
| `pilote-20261022-144633` | StringMatch | hard | 200 | 0,005 | 0,003 | 0,044 | 0,044 | 0,998 | 0,04 ms | biaisé |
| `c1-multiatis-en-s13-20261022-153720` | Laya single | hard | 200 | 0,006 | 0,003 | 0,044 | 0,044 | 0,873 | 3,5 ms | biaisé |
| `c1-multiatis-en-s13-20261022-153752` | Laya single | easy | 200 | 0,670 | 1,000 | 0,504 | 0,506 | 0 | 3,1 ms | **invalide** |
| `c4-dev-multiatis-en-s13-20261022-160602` | Qwen3-32B | hard | 20 | 0,260 | 0,153 | 0,870 | 0,900 | 0,365 | 284 ms | biaisé, essai |
| `c4-dev-multiatis-en-s13-20261022-160829` | Qwen3-32B | hard | 200 | 0,266 | 0,157 | 0,878 | 0,886 | 0,298 | 10,1 ms | biaisé |
| `c4-dev-multiatis-en-s13-20261022-161119` | Qwen3-32B | easy | 200 | 0,972 | 0,999 | 0,948 | 0,956 | 0 | 8,2 ms | **invalide** |
| C1 `two_step`, `expressed_qtype=choice` (18:52) | Laya | hard | 200 | 0,010 | – | – | – | – | 2,4 ms | biaisé |
| C1 `rotations=4` (18:53) | Laya | hard | 200 | 0,005 | – | – | – | – | 3,2 ms | biaisé |
| C1 `two_step`, `choice`, checkpoint `english` (18:57) | Laya | hard | 200 | 0,030 | 0,017 | 0,133 | 0,134 | 0,453 | 2,7 ms | biaisé |
| C1 `two_step`, `noul` (18:58) | Laya | hard | 200 | 0,008 | 0,004 | 0,058 | 0,058 | 0,796 | 2,2 ms | biaisé |

- Les deux derniers runs ne sont pas dans `results/` en local ; leurs chiffres viennent des logs.
- Rappel des candidats oracle : 0,999 dans tous les runs (une paire gold non retrouvée sur 746).
- Mémoire GPU de pointe de Laya : 1,7 Gio. Pour vLLM, `gpu_peak_mem_gb` vaut 0 (le modèle tourne dans un autre processus) ; Qwen3-32B occupe 61 Gio.


#### 22 octobre 2026 (soir) — fine-tuning de Laya : méthode et premier essai C2

Code : commits locaux `4497f2e` (correction `fit`), `f141a5d` (fine-tuning), `417a773` (tableaux). Tous les runs de cette partie ont `candidates.fit: true`.

**Méthode de fine-tuning** (pour la section Méthode de l'article)
- **Pas de module d'entraînement public dans laya 0.3.22.** Le notebook officiel de fine-tuning importe `laya.train` (RLCD : gradient de politique sur des récompenses de règles de score propres), absent du paquet publié. Nous écrivons notre propre boucle, sur les primitives d'inférence de laya (`Agent._to_internal`, `Agent._encode_state`, `laya.common.collate_items`, `DecisionModel`).
- **Objectif : log score**, c'est-à-dire l'entropie croisée sur les options de chaque question. C'est une règle de score strictement propre, comme celles de RLCD, mais optimisée directement (supervision) et non par gradient de politique. Écart à signaler par rapport à l'entraînement d'origine de Laya.
- **Items construits par le code d'inférence.** Chaque requête passe par `LayaJudge._plan`, la fonction qui prépare les questions à l'évaluation : même état, mêmes gabarits, même ordre (mélangé) et même troncature des candidats. Seule la cible s'ajoute. Entraînement et évaluation ne peuvent donc pas différer de format.
- **Cibles.**
  - Mode `single` : la position de la gold parmi les options, sinon « non précisé » (slot non exprimé, ou gold absente de la liste).
  - Mode `two_step` : question « slot exprimé ? » (A/B ou `noul`) sur toutes les requêtes ; question de choix seulement quand la gold est présentée.
  - Rotations et tournoi sont des procédés d'inférence : on entraîne sur la question simple.
- **Requêtes.** Mêmes requêtes qu'à l'évaluation (`requests.slots=intent`, intention gold), construites sur le split train par `experiments.run.RequestBuilder`.
- **Déséquilibre.** On garde toutes les requêtes à slot exprimé et au plus 2 requêtes non exprimées par requête exprimée et par exemple (`unexpressed_ratio=2.0`), tirées au hasard avec une graine. Sans ce sous-échantillonnage, 94 % des questions seraient des « non précisé ».
- **Sélection du checkpoint.** Sur les 300 premiers exemples du dev, avec toutes leurs requêtes (même distribution qu'au test) : on garde l'époque de plus faible perte dev.
- **Températures.** Le checkpoint écrit des températures neutres (T = 1). La calibration se fera sur le dev, dans `laya_slots/decision/`.
- **Format de sortie.** Celui des checkpoints Laya (`rl_agent_config.json`, `model.safetensors`, `tokenizer/`, `encoder/`), chargé par `LayaJudge(model_path=…)` à la place du checkpoint publié.

**Hyperparamètres C2** (`configs/train/c2.yaml`)

| Paramètre | Valeur |
|---|---|
| Checkpoint de départ | `convaiinnovations/laya`, sous-dossier `multilingual` |
| Candidats | oracle k = 5, p_absent = 0, distracteurs `hard` (ceux de la grille d'évaluation) |
| Époques | 3 |
| Lot | 32 questions, sans accumulation |
| Optimiseur | AdamW, lr encodeur 2e-5, lr tête 1e-4, weight decay 0,01 |
| Planning | montée linéaire sur 5 % des pas, puis décroissance cosinus |
| Divers | clip de gradient 1,0 ; bf16 (autocast) sur poids fp32 ; gradient checkpointing (encodeur et tête) ; graine 13 |

**C3, augmentation par options aléatoires** (`configs/train/c3.yaml`) : mêmes réglages que C2, mais pour chaque exemple d'entraînement on tire une combinaison parmi k ∈ {2, 5, 10, 20}, p_absent ∈ {0 ; 0,2 ; 0,5} et distracteurs ∈ {`easy`, `hard_no_overlap`}. L'ordre des options est déjà mélangé par le juge. Ablations prévues : une composante figée à la fois (par exemple `train.augment.k=[5]`). Attention : `k = 20` est ramené à 15 candidats par `max_options=15` du juge.

**Effectifs** (MultiATIS++ EN, préparation vérifiée en local)
- 400 exemples de train : 1 479 requêtes exprimées, 18 401 non exprimées, dont 2 960 gardées ; 4 439 questions de 6 options (5 candidats + « non précisé »).
- 200 exemples de train (essai sur la machine) : 748 exprimées, 8 652 non exprimées, dont 1 498 gardées ; 2 246 questions. Dev (30 exemples, toutes les requêtes) : 1 410 questions, dont 1 296 sans slot exprimé.

**Essai 1 : chaîne d'entraînement** (200 exemples de train, 30 de dev, 1 époque, sur la machine)
- Encodage en 1 s ; époque en 11 s sur une H200.
- Perte dev 2,958 → 0,203 ; exactitude dev 0,167 → 0,943.
- Exactitude peu informative : 92 % des questions dev n'ont pas de slot exprimé, et répondre « non précisé » partout donnerait déjà 0,919. Le suivi sépare depuis les questions exprimées et non exprimées.

**Essai 2 : premier résultat C2** — **valide, préliminaire** (1 000 exemples de train, 100 de dev, 1 époque ; test : 200 premiers exemples, oracle k = 5, `hard`)

| Juge | F1 | P | R | choix\|présent | p95/req. |
|---|---|---|---|---|---|
| **Laya C2 (essai)** | **0,789** | 0,737 | 0,847 | **0,855** | 2,3 ms |
| Laya C1 zero-shot, pour mémoire (runs biaisés, § ci-dessus) | ≈ 0,01 | ≈ 0,01 | ≈ 0,07 | ≈ 0,07 | ≈ 3 ms |
| Qwen3-32B zero-shot, pour mémoire (run biaisé) | 0,266 | 0,157 | 0,878 | 0,886 | 10,1 ms |

- La ligne C1 du tableau de la machine moyennait 6 runs (avant et après correction, plusieurs modes du juge) : F1 0,012 ± 0,009. Depuis, `make tableaux` sépare aussi `fit`, le mode, le checkpoint, les rotations et le type de question.
- Runs de ces essais : `c2-multiatis-en-s13-20261022-194445` (C2) et `c1-multiatis-en-s13-20261022-194537` (C1 corrigé : F1 0,014, choix|présent 0,099, FP 0,820 sur les mêmes 200 exemples).

#### 22 octobre 2026 (nuit) — entraînement C2 complet et première comparaison valide

**Entraînement C2 complet** (`make entrainer-c2`, MultiATIS++ EN, graine 13, checkpoint `checkpoints/c2/multiatis-en-s13`)
- 4 478 exemples de train → 14 336 requêtes exprimées, 256 251 non exprimées dont 28 671 gardées → 43 007 questions. Dev : 300 exemples, 20 148 questions, dont 1 070 à slot exprimé.
- Encodage 10 s ; environ 80 s par époque sur une H200 ; 3 époques en 4 min.

| Époque | Perte train | Perte dev (toutes) | Exact. dev, exprimés | Exact. dev, non exprimés | Perte de sélection (`balanced`) |
|---|---|---|---|---|---|
| 0 (zero-shot) | – | 3,286 | 0,099 | 0,152 | 3,407 |
| 1 | 0,1945 | 0,014 | 0,965 | 0,997 | 0,061 |
| **2 (gardée)** | 0,0321 | 0,018 | **0,988** | 0,996 | **0,038** |
| 3 | 0,0061 | 0,015 | 0,988 | 0,998 | 0,046 |

- **Changement de critère, décidé sur le dev uniquement.** Le premier entraînement complet choisissait le checkpoint sur la perte dev globale, dominée à 95 % par les questions non exprimées : il gardait l'époque 1. Le critère est devenu la moyenne des pertes exprimé / non exprimé (`train.select_on: balanced`), qui garde l'époque 2. La décision a été prise sur le log dev, avant de voir le test.
- **Reproductibilité.** Les deux entraînements complets (19:54 et 20:54) donnent les mêmes pertes, à la troisième décimale, à chaque époque.

**Évaluation sur tout le test** — **valide, ablation `hard`** depuis la décision du 23/10 (la grille principale passe à `hard_no_overlap`, à relancer) (MultiATIS++ EN test, 893 exemples, 51 834 requêtes, 2 837 paires gold ; oracle k = 5, p_absent = 0, `hard`, `fit: true` ; intention gold ; graine 13)

| Condition | Run | F1 | P | R | Exact. de trame | choix\|présent | FP | p50 / p95 par req. | Débit |
|---|---|---|---|---|---|---|---|---|---|
| C1 Laya zero-shot | `c1-…-200049` | 0,012 | 0,006 | 0,094 | 0,000 | 0,094 | 0,830 | 1,1 / 1,7 ms | 819 req/s |
| **C2 Laya fine-tuné** (époque 2) | `c2-…-205848` | **0,919** | **0,888** | **0,952** | **0,703** | **0,956** | **0,006** | 1,1 / 1,8 ms | 823 req/s |
| C4-dev Qwen3-32B zero-shot | `c4-dev-…-200217` | 0,276 | 0,164 | 0,887 | 0,035 | 0,892 | 0,257 | 9,0 / 10,0 ms | 102 req/s |

Décomposition des 2 837 paires gold et des 49 057 requêtes non exprimées :

| Condition | Correctes | Erreurs de choix | dont abstentions à tort | Hors liste | Faux positifs | Abstentions correctes |
|---|---|---|---|---|---|---|
| C1 | 268 | 2 568 | 784 | 1 | 40 726 | 8 331 |
| C2 | 2 712 | 124 | 51 | 1 | 314 | 48 743 |
| C4-dev | 2 529 | 307 | 29 | 1 | 12 617 | 36 440 |

- C2 avec le checkpoint de l'époque 1 (critère « all ») : F1 0,909, P 0,882, R 0,937, choix|présent 0,941, FP 0,007 (`c2-…-195920`).
- **Reproductibilité.** C1 et C4-dev relancés à l'identique (`c1-…-210016`, `c4-dev-…-210226`) donnent exactement les mêmes métriques.
- **Traçabilité à corriger.** Tous ces runs enregistrent `git_commit = 143f163`, `git_dirty = True` : le dossier `.git` de la machine n'est pas celui du poste à jour. Le code exécuté est bien celui des commits `4497f2e` à `67122a7`, mais `metrics.json` ne le prouve pas (incident 10).

#### 22 octobre 2026 (22 h) — premier C3, et une grille lancée sur du code non synchronisé

**Ce qui devait tourner** : la grille principale en `hard_no_overlap` (décision du 23/10). **Ce qui a tourné** : du code antérieur, avec `configs/candidats/oracle.yaml` encore en `hard`. La copie vers la machine n'avait pas eu lieu (tous les runs enregistrent toujours `git_commit = 143f163`). Ces runs sont donc tous des runs **`hard`, c'est-à-dire l'ablation**. Ils restent valides comme tels (`fit: true`, tout le test).

**Entraînements** (MultiATIS++ EN, graine 13, mêmes effectifs que le C2 précédent : 43 007 questions)

| Époque | C2 : exact. dev exprimés / non exprimés | C2 : perte de sélection | C3 : exact. dev exprimés / non exprimés | C3 : perte de sélection |
|---|---|---|---|---|
| 1 | 0,965 / 0,997 | 0,061 | **0,738** / 0,992 | **0,382** (gardée) |
| 2 | **0,987** / 0,996 | **0,038** (gardée) | 0,779 / 0,994 | 0,420 |
| 3 | 0,987 / 0,998 | 0,046 | 0,771 / 0,997 | 0,504 |

- C2 reproduit à nouveau les pertes du 22/10 à 19:54 et 20:54. Seule la perte de sélection de l'époque 2 bouge à la 4e décimale (0,0378 contre 0,0383).
- C3 : l'exactitude sur les slots exprimés monte de l'époque 1 à 2, mais la perte monte aussi. Le modèle devient trop sûr de lui sur ses erreurs. Le critère, fondé sur la perte, garde l'époque 1.

**Évaluation sur tout le test, ablation `hard`** (893 exemples, oracle k = 5, `fit: true`)

| Condition | Run | F1 | P | R | Exact. de trame | choix\|présent | dont abstentions à tort | FP |
|---|---|---|---|---|---|---|---|---|
| C1 | `c1-…-221210` | 0,012 | 0,006 | 0,094 | 0,000 | 0,094 | 784 | 0,830 |
| C2 | `c2-…-221340` | **0,919** | 0,887 | 0,952 | **0,700** | **0,957** | 49 | **0,006** |
| C3 | `c3-…-221508` | 0,708 | 0,731 | 0,687 | 0,241 | 0,690 | **743** | 0,013 |
| C4-dev | `c4-dev-…-221637` | 0,276 | 0,164 | 0,887 | 0,035 | 0,892 | 29 | 0,257 |

- **Lecture.** Sur ce réglage, qui est exactement celui de l'entraînement de C2, C3 fait nettement moins bien que C2. La perte vient surtout d'**abstentions à tort** : 743 paires gold présentes dans la liste mais refusées, contre 49 pour C2.
- **Ce n'est pas encore un test de H3, pour deux raisons.**
  1. **C3 est évalué hors de sa distribution d'entraînement.** Son augmentation tire les distracteurs parmi `easy` et `hard_no_overlap`, donc il n'a jamais vu les variantes de frontière de `hard`.
  2. **L'entraînement avec p_absent ∈ {0,2 ; 0,5} lui apprend à s'abstenir quand la gold manque**, et il étend apparemment ce comportement à des cas où elle est présente.
  H3 se teste hors de la distribution de C2 : balayages k × p_absent, candidats réels, et grille `hard_no_overlap`.
- **Pistes si l'écart persiste en `hard_no_overlap`** :
  - abaisser la part de p_absent > 0 dans l'augmentation ;
  - ajouter `hard` à la liste des distracteurs d'entraînement ;
  - choisir le checkpoint sur l'exactitude dev des exprimés plutôt que sur la perte (C3 est surconfiant) ;
  - calibrer les températures.
- **Runs écartés des tableaux** (fichier `OBSOLETE`) :
  - les répliques à graine identique de C1, C4-dev et C2, aux métriques identiques (reproductibilité vérifiée une fois de plus) ;
  - le run `c3-…-215028`, fait avec un checkpoint C3 antérieur dont le log a été écrasé (F1 0,769).
- **Correction pour la traçabilité.**
  - `metrics.json` enregistre désormais `code_sha256`, une empreinte des fichiers `.py` et `.yaml` de `laya_slots/`, `experiments/` et `configs/`. Elle ne dépend pas de `.git`.
  - `make version` l'affiche, avec le commit et le réglage des distracteurs, pour comparer le poste et la machine avant chaque série de runs.

#### 23 octobre 2026 (matin) — grille principale `hard_no_overlap`, MultiATIS++ EN — **valide**

Code vérifié : tous les runs enregistrent `code_sha256 = 3f8c77c319d3…`, identique au poste (commit `99a0c5e`). Entraînements C2 et C3 refaits sur les candidats de la grille ; les checkpoints `hard` du 22/10 sont gardés pour l'ablation (`checkpoints/c2|c3/multiatis-en-s13-hard`).

**Entraînements** (graine 13, 43 007 questions ; dev 300 exemples, 20 148 questions dont 1 070 exprimées)

| Époque | C2 : exact. dev exprimés / non exprimés | C2 : perte de sélection | C3 : exact. dev exprimés / non exprimés | C3 : perte de sélection |
|---|---|---|---|---|
| 0 (zero-shot) | 0,223 / 0,152 | 3,149 | 0,223 / 0,152 | 3,149 |
| 1 | 0,981 / 0,992 | 0,046 | 0,952 / 0,992 | 0,099 |
| 2 | 0,985 / 0,998 | **0,040** (gardée) | 0,983 / 0,993 | **0,051** (gardée) |
| 3 | 0,988 / 0,998 | 0,049 | 0,980 / 0,996 | 0,069 |

- Environ 80 s par époque sur une H200 pour chacun ; C3 garde l'époque 2, comme C2.
- Laya zero-shot est moins mauvais en `hard_no_overlap` qu'en `hard` (exact. dev exprimés 0,223 contre 0,099) : une partie de ses erreurs étaient des variantes de frontière.

**Évaluation sur tout le test** (893 exemples, 51 834 requêtes, 2 837 paires gold ; oracle k = 5, p_absent = 0, `hard_no_overlap`, `fit: true` ; intention gold ; graine 13)

| Condition | Run | F1 | P | R | Exact. de trame | choix\|présent | Abstentions à tort | Faux positifs (taux) | p50 par req. |
|---|---|---|---|---|---|---|---|---|---|
| C1 Laya zero-shot | `c1-…-20261023-084135` | 0,025 | 0,013 | 0,202 | 0,000 | 0,202 | 763 | 40 726 (0,830) | 1,1 ms |
| **C2 Laya fine-tuné** | `c2-…-20261023-084304` | **0,938** | **0,925** | 0,952 | **0,782** | **0,957** | 55 | **196 (0,004)** | 1,1 ms |
| C3 Laya fine-tuné + augmentation | `c3-…-20261023-084432` | 0,892 | 0,846 | 0,944 | 0,628 | 0,949 | 75 | 462 (0,009) | 1,1 ms |
| C4-dev Qwen3-32B zero-shot | `c4-dev-…-20261023-084601` | 0,295 | 0,175 | 0,947 | 0,035 | 0,952 | 29 | 12 617 (0,257) | 9,1 ms |

- **C2 et C3 dépassent nettement le LLM-juge 32 B zero-shot** (F1 0,938 et 0,892 contre 0,295), avec une latence par requête environ 8 fois plus faible (mesure par lot). Le choix de la valeur est au même niveau (0,95–0,96) ; l'écart vient presque entièrement des faux positifs du LLM.
- **C3 fait un peu moins bien que C2 sur ce réglage**, qui est exactement celui de l'entraînement de C2 (k = 5, p_absent = 0) : plus de faux positifs (462 contre 196) et un peu plus d'abstentions à tort. C'est attendu ; l'intérêt de C3 se mesure hors de ce réglage (balayages k × p_absent, candidats réels).
- **Effet de `hard_no_overlap` par rapport à l'ablation `hard`.**
  - Sur le choix (choix|présent) : C2 0,957 contre 0,957 (inchangé) ; C4-dev 0,952 contre 0,892 ; C1 0,202 contre 0,094.
  - Le LLM et Laya zero-shot étaient donc pénalisés par les variantes de frontière, mais pas Laya fine-tuné sur le même réglage.
  - Les faux positifs sont identiques entre les deux réglages pour C1 et C4-dev (40 726 et 12 617). C'est normal : les listes des slots non exprimés n'ont pas de gold, donc rien à exclure.
- **C3 entraîné sur ses propres distracteurs** : en `hard_no_overlap`, il passe de F1 0,708 (ablation `hard`) à 0,892 ; ses abstentions à tort tombent de 743 à 75. La chute du 22/10 venait bien des distracteurs `hard` jamais vus à l'entraînement.

**Suite** : balayages k × p_absent, faits le même jour (sous-section suivante).

#### 23 octobre 2026 (matinée) — balayages k × p_absent, grille `hard_no_overlap` — **valide** (QR2, QR5, H3)

36 runs (C2, C3, C4-dev × k ∈ {2, 5, 10, 20} × p_absent ∈ {0 ; 0,2 ; 0,5}), MultiATIS++ EN, tout le test, graine 13, `max_options=20` (aucune troncature), `code_sha256 = 3f8c77c319d3…`. Commande : `make balayage COND=c2|c3|c4-dev`. Les checkpoints C2 et C3 sont ceux de la grille (k = 5, p_absent = 0 pour C2 ; augmentation pour C3).

**Nouvelle métrique : abstention quand la gold est absente.** Quand le slot est exprimé mais que sa valeur a été retirée de la liste (p_absent), la seule réponse sans faux positif est « non précisé ». `decomposition.absent_abstention_rate` donne la part de ces cas où le juge s'abstient. Ajoutée à `laya_slots/evaluation/metrics.py` le 23/10, et recalculée depuis `predictions.jsonl` pour les 63 runs existants (`decomposition_backfilled` dans `metrics.json`). Avec p_absent = 0, elle porte sur une seule paire et n'a pas de sens (« – »).

| k | p_absent | F1 C2 | F1 C3 | F1 C4-dev | P C2 | P C3 | P C4-dev | Abst. si gold absente C2 | C3 | C4-dev | FP C2 | FP C3 | FP C4-dev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | **0,947** | 0,905 | 0,451 | 0,948 | 0,863 | 0,295 | – | – | – | 0,003 | 0,009 | 0,132 |
| 2 | 0,2 | **0,824** | 0,791 | 0,362 | 0,913 | 0,832 | 0,237 | 0,883 | **0,976** | 0,242 | 0,003 | 0,009 | 0,132 |
| 2 | 0,5 | **0,593** | 0,573 | 0,226 | 0,825 | 0,744 | 0,149 | 0,900 | **0,977** | 0,244 | 0,003 | 0,009 | 0,132 |
| 5 | 0 | **0,938** | 0,892 | 0,295 | 0,925 | 0,846 | 0,175 | – | – | – | 0,004 | 0,009 | 0,257 |
| 5 | 0,2 | **0,811** | 0,778 | 0,234 | 0,876 | 0,810 | 0,139 | 0,837 | **0,974** | 0,062 | 0,004 | 0,009 | 0,257 |
| 5 | 0,5 | **0,574** | 0,560 | 0,145 | 0,742 | 0,712 | 0,086 | 0,814 | **0,962** | 0,074 | 0,004 | 0,009 | 0,257 |
| 10 | 0 | **0,904** | 0,887 | 0,265 | 0,860 | 0,839 | 0,155 | – | – | – | 0,008 | 0,010 | 0,293 |
| 10 | 0,2 | 0,765 | **0,772** | 0,209 | 0,775 | 0,801 | 0,122 | 0,665 | **0,959** | 0,034 | 0,008 | 0,010 | 0,293 |
| 10 | 0,5 | 0,522 | **0,555** | 0,129 | 0,592 | 0,701 | 0,075 | 0,649 | **0,956** | 0,040 | 0,008 | 0,010 | 0,293 |
| 20 | 0 | 0,813 | **0,903** | 0,263 | 0,721 | 0,884 | 0,154 | – | – | – | 0,019 | 0,006 | 0,291 |
| 20 | 0,2 | 0,679 | **0,787** | 0,208 | 0,630 | 0,854 | 0,122 | 0,615 | **0,969** | 0,033 | 0,019 | 0,006 | 0,291 |
| 20 | 0,5 | 0,454 | **0,571** | 0,129 | 0,452 | 0,776 | 0,075 | 0,587 | **0,969** | 0,038 | 0,019 | 0,006 | 0,291 |

Les candidats retirés font chuter le rappel de la même façon pour tous les juges (rappel des candidats 1,00 → 0,79 → 0,49 pour p_absent 0 → 0,2 → 0,5) : le F1 plafonne, et ce sont la précision et l'abstention qui départagent.

**Constats**
- **H3 : soutenue sur MultiATIS++ EN (une graine).** Hors de son réglage d'entraînement, C2 se dégrade ; C3 reste stable.
  - **Valeur absente.** C3 s'abstient correctement dans 96–98 % des cas, quel que soit k. C2 passe de 0,90 (k = 2) à 0,59 (k = 20) : plus la liste est longue, plus il choisit un distracteur quand la bonne valeur manque. Qwen3-32B ne s'abstient presque jamais (3–24 %).
  - **Nombre d'options.** À p_absent = 0, C2 passe de F1 0,947 (k = 2) à 0,813 (k = 20), avec 1,9 % de faux positifs à k = 20 ; C3 reste entre 0,887 et 0,905.
  - **Croisement.** C2 garde l'avantage pour k ≤ 5 (son réglage d'entraînement et en dessous) ; C3 l'emporte à k = 10 avec valeur absente et pour tout k = 20 (F1 0,903 contre 0,813 à p_absent = 0, 0,571 contre 0,454 à p_absent = 0,5).
- **Prix de l'augmentation sur le réglage de C2** : 0,03 à 0,05 de F1 à k ≤ 5, surtout des faux positifs sur les slots non exprimés (0,9 % contre 0,4 %). Piste : ajouter k = 5, p_absent = 0 plus souvent dans l'augmentation, ou régler un seuil d'abstention sur le dev.
- **QR2 : la décomposition de l'erreur.** Avec la valeur présente, le choix reste au-dessus de 0,92 pour les trois juges jusqu'à k = 20 (non montré dans le tableau ; `choice_accuracy_given_present`). L'erreur vient donc surtout du rappel des candidats et de l'abstention, pas du choix.
- **Qwen3-32B** : ses faux positifs sur les slots non exprimés augmentent avec k (13 % à k = 2, 29 % à k ≥ 10) ; son F1 reste sous 0,46 partout.
- **Latence** (p50 par requête, par lot) : Laya 1,0 ms (k = 2) à 1,5 ms (k = 20) ; Qwen3-32B 7,3 ms à 18,5 ms. Le facteur va de 7 à 12 selon k.
- **Répliques** : les runs k = 5, p_absent = 0 du balayage reproduisent ceux de la grille ; ils sont marqués `OBSOLETE`.

#### 23 octobre 2026 — intervalles de confiance et tests appariés (méthode statistique)

**Méthode** (pour la section Cadre expérimental de l'article), dans `laya_slots/evaluation/bootstrap.py` et `experiments/intervalles.py` :
- **Grappes.** On rééchantillonne des dialogues pour Multi3WOZ et des énoncés pour les jeux à un tour, jamais des tours ni des requêtes isolées. Chaque grappe est résumée par des compteurs additifs (vrais et faux positifs, choix corrects, abstentions…). Chaque métrique est un rapport de sommes, avec les mêmes définitions que `metrics.py`, vérifiées à l'identique sur les runs réels.
- **Intervalles.** Bootstrap percentile à 95 %, 1 000 tirages, graine fixée. Ils sont écrits dans `ci.json` à côté de `metrics.json` (`make intervalles`) et lus par `make tableaux` (colonnes `ci.<métrique>.lo|hi`).
- **Comparaisons appariées.** Sur les grappes communes, on calcule la différence A − B et son intervalle par bootstrap apparié (les mêmes grappes tirées pour A et B). Le test de permutation échange A et B grappe par grappe (10 000 permutations, p bilatérale). La **correction de Holm** porte sur l'ensemble des comparaisons × métriques d'un appel (`make comparer PAIRES="a b c d"`).
- **Seuil de résolution.** Avec 10 000 permutations, la plus petite p-valeur possible est 1/10 001 ; après Holm sur 36 tests, elle vaut 0,0036. Tous les « 0,0036 » ci-dessous signifient donc « p < 10⁻⁴ avant correction ».
- **Coût.** Moins de 2 s pour un run de 893 grappes, et environ 1 s par comparaison.

**Intervalles sur la grille principale** (k = 5, p_absent = 0, graine 13), F1 [IC 95 %] :
- C2 : 0,938 [0,930 ; 0,948] ;
- C4-dev : 0,295 [0,289 ; 0,301] ;
- choix|présent de C2 : 0,957 [0,947 ; 0,966] ;
- faux positifs de C4-dev : 0,257 [0,249 ; 0,266].

**H3 : C3 − C2 sur le balayage** (graine 13 ; 12 points × 3 métriques = 36 tests, Holm ; fichier `results/comparaisons_c3_c2_balayage.csv`)

| k | p_absent | Δ F1 [IC 95 %] | p Holm | Δ abst. si gold absente [IC] | p Holm | Δ FP [IC] | p Holm |
|---|---|---|---|---|---|---|---|
| 2 | 0 | −0,042 [−0,049 ; −0,035] | 0,0036 | – | – | +0,006 [+0,005 ; +0,007] | 0,0036 |
| 2 | 0,2 | −0,032 [−0,040 ; −0,026] | 0,0036 | +0,093 [+0,070 ; +0,120] | 0,0036 | +0,006 [+0,005 ; +0,007] | 0,0036 |
| 2 | 0,5 | −0,020 [−0,027 ; −0,014] | 0,0036 | +0,076 [+0,061 ; +0,092] | 0,0036 | +0,006 [+0,005 ; +0,007] | 0,0036 |
| 5 | 0 | −0,046 [−0,054 ; −0,039] | 0,0036 | – | – | +0,005 [+0,005 ; +0,006] | 0,0036 |
| 5 | 0,2 | −0,033 [−0,041 ; −0,025] | 0,0036 | +0,137 [+0,108 ; +0,168] | 0,0036 | +0,005 [+0,005 ; +0,006] | 0,0036 |
| 5 | 0,5 | −0,014 [−0,022 ; −0,006] | 0,0081 | +0,148 [+0,130 ; +0,168] | 0,0036 | +0,005 [+0,005 ; +0,006] | 0,0036 |
| 10 | 0 | −0,017 [−0,025 ; −0,009] | 0,0036 | – | – | +0,002 [+0,001 ; +0,003] | 0,0104 |
| 10 | 0,2 | +0,006 [−0,003 ; +0,015] | 0,79 | +0,294 [+0,256 ; +0,330] | 0,0036 | +0,002 [+0,001 ; +0,003] | 0,0104 |
| 10 | 0,5 | **+0,033** [+0,024 ; +0,042] | 0,0036 | +0,307 [+0,283 ; +0,330] | 0,0036 | +0,002 [+0,001 ; +0,003] | 0,0104 |
| 20 | 0 | **+0,090** [+0,078 ; +0,101] | 0,0036 | – | – | −0,013 [−0,015 ; −0,012] | 0,0036 |
| 20 | 0,2 | **+0,108** [+0,098 ; +0,119] | 0,0036 | +0,354 [+0,312 ; +0,394] | 0,0036 | −0,013 [−0,015 ; −0,012] | 0,0036 |
| 20 | 0,5 | **+0,117** [+0,107 ; +0,126] | 0,0036 | +0,381 [+0,352 ; +0,409] | 0,0036 | −0,013 [−0,015 ; −0,012] | 0,0036 |

Avec p_absent = 0, l'abstention si gold absente ne porte que sur une paire : sans objet (« – »).

- **Sur l'abstention quand la gold est absente, C3 l'emporte partout, de façon significative**, et l'écart grandit avec k (+0,08 à k = 2, +0,38 à k = 20).
- **Sur le F1, la différence change de signe avec la distance au réglage d'entraînement de C2.**
  - C2 l'emporte à k ≤ 5 et à k = 10 sans valeur absente (−0,014 à −0,046).
  - Pas de différence à k = 10, p_absent = 0,2.
  - C3 l'emporte à k = 10, p_absent = 0,5 et pour tout k = 20 (+0,03 à +0,12).
- **Limite : une graine par condition.** Les intervalles mesurent la variabilité due à l'échantillon de test, pas celle de l'entraînement. Les graines 14 à 17 l'ajoutent (sous-section suivante).

#### 23 octobre 2026 (après-midi) — graines supplémentaires : C2 × 3, C3 × 5 — **valide**

72 runs (`make graines`) : C2 graines 14 et 15, C3 graines 14 à 17 ; pour chaque graine, fine-tuning puis balayage complet k × p_absent (`hard_no_overlap`, tout le test). Code : `code_sha256 = 60c814f1cd09…`, identique à l'empreinte recalculée du commit `47a484f`. La graine change à la fois l'entraînement (ordre, sous-échantillonnage, initialisation de l'aléa) et le tirage des candidats oracle du test. Les logs d'entraînement des graines 14 à 17 ne sont pas rapatriés (`train_metrics.json` restés dans les checkpoints, sur la machine).

**Moyenne ± écart-type sur les graines** (C2 : graines 13 à 15 ; C3 : 13 à 17)

| k | p_absent | F1 C2 (3 graines) | F1 C3 (5 graines) | Abst. si gold absente C2 | C3 | FP C2 | FP C3 |
|---|---|---|---|---|---|---|---|
| 2 | 0,0 | 0,950 ± 0,003 | 0,904 ± 0,024 | – | – | 0,003 ± 0,000 | 0,009 ± 0,003 |
| 2 | 0,2 | 0,830 ± 0,007 | 0,793 ± 0,025 | 0,875 ± 0,014 | 0,963 ± 0,013 | 0,003 ± 0,000 | 0,009 ± 0,003 |
| 2 | 0,5 | 0,592 ± 0,014 | 0,575 ± 0,021 | 0,884 ± 0,015 | 0,971 ± 0,008 | 0,003 ± 0,000 | 0,009 ± 0,003 |
| 5 | 0,0 | 0,939 ± 0,001 | 0,890 ± 0,020 | – | – | 0,004 ± 0,000 | 0,009 ± 0,003 |
| 5 | 0,2 | 0,813 ± 0,007 | 0,779 ± 0,021 | 0,802 ± 0,031 | 0,972 ± 0,012 | 0,004 ± 0,000 | 0,009 ± 0,003 |
| 5 | 0,5 | 0,569 ± 0,016 | 0,562 ± 0,017 | 0,792 ± 0,018 | 0,965 ± 0,011 | 0,004 ± 0,000 | 0,009 ± 0,003 |
| 10 | 0,0 | 0,914 ± 0,010 | 0,889 ± 0,018 | – | – | 0,007 ± 0,002 | 0,009 ± 0,003 |
| 10 | 0,2 | 0,780 ± 0,020 | 0,778 ± 0,020 | 0,679 ± 0,041 | 0,969 ± 0,009 | 0,007 ± 0,002 | 0,009 ± 0,003 |
| 10 | 0,5 | 0,530 ± 0,026 | 0,561 ± 0,018 | 0,662 ± 0,034 | 0,964 ± 0,008 | 0,007 ± 0,002 | 0,009 ± 0,003 |
| 20 | 0,0 | 0,823 ± 0,039 | 0,906 ± 0,013 | – | – | 0,018 ± 0,007 | 0,005 ± 0,002 |
| 20 | 0,2 | 0,690 ± 0,044 | 0,795 ± 0,013 | 0,557 ± 0,062 | 0,982 ± 0,008 | 0,018 ± 0,007 | 0,005 ± 0,002 |
| 20 | 0,5 | 0,458 ± 0,043 | 0,577 ± 0,012 | 0,550 ± 0,047 | 0,979 ± 0,007 | 0,018 ± 0,007 | 0,005 ± 0,002 |

- **Variabilité entre graines.**
  - Au point de grille (k = 5, p_absent = 0), F1 de C2 : 0,9385 / 0,9394 / 0,9400. F1 de C3 : 0,892 / 0,911 / 0,890 / **0,858** / 0,899.
  - C3 est nettement plus sensible à la graine que C2 sur ce réglage (écart-type 0,020 contre 0,001).
  - En revanche, C3 est **plus stable que C2 à k = 20** (0,013 contre 0,039).
- **Tests appariés C3 − C2 par graine** (graines 14 et 15 ; 2 × 12 points × 2 métriques = 48 tests, Holm ; `results/comparaisons_c3_c2_graines14-15.csv`). Le motif de la graine 13 se reproduit :
  - **Abstention quand la gold est absente** : C3 l'emporte sur les 18 points où elle est définie, pour chacune des deux graines (+0,09 à +0,49, tous significatifs).
  - **F1** :
    - C2 l'emporte à k ≤ 5 avec p_absent ≤ 0,2, et à k = 10 sans valeur absente (tous significatifs) ;
    - à k ≤ 5 avec p_absent = 0,5, l'écart devient faible : 3 des 4 cellules ne sont pas significatives (+0,004, +0,008, −0,002), la quatrième vaut −0,016 ;
    - aucune différence à k = 10, p_absent = 0,2 ;
    - C3 l'emporte à k = 10, p_absent = 0,5 (+0,031 et +0,037) et sur les trois points de k = 20 (+0,045 à +0,148).
- **Résumés d'entraînement des graines** (`results/checkpoints-resumes/`) : époque gardée par le critère `balanced`.
  - C2 : époque 2 pour les graines 13 et 14, époque 3 pour la graine 15.
  - C3 : époque 2 pour les graines 13, 14, 15 et 17, **époque 1 pour la graine 16**.
  - C'est ce qui explique la graine 16, plus faible (F1 0,858). Son checkpoint de l'époque 1 laisse passer plus de faux positifs au test (663 contre 304 à 462 pour les autres graines), et il est aussi moins bon sur le dev pour les slots non exprimés (0,991 contre 0,997 à l'époque 2). Le critère l'a pourtant choisi, parce que la perte des slots exprimés remonte à l'époque 2 (0,113 → 0,158) : le modèle devient trop sûr de lui sur ses erreurs.
  - **Décision ouverte** : choisir le checkpoint sur une métrique de décision (F1 ou exactitude équilibrée sur le dev) plutôt que sur une perte, que la surconfiance fausse ; ou calibrer avant de choisir.
- **Conclusion pour H3 (MultiATIS++ EN, oracle, 3 à 5 graines).** L'augmentation par options aléatoires rend le juge robuste au nombre d'options et à l'absence de la bonne valeur, au prix de 0,02 à 0,05 de F1 sur le réglage d'entraînement de C2. Il reste à vérifier :
  - avec les candidats réels ;
  - sur les autres jeux ;
  - en transfert vers le français.
- **Point de grille, quatre conditions sur les mêmes tirages de candidats** (graines 13, 14 et 15 ; C1 et C4-dev relancés le 26/10 avec le code `60c814f1cd09…`) :

| Condition | F1 | Exact. de trame | choix\|présent | FP |
|---|---|---|---|---|
| C1 Laya zero-shot | 0,025 ± 0,001 | 0,000 ± 0,001 | 0,203 ± 0,005 | 0,831 ± 0,001 |
| **C2 Laya fine-tuné** | **0,939 ± 0,001** | **0,782 ± 0,004** | **0,954 ± 0,004** | **0,004 ± 0,000** |
| C3 Laya fine-tuné + augmentation | 0,898 ± 0,011 | 0,632 ± 0,042 | 0,942 ± 0,006 | 0,008 ± 0,002 |
| C4-dev Qwen3-32B zero-shot | 0,296 ± 0,001 | 0,035 ± 0,001 | 0,953 ± 0,001 | 0,257 ± 0,001 |

  Pour C3, la moyenne porte ici sur les graines 13 à 15 ; sur 5 graines, F1 0,890 ± 0,020. La variation de C1 et de C4-dev entre graines ne vient que du tirage des candidats : elle est négligeable.

#### 23 octobre 2026 (soir) — préparation des candidats réels

**Configuration** (`configs/candidats/reels.yaml`, `configs/conditions/*_reels.yaml`, `make <condition> CAND=reels`)
- **Générateurs**, dans cet ordre :
  - `LexiconGenerator` : valeurs vues au train, retrouvées dans l'énoncé ;
  - `GlinerGenerator` : GLiNER multilingue zero-shot, seuil 0,3, label = description du slot, sur GPU ;
  - `OntologyGenerator` : valeurs fermées de l'ontologie, similarité ≥ 0,85 ;
  - `HistoryGenerator` : valeurs de l'historique, Multi3WOZ seulement.
- **Fusion.** Union dédoublonnée par valeur, le premier générateur l'emportant ; au plus 20 candidats, ordre mélangé (graine du run). Le juge reçoit `max_options = 20`, donc aucune troncature de sa part.
- **Mêmes juges et mêmes checkpoints.** Ce sont ceux de la grille oracle : C2 et C3 ne sont **pas** réentraînés sur des candidats réels. C'est le test hors distribution de QR5 et H3.
- `settings.generators` est ajouté aux groupes de `make tableaux`, pour séparer les runs oracle des runs réels.

**Mesure préliminaire en local** (CPU, sans GLiNER : lexique + ontologie seulement ; 200 premiers exemples du test MultiATIS++ EN ; juge par correspondance de chaînes)
- Rappel des candidats : **0,95** (39 paires gold manquées sur 746, surtout `airport_name` et `flight_number`).
- Les listes ont une forme très différente de l'oracle :
  - 1,8 candidat en moyenne quand le slot est exprimé ;
  - 92 % de listes **vides** quand il ne l'est pas, ce que le juge traite comme « non précisé » sans même être interrogé.
  - Les faux positifs ne peuvent donc venir que des 8 % de listes non vides.
- GLiNER ajoutera des candidats (rappel en hausse, listes plus longues) : à mesurer sur la machine.

#### 26 octobre 2026 — candidats réels, MultiATIS++ EN — **valide** (QR5, H3 hors oracle)

10 runs (`make <condition> CAND=reels`) : C1 et C4-dev (graine 13), C2 (graines 13 à 15), C3 (graines 13 à 17). Tout le test, 893 exemples, 51 834 requêtes. Code `908a4eff2564…`, commit `f2cc6f1`, vérifié des deux côtés par `make version`. Checkpoints C2 et C3 de la grille oracle, sans réentraînement.

**Les listes réelles** (lexique + GLiNER + ontologie, ≤ 20 candidats)
- **Rappel des candidats : 0,959** (118 paires gold sur 2 837 manquent à la liste).
- **Listes très courtes** : 0,27 candidat par requête en moyenne. Quand une liste n'est pas vide, elle compte 2,0 candidats si le slot est exprimé et 1,6 s'il ne l'est pas.
- **89 % des requêtes à slot non exprimé ont une liste vide** (43 494 sur 49 057) : « non précisé » sans interroger le juge. Les faux positifs ne peuvent venir que des 5 563 listes non vides, qui contiennent surtout des valeurs **d'autres slots** de l'énoncé (« boston » proposé pour l'escale alors que c'est le départ).
- Variantes de frontière : sur les 2 663 requêtes dont la gold est présente, 304 listes en contiennent une (11 %).

**Résultats** (moyenne ± écart-type sur les graines quand il y en a plusieurs)

| Condition | Graines | F1 | P | R | Exact. de trame | choix\|présent | FP (taux) | Abst. si gold absente |
|---|---|---|---|---|---|---|---|---|
| C1 Laya zero-shot | 13 | 0,279 | 0,213 | 0,405 | 0,035 | 0,424 | 0,069 | 0,81 |
| **C2 Laya fine-tuné** | 13–15 | **0,825 ± 0,005** | **0,827 ± 0,008** | 0,823 ± 0,002 | **0,541 ± 0,005** | 0,863 ± 0,002 | **0,005 ± 0,000** | **0,87 ± 0,01** |
| C3 Laya fine-tuné + augmentation | 13–17 | 0,776 ± 0,018 | 0,728 ± 0,034 | 0,832 ± 0,005 | 0,430 ± 0,033 | 0,872 ± 0,005 | 0,013 ± 0,003 | 0,84 ± 0,02 |
| C4-dev Qwen3-32B zero-shot | 13 | 0,506 | 0,366 | 0,817 | 0,084 | 0,856 | 0,075 | 0,76 |

Tests appariés (Holm sur 25 tests, `results/comparaisons_reels.csv`) :
- **C3 − C2, pour chacune des graines 13, 14 et 15** :
  - F1 : −0,045, −0,025 et −0,050, tous significatifs (p Holm = 0,0025) ;
  - faux positifs : +0,004 à +0,009, tous significatifs ;
  - choix|présent : +0,004 à +0,010, aucun significatif ;
  - abstention si gold absente : −0,02 à −0,06, aucun significatif après Holm (118 paires seulement).
- **C2 − C4-dev** : F1 +0,314 [+0,303 ; +0,325] ; faux positifs −0,070 ; abstention si gold absente +0,119 (p Holm 0,039).
- **C3 − C4-dev** : F1 +0,269.

**Constats**
- **H3 n'est pas soutenue sur les candidats réels de MultiATIS++ EN.** C3 fait moins bien que C2 (−0,025 à −0,050 de F1). L'écart vient presque entièrement des **faux positifs** sur les listes non vides des slots non exprimés : 10 % de ces listes contre 4 % pour C2. Le choix de la valeur, lui, est au même niveau.
- **Pourquoi, d'après les données.** Ce que l'augmentation de C3 apporte (robustesse aux longues listes et à l'absence de la gold, § balayages) ne sert pas ici.
  - Les listes réelles sont **courtes** (≈ 2 candidats), là où C2 l'emportait déjà avec l'oracle (k = 2).
  - Le piège réel est la **confusion entre slots** : la liste d'un slot non exprimé contient la valeur d'un autre slot de l'énoncé.
  - Or l'augmentation de C3 tire la moitié du temps des distracteurs `easy`, d'autres types de valeurs, faciles à rejeter. Elle l'entraîne peut-être à accepter une valeur dès qu'elle a le bon type.
- **Le fine-tuning reste décisif face au LLM** : C2 bat Qwen3-32B de 0,31 de F1 sur candidats réels, encore une fois par l'abstention (0,5 % de faux positifs contre 7,5 %).
- **Le choix est plus difficile qu'avec l'oracle** : choix|présent 0,86 contre 0,95 pour tous les juges. Les listes réelles mettent face à face des valeurs du même énoncé.
- **Latence.** Le p50 par requête vaut 0 sur ces runs : la plupart des requêtes ont une liste vide et ne sont pas jugées. La latence de H1 se mesure sur les runs oracle.

**Pistes** (décision à prendre)
1. **Augmentation centrée sur la confusion entre slots** : dans C3, remplacer `easy` par `other_slots` (valeurs des autres slots de l'énoncé) et surreprésenter les listes courtes (k = 2 ou 3).
2. **Entraîner sur des candidats réels** (générateurs appliqués au split train) : une condition « C2-réels », pour isoler l'effet de la distribution des candidats.
3. **Seuil d'abstention réglé sur le dev**, pour C2 et C3.

#### 26 octobre 2026 — décisions de l'auteur et phase 1 (gel du protocole)

- **Conséquences pour l'article.**
  - Présenter le juge comme un composant **digne de confiance** d'un système **hybride** : l'extracteur propose, le juge tranche ou s'abstient, et ses probabilités servent à décider d'agir ou de redemander.
  - QR4 (calibration, abstention, risque-couverture) devient centrale.
  - Mettre en avant la reproductibilité (graines, `code_sha256`, runs reproduits à l'identique).
- **Partage FAIR à préparer.**
  - Code et configs publics, archivés avec un DOI (Zenodo).
  - Checkpoints fine-tunés sur Hugging Face, avec fiche modèle.
  - Splits convertis de MTOP et Multi3WOZ si leurs licences le permettent. Pour MultiATIS++ (LDC), seulement les scripts de conversion et les sha256.

**Décisions**
1. **C3 v2 : augmentation centrée sur la confusion entre slots** (`configs/train/c3v2.yaml`, condition `C3v2`).
   - Nouveau préréglage de distracteurs `confusion` : autres slots de l'énoncé (poids 3), valeurs du même slot au train, ontologie, n-grammes de l'énoncé (poids 0,5), historique ; sans variantes de frontière.
   - Il remplace `easy` dans l'augmentation, qui tire désormais entre `confusion` et `hard_no_overlap`.
   - Les listes courtes sont surreprésentées : k tiré dans {2, 2, 3, 3, 5, 10, 20}.
   - Mesure sur 200 exemples de train : pour les slots non exprimés, 63 % des distracteurs viennent d'autres slots de l'énoncé avec `confusion`, contre 33 % avec `hard_no_overlap`.
   - C3 v1 est conservé, avec ses checkpoints et ses runs : il reste une condition de l'article (ablation de la conception de l'augmentation).
2. **Le checkpoint est choisi sur le F1 des slots du dev** (`train.select_on: f1`, valeur par défaut).
   - Calcul : questions de choix du dev, toutes requêtes gardées, dernière option = « non précisé ». VP si la bonne option est choisie et n'est pas « non précisé », FP si une autre valeur est choisie, FN si la gold n'est pas retrouvée.
   - En mode `two_step`, sans question de choix unique, le critère se rabat sur `balanced`.
   - Raison : la graine 16 de C3, choisie sur une perte faussée par la surconfiance.
   - Conséquence : C2 et C3 doivent être réentraînés avec ce critère pour l'article. Les checkpoints existants ont été choisis sur `balanced`.
3. **Décisions de protocole sur le dev uniquement.**
   - `make <condition> SPLIT=dev` évalue sur le dev ; `split` est ajouté aux groupes de `make tableaux`.
   - Le test ne sert plus qu'aux runs gelés.
   - Limite à noter : le choix du checkpoint utilise les 300 premiers exemples du dev de MultiATIS++ (sur 500), qui servent donc aussi aux décisions de protocole.
4. **Ensuite : MTOP et Multi3WOZ, en anglais et en français** (phase 2).

#### 26 octobre 2026 (soir) — phase 1 sur le **dev** : C2 (critère F1) contre C3 v2 — **valide**, décision

104 runs sur le **dev** de MultiATIS++ EN (500 exemples). C2 (graines 13 à 15) et C3 v2 (graines 13 à 17), réentraînés avec le critère `select_on: f1` ; balayage k × p_absent (`hard_no_overlap`) et candidats réels. Code `1f990bc7844b…` (commit `bb74008`), identique au poste. Le test n'a pas été ouvert.

**Entraînements** (F1 des slots sur les 300 premiers exemples du dev, oracle k = 5)
- C2 : 0,972 à 0,977, époque 3 gardée pour les trois graines.
- C3 v2 : 0,961 à 0,972, époque 3 gardée pour quatre graines et époque 2 pour une.
- Le F1 progresse encore entre les époques 2 et 3 : un essai à 4 ou 5 époques reste possible (sur le dev).

**Moyenne ± écart-type sur les graines**

| Réglage | F1 C2 | F1 C3 v2 | P C2 | P C3 v2 | Abst. si gold absente C2 | C3 v2 |
|---|---|---|---|---|---|---|
| **Candidats réels** | 0,876 ± 0,006 | **0,878 ± 0,007** | 0,871 | 0,874 | 0,78 ± 0,03 | 0,81 ± 0,04 |
| k = 2, p_absent = 0 | 0,965 ± 0,002 | 0,963 ± 0,003 | 0,967 | 0,968 | – | – |
| k = 5, p_absent = 0 | **0,957 ± 0,002** | 0,952 ± 0,005 | 0,954 | 0,950 | – | – |
| k = 5, p_absent = 0,5 | 0,597 ± 0,012 | **0,625 ± 0,007** | 0,775 | 0,891 | 0,79 | **0,97** |
| k = 10, p_absent = 0 | 0,930 ± 0,006 | **0,945 ± 0,003** | 0,906 | 0,946 | – | – |
| k = 10, p_absent = 0,5 | 0,557 ± 0,013 | **0,620 ± 0,005** | 0,657 | 0,883 | 0,66 | **0,97** |
| k = 20, p_absent = 0 | 0,827 ± 0,043 | **0,936 ± 0,004** | 0,743 | 0,948 | – | – |
| k = 20, p_absent = 0,5 | 0,477 ± 0,037 | **0,613 ± 0,007** | 0,484 | 0,894 | 0,58 | **0,99** |

(Tableau complet des 12 réglages : `make tableaux`. Les tests appariés sont dans `results/comparaisons_c3v2_c2_dev.csv`.)

**Tests appariés C3 v2 − C2** (graines 13, 14 et 15 ; 5 réglages × 3 métriques par graine, Holm)
- **Candidats réels : aucune différence significative** pour aucune des trois graines. F1 : +0,001, −0,003 et +0,003 ; faux positifs : de −0,0001 à +0,0004.
- k = 2 et k = 5 sans valeur absente : différences nulles, sauf la graine 14 à k = 5 (F1 −0,012, significatif).
- k = 20 sans valeur absente : F1 +0,065 à +0,148 ; faux positifs −0,008 à −0,023 ; tous significatifs.
- k = 10, p_absent = 0,5 : F1 +0,046 à +0,077 ; abstention si gold absente +0,27 à +0,37 ; tous significatifs.

**Constats**
- **C3 v2 corrige l'échec de C3 v1 sur candidats réels** : il fait jeu égal avec C2 (0,878 contre 0,876), là où C3 v1 perdait 0,03 à 0,05 sur le test. Les faux positifs reviennent au niveau de C2 (0,4 %).
- **C3 v2 garde l'avantage de l'augmentation** hors du réglage de C2. Il est même plus stable que C3 v1 : écart-type de 0,004 à 0,007 au lieu de 0,012 à 0,025.
- **Le prix sur le réglage de C2 devient faible** : −0,002 à −0,005 de F1 (contre −0,04 à −0,05 pour C3 v1).
- **H3, révisée** : une augmentation par options aléatoires **qui reproduit les confusions réelles** (valeurs des autres slots, listes courtes) rend le juge robuste aux changements de contexte, sans coût sur les candidats réels. Une augmentation générique (distracteurs aléatoires) ne le fait pas. Les deux versions sont à présenter dans l'article : c'est l'ablation de la conception de l'augmentation.

**Décision (fin de phase 1)** : **C3 v2 devient la condition C3 de la grille.** C3 v1 est gardé comme ablation. Les réglages d'entraînement (3 époques, `select_on: f1`, `unexpressed_ratio: 2.0`) sont figés pour la phase 2.

#### 26 octobre 2026 (soir) — préparation de la phase 2 : MTOP et Multi3WOZ, EN et FR

**Décision de l'auteur (option A) : pour Multi3WOZ, n'interroger que les slots du domaine gold du tour.**
- Multi3WOZ n'annote pas d'intention par tour, mais son ontologie a une « intention » par domaine (hotel, train, restaurant, attraction, taxi).
- Nouveau mode `requests.intent=domain` : le domaine gold du tour joue le rôle de l'intention gold de MultiATIS++ et MTOP. Les slots interrogés sont ceux du domaine, et le juge reçoit le domaine comme indice. Un tour sans domaine (15 % des tours) est interrogé sur toute l'ontologie (30 slots).
- Contexte multi-tours activé : historique des tours (4 derniers pour Laya) et état déjà rempli (`requests.filled=state_before`).
- Le Makefile ajoute ces deux réglages dès que `DATASET=multi3woz`.
- Mesure sur 300 tours du dev EN : 3 282 requêtes (≈ 11 par tour, contre 30 sans filtre), dont 12,6 % à slot exprimé. 8 paires gold sur 420 (1,9 %) sont hors du domaine du tour et ne sont pas interrogées, ce qui plafonne le rappel.
- Variante B (les 30 slots à chaque tour), écartée : 4 fois plus de requêtes, dont 99 % non exprimées. À mentionner dans l'article comme choix de protocole.

**Volumes du test** (requêtes à juger)
- MTOP : EN 4 386 énoncés et ≈ 29 000 requêtes ; FR 3 193 énoncés et ≈ 19 000 requêtes.
- Multi3WOZ : 6 266 tours par langue, ≈ 70 000 requêtes avec le filtre par domaine.

**Exécution** : `scripts/phase2.sh <carte> <jeu> [langues]`.
- Pour chaque langue, le script entraîne C2 (graines 13 à 15) et C3 v2 (13 à 17), puis évalue sur le **dev**, avec candidats oracle (point de grille) et candidats réels. C1 et C4-dev sont évalués aussi (graine 13).
- Il reprend où il s'est arrêté, et garde son journal dans `results/phase2-<jeu>-gpu<carte>.log`.

#### 27 octobre 2026 — module de décision : calibration et décision à trois zones (QR4)

**Méthode** (`laya_slots/decision/`, `experiments/decision.py`), pour la section Méthode de l'article :
- **Unité.** Une requête jugée : la distribution du juge sur ses options (candidats, puis « non précisé »), et l'option correcte (position de la gold, sinon « non précisé »). Les listes vides sont exclues : leur décision est certaine et commune à tous les juges.
- **Calibration réglée sur le dev.**
  - *Température* : p_T ∝ p^(1/T), T minimisant la NLL du dev. L'ordre des options est conservé, donc la décision ne change pas.
  - *Régression isotone* de la justesse sur la confiance de l'option retenue.
  - Mesures : NLL, Brier multiclasse, ECE (15 intervalles).
- **Décision à trois zones.** Pour une confiance c et un seuil τ :
  - c ≥ τ et une valeur retenue : **agir** (remplir le slot sans demander) ;
  - c ≥ τ et « non précisé » retenu : **ignorer** ;
  - c < τ : **redemander** (à l'utilisateur, ou à un juge plus coûteux).
  - **Courbe risque-couverture** sur tous les seuils, et son **AURC**.
  - Pour un risque cible (0,5 %, 1 %, 2 %, 5 %), on choisit sur le dev le seuil qui maximise la couverture, puis on l'applique tel quel.
- **Sans toucher au test.** Tant que le test est fermé, `experiments.decision --dev <run>` coupe le dev en deux moitiés de grappes (dialogues ou énoncés). Il règle sur l'une, évalue sur l'autre, dans les deux sens, et fait la moyenne. Avec `--test`, il règle sur le dev et évalue sur le test. Le résultat va dans `decision.json`.

**Premiers résultats** (dev de MultiATIS++ EN, graine 13, 2 plis ; « couv. » = part des requêtes décidées sans redemander)

| Juge | Réglage | Requêtes jugées | T | ECE brute → T | AURC | Couv. à 1 % de risque | Couv. à 2 % de risque |
|---|---|---|---|---|---|---|---|
| C2 | oracle k = 20, p_absent = 0,5 | 15 155 | 1,64 | 0,020 → 0,008 | 0,0033 | 0,907 | 0,979 |
| C3 v2 | oracle k = 20, p_absent = 0,5 | 15 155 | 1,40 | 0,004 → 0,002 | **0,0000** | **1,000** (risque 0,46 %) | 1,000 |
| C2 | **candidats réels** | 2 431 | 3,17 | 0,046 → 0,014 | 0,0208 | **0** (inatteignable) | 0,724 |
| C3 v2 | **candidats réels** | 2 431 | 2,73 | 0,046 → 0,021 | **0,0065** | **0,833** | **0,897** |

- **Laya fine-tuné est surconfiant**, d'autant plus que les candidats s'éloignent de l'entraînement : T de 1,4 à 3,2 (T > 1 signifie qu'il faut aplatir les probabilités). La température divise l'ECE par 2 à 3 et réduit fortement la NLL (candidats réels : 0,42 → 0,19 pour C2).
- **Sur candidats réels, C3 v2 trie bien mieux ses certitudes que C2** : AURC 0,0065 contre 0,0208, soit trois fois plus bas. À 1 % de risque, C2 ne peut rien décider sans redemander, alors que C3 v2 en décide 83 %. À 2 % de risque : 90 % contre 72 %. À F1 égal (§ phase 1), C3 v2 est donc **plus utilisable comme juge « digne de confiance »**. C'est un argument direct pour la revue, et un nouveau volet de H3.
- Avec l'oracle, la plupart des requêtes sont faciles (≈ 95 % « ignorer ») : les différences se jouent sur les réglages difficiles et sur les candidats réels.
- **À faire** : le même tableau pour toutes les graines et pour C4 (le LLM s'abstient-il mieux une fois calibré ?), puis sur MTOP et Multi3WOZ (phase 2), et une figure risque-couverture C2 / C3 v2 / C4.

#### 27 octobre 2026 — changement des LLM-juges (C4/C5) : mise en œuvre, **non lancé**

Décision de l'auteur du 27/10. Les résultats C4 antérieurs (Qwen3-32B) ne sont pas figés et seront refaits.

| Rôle | Modèle | Taille | Licence | Cartes | Cible |
|---|---|---|---|---|---|
| C4-dev (mise au point), à la place de Qwen3-32B | `Qwen/Qwen3.8-27B` | dense 27,8 B, BF16, 55,6 Go | Apache 2.0 | 1 | `make c4-dev` |
| **Référence C4/C5**, à la place de Qwen3-235B-2507 (jamais lancé) | `Qwen/Qwen3.8-Flash-Next-FP8` | MoE 180 B (10 experts actifs sur 512), FP8 par blocs, 185,5 Go | Qwen Community 1.0 | 2 | `make c4` |
| Autre famille | `mistralai/Mistral-Small-4-119B-2603` | MoE 119 B (4 experts actifs sur 128), FP8, 120,9 Go au format HF | Apache 2.0 | 2 | `make c4-alt` |
| Autre famille, légère | `google/gemma-4-31B-it` | dense 31,3 B, BF16, 62,5 Go | Apache 2.0 | 1 | `make c4-alt2` |

- Les quatre modèles sont multimodaux et servis **en texte seul** (`language_model_only`), avec `max_model_len = 4096`.
- Le prompt (`slot-judge-v1`) et la lecture des logprobs du premier jeton sont inchangés.
- **Raisonnement coupé selon la famille** (`chat_template_kwargs: auto`) : `enable_thinking: false` pour Qwen3.8 (actif par défaut) et pour Gemma 4 ; `reasoning_effort: none` pour Mistral Small 4. Ce réglage a été vérifié sur les gabarits de chat du Hub, et il équivaut à l'ancien réglage pour Qwen3.
- **Compatibilité.** Les quatre architectures figurent dans le registre de vLLM 0.30.0 (lecture du code, pas encore d'essai). Point le plus risqué : Qwen3.8-Flash-Next. vLLM 0.31 corrige des problèmes de mémoire sur cette architecture ; s'il échoue, on passera à vLLM ≥ 0.31.
- **Téléchargement.** Le dépôt Mistral publie deux copies des poids : seule la copie au format HF est téléchargée, celle que vLLM charge hors ligne. Il faut ≈ 425 Go pour les quatre LLM.
- **Essai rapide.** `scripts/tester_llm.sh <modèle> <cartes>` vérifie l'architecture, puis juge 20 exemples du dev. Il affiche F1, choix|présent, faux positifs, latence et la part de réponses « sans lettre » (signe d'un raisonnement mal coupé).
- Révisions exactes : dans `MODELES.lock.json` après `make modeles`.

#### 27 octobre 2026 — phase 2 : MTOP et Multi3WOZ, EN et FR, sur le **dev** — **valide**

80 runs (`scripts/phase2.sh`, du 26/10 15 h au 27/10 13 h 30) : pour chaque jeu × langue, C2 (graines 13 à 15) et C3 v2 (13 à 17) entraînés, puis C1, C2, C3 v2 et C4-dev évalués sur le dev, avec candidats oracle (k = 5, `hard_no_overlap`) et réels. Code `0d5abb75867e…` (commit `5656c86`). Le test n'a pas été ouvert. **C4-dev est encore Qwen3-32B** : ces runs seront refaits avec Qwen3.8-27B.

**F1 des slots** (moyenne ± écart-type ; C2 : 3 graines, C3 v2 : 5, C1 et C4-dev : 1)

| Jeu | Langue | Candidats | Rappel cand. | C1 | C2 | C3 v2 | C4-dev (Qwen3-32B) |
|---|---|---|---|---|---|---|---|
| MTOP | EN | oracle | 1,000 | 0,137 | **0,945 ± 0,003** | 0,941 ± 0,001 | 0,665 |
| MTOP | EN | réels | 0,766 | 0,423 | 0,692 ± 0,004 | **0,695 ± 0,005** | 0,543 |
| MTOP | FR | oracle | 1,000 | 0,140 | **0,937 ± 0,002** | 0,932 ± 0,004 | 0,625 |
| MTOP | FR | réels | 0,698 | 0,335 | 0,626 ± 0,006 | **0,639 ± 0,008** | 0,500 |
| Multi3WOZ | EN | oracle | 1,000 | 0,069 | 0,944 ± 0,001 | **0,951 ± 0,001** | 0,682 |
| Multi3WOZ | EN | réels | 0,911 | 0,197 | **0,870 ± 0,004** | 0,866 ± 0,005 | 0,604 |
| Multi3WOZ | FR | oracle | 1,000 | 0,054 | 0,923 ± 0,002 | **0,928 ± 0,001** | 0,616 |
| Multi3WOZ | FR | réels | 0,559 | 0,120 | **0,599 ± 0,008** | 0,595 ± 0,011 | 0,382 |

Volumes du dev : MTOP 15 096 requêtes (EN) et 9 832 (FR) ; Multi3WOZ 68 863 requêtes par langue (7 708 paires gold). Détail (précision, rappel, trame, choix|présent, faux positifs, abstention) : `make tableaux`. Tests appariés : `results/comparaisons_phase2_dev.csv`.

**Tests appariés** (Holm sur 64 tests)
- **C2 − C4-dev : C2 l'emporte partout**, avec 8 réglages sur 8 significatifs. F1 +0,13 à +0,31 et faux positifs −0,02 à −0,20.
- **C3 v2 − C2, par graine (13 à 15)** :
  - oracle MTOP : différences nulles ou faibles (−0,011 à 0,000) ;
  - oracle Multi3WOZ : C3 v2 légèrement meilleur (+0,003 à +0,009, significatif sur 5 graines sur 6), avec moins de faux positifs ;
  - réels : F1 non différent (−0,012 à +0,014), mais faux positifs un peu plus nombreux pour C3 v2 (+0,001 à +0,012, significatif sur presque toutes les graines).

**Décision et calibration sur candidats réels** (graine 13 ; 2 plis sur le dev ; couv. = part décidée sans redemander)

| Jeu, langue | Juge | T | ECE brute → T | AURC | Couv. à 2 % de risque | Couv. à 5 % de risque |
|---|---|---|---|---|---|---|
| MTOP EN | C2 / C3 v2 / C4-dev | 3,9 / 4,5 / 10,2 | 0,18 → 0,06 / 0,19 → 0,07 / 0,33 → 0,04 | 0,075 / **0,068** / 0,228 | 0,14 / **0,22** / 0,00 | 0,38 / **0,52** / 0,01 |
| MTOP FR | C2 / C3 v2 / C4-dev | 4,0 / 3,4 / 7,8 | 0,15 → 0,05 / 0,15 → 0,05 / 0,28 → 0,03 | 0,047 / **0,046** / 0,182 | 0,35 / **0,39** / 0,01 | **0,67** / 0,66 / 0,01 |
| Multi3WOZ EN | C2 / C3 v2 / C4-dev | 2,2 / 1,8 / 3,2 | 0,015 → 0,004 / 0,015 → 0,003 / 0,09 → 0,007 | **0,003** / 0,004 / 0,037 | **1,00** / 1,00 / 0,36 | 1,00 / 1,00 / 0,74 |
| Multi3WOZ FR | C2 / C3 v2 / C4-dev | 1,9 / 2,1 / 3,9 | 0,017 → 0,004 / 0,018 → 0,007 / 0,11 → 0,04 | **0,003** / 0,004 / 0,093 | **0,99** / 0,99 / 0,00 | 1,00 / 1,00 / 0,00 |

**Constats**
- **QR1 généralisé : Laya fine-tuné (C2 ou C3 v2) bat le LLM-juge 32 B sur les deux jeux et les deux langues**, avec les deux types de candidats. L'écart vient d'abord de l'abstention (faux positifs du LLM : 5 à 22 %, contre 0,3 à 2,8 % pour Laya). Il faudra le confirmer avec Qwen3.8 et les autres familles.
- **QR2 / H2 : avec les candidats réels, le rappel des candidats devient la première source d'erreur.** Il vaut 0,77 et 0,70 sur MTOP, 0,91 et **0,56** sur Multi3WOZ (EN et FR). Le juge choisit bien quand la valeur est présente (choix|présent 0,80 à 0,92 pour C2). **Les générateurs sont nettement plus faibles en français** : sur Multi3WOZ FR, le F1 réel plafonne à 0,60 pour un F1 oracle de 0,92. Améliorer les candidats français (tagger C0, GLiNER, lexique localisé) est la piste la plus rentable.
- **QR3 (langue) : avec l'oracle, l'écart EN → FR est faible** (−0,008 à −0,023 de F1) quand on entraîne et teste en français. Le juge lui-même marche bien en français ; l'écart réel vient des candidats. Le transfert EN → FR sans données françaises reste à mesurer.
- **H3 : C3 v2 fait jeu égal avec C2 sur les réglages de grille et sur les candidats réels** (différences de F1 inférieures à 0,015), avec un léger gain sur Multi3WOZ oracle. Son avantage attendu est hors distribution (balayages k × p_absent), pas encore mesuré sur ces jeux.
- **QR4 : les deux Laya fine-tunés sont bien plus fiables dans leurs certitudes que le LLM.**
  - AURC 3 à 30 fois plus faible.
  - Le LLM est très surconfiant (T jusqu'à 10) et ne peut presque rien décider seul à 2 % de risque, sauf en Multi3WOZ EN (36 %).
  - Entre C2 et C3 v2, l'avantage vu sur MultiATIS++ ne se généralise pas : C3 v2 est un peu meilleur sur MTOP et C2 un peu meilleur sur Multi3WOZ, mais les écarts sont faibles.
- **Multi3WOZ réel est plus facile que MTOP réel pour Laya** (0,87 contre 0,69 en EN) : les valeurs de base de données et l'historique donnent de bons candidats, et le filtre par domaine écarte les slots hors sujet.
- **Latence par requête** (p95, par lot) : Laya 5 à 12 ms, Qwen3-32B 7 à 29 ms. Le rapport est d'environ 5 sur Multi3WOZ, où les prompts sont longs à cause de l'historique. Sur MTOP, il est faible, car les deux juges ont des requêtes courtes. La mesure propre pour H1 reste à faire.

#### 27 octobre 2026 (soir) — essais rapides des nouveaux LLM-juges

`scripts/tester_llm.sh` : 20 exemples du dev de MultiATIS++ EN, candidats oracle (k = 5, `hard_no_overlap`), vLLM 0.30.0, transformers 5.16.1. Indicatif seulement : 20 exemples, 940 requêtes.

| Modèle | Cartes | Statut | F1 | choix\|présent | FP | « sans lettre » | p50 |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 1 | ok | 0,515 | 0,987 | 0,169 | 0 / 940 | 22 ms |
| Gemma-4-31B-it | 1 | ok | 0,634 | 0,987 | 0,103 | 0 / 940 | 11 ms |
| Qwen3.8-Flash-Next-FP8 | 2 | ok | 0,580 | 0,987 | 0,130 | 0 / 940 | 8 ms |
| Mistral-Small-4-119B | 2 | **échec** | – | – | – | – | – |

- Les trois modèles qui démarrent choisissent la valeur aussi bien. Le raisonnement est bien coupé (aucune réponse sans lettre d'option). Ils ne diffèrent que sur l'abstention.
- **Mistral-Small-4 ne démarre pas avec vLLM 0.30.0** : `ValueError: No model architectures are specified`, levée par `vllm/model_executor/models/mistral3.py:487`.
  - Le modèle multimodal `Mistral3ForConditionalGeneration` construit sa partie texte à partir de `text_config` (`model_type mistral4`), qui ne déclare pas d'architecture, et le registre de vLLM 0.30.0 n'a pas d'entrée Mistral 4.
  - Essais sans succès : format HF imposé (`config_format=hf`, `load_format=safetensors`, `tokenizer_mode=hf`), `language_model_only=false`, `hf_overrides` de l'architecture texte (`MistralLarge3ForCausalLM`, puis `DeepseekV3ForCausalLM`), et format natif Mistral (`consolidated*`, `config_format/load_format/tokenizer_mode=mistral`).
- **Décision (27/10) : Mistral-Small-4 est retiré de l'étude.** La comparaison à une autre famille repose sur Gemma-4-31B. On pourra réessayer après une mise à jour de vLLM (≥ 0.31), une fois les runs principaux figés, si un relecteur le demande.

#### 28 octobre 2026 — nouveaux LLM-juges contre Laya fine-tuné, MultiATIS++ EN, **dev** — **valide**

Runs complets du dev (500 exemples ; oracle k = 5 `hard_no_overlap`, et candidats réels), graine 13, code `5a5b381e1f49…`. Qwen3.8-Flash-Next avec candidats réels a demandé la correction mémoire de l'incident 12 (`c4ref-reels-dev-…-230046`). La réplique oracle du 27/10 22 h 35 est écartée (`OBSOLETE`).

| Candidats | Juge | F1 | P | R | Exact. de trame | choix\|présent | FP | p50 / p95 par req. | Débit |
|---|---|---|---|---|---|---|---|---|---|
| oracle | **C2** (3 graines) | **0,957 ± 0,002** | 0,954 | 0,960 | **0,821** | 0,970 | **0,002** | 1,2 / 2,1 ms | 691 req/s |
| oracle | C3 v2 (5 graines) | 0,952 ± 0,005 | 0,950 | 0,953 | 0,800 | 0,963 | 0,003 | 1,2 / 2,1 ms | 693 req/s |
| oracle | Gemma-4-31B | 0,429 | 0,277 | 0,956 | 0,008 | 0,967 | 0,146 | 10,4 / 11,4 ms | 38 req/s |
| oracle | Qwen3.8-Flash-Next-FP8 | 0,399 | 0,252 | 0,961 | 0,006 | 0,971 | 0,167 | 8,0 / 8,3 ms | 83 req/s |
| oracle | Qwen3.8-27B | 0,360 | 0,221 | 0,955 | 0,010 | 0,966 | 0,196 | 22,8 / 23,5 ms | 18 req/s |
| réels | C2 (3 graines) | 0,876 ± 0,006 | 0,871 | 0,880 | 0,605 | 0,911 | 0,004 | – / 6,2 ms | 1 070 req/s |
| réels | **C3 v2** (5 graines) | **0,878 ± 0,007** | 0,874 | 0,881 | **0,613** | 0,911 | 0,004 | – / 6,2 ms | 1 097 req/s |
| réels | Gemma-4-31B | 0,569 | 0,422 | 0,873 | 0,044 | 0,904 | 0,066 | – / 9,0 ms | 274 req/s |
| réels | Qwen3.8-Flash-Next-FP8 | 0,562 | 0,413 | 0,878 | 0,042 | 0,909 | 0,069 | – / 8,8 ms | 186 req/s |
| réels | Qwen3.8-27B | 0,548 | 0,400 | 0,869 | 0,038 | 0,900 | 0,072 | – / 20,9 ms | 143 req/s |

(Avec candidats réels, la plupart des requêtes ont une liste vide et ne sont pas jugées : p50 = 0, débit gonflé. Le p95 et la comparaison oracle sont les mesures pertinentes. Latence et débit sont mesurés par lot : la mesure de H1, requête par requête, reste à faire.)

- **C2 − LLM** (Holm sur 12 tests, `results/comparaisons_c2_llm_dev.csv`) :
  - oracle : F1 +0,53 à +0,60 ; faux positifs −0,14 à −0,19 ;
  - réels : F1 +0,30 à +0,32 ; faux positifs −0,06 à −0,07 ;
  - tous significatifs (p Holm = 0,0012, le plancher).
- **Le choix de la valeur est au même niveau pour tous** (choix|présent 0,90 à 0,97), y compris pour le LLM de 180 B. **Tout l'écart vient de l'abstention** : les LLM répondent à tort sur 15 à 20 % des slots non exprimés avec l'oracle, et sur 7 % avec les candidats réels, contre 0,2 à 0,4 % pour Laya fine-tuné.
- **La taille ne règle pas l'abstention** : le MoE de 180 B fait à peine mieux que le 27 B dense, et moins bien que Gemma-4-31B. Le résultat ne tient pas à une famille de modèles (Qwen et Gemma).
- **Coût** : par requête, Laya est 7 fois plus rapide que Qwen3.8-Flash-Next (p50 1,2 contre 8,0 ms) sur **une** carte contre **deux**, et 8 à 19 fois plus rapide que les LLM denses sur une carte. En débit : 691 req/s contre 18 à 83.
- **Décision (QR4, candidats réels, 2 plis sur le dev)** :
  - Les LLM sont très surconfiants : T de 5 à 16. Même une fois calibrés, leurs certitudes trient mal leurs erreurs : AURC de 0,32 à 0,45, contre 0,021 pour C2 (et 0,0065 pour C3 v2, § 27/10). Sur les requêtes jugées, ils se trompent dans 42 à 46 % des cas, contre 5 % pour Laya.
  - **Aucun LLM n'atteint 5 % de risque, même en redemandant tout ce qui est incertain.** C2 décide 99 % des requêtes à 5 % de risque et 72 % à 2 %.

#### 28 octobre 2026 — nouveaux LLM-juges sur MTOP et Multi3WOZ, EN et FR, **dev** — **valide**

24 runs (`scripts/phase2_llm.sh`, 27/10 23 h 30 → 28/10 8 h 40) : Qwen3.8-27B, Gemma-4-31B et Qwen3.8-Flash-Next-FP8, graine 13, candidats oracle (k = 5, `hard_no_overlap`) et réels. Code `5a5b381e…`. C2 et C3 v2 : runs de la phase 2 (code `0d5abb75…`, mêmes données, mêmes candidats à graine égale).

**F1 des slots** (C2 : 3 graines, C3 v2 : 5 ; LLM : 1)

| Jeu | Langue | Candidats | C2 | C3 v2 | Qwen3.8-27B | Gemma-4-31B | Qwen3.8-Flash-Next |
|---|---|---|---|---|---|---|---|
| MTOP | EN | oracle | **0,945** | 0,941 | 0,691 | 0,739 | 0,709 |
| MTOP | EN | réels | 0,692 | **0,695** | 0,568 | 0,580 | 0,567 |
| MTOP | FR | oracle | **0,937** | 0,932 | 0,687 | 0,732 | 0,716 |
| MTOP | FR | réels | 0,626 | **0,639** | 0,527 | 0,526 | 0,528 |
| Multi3WOZ | EN | oracle | 0,944 | **0,951** | 0,835 | 0,854 | 0,822 |
| Multi3WOZ | EN | réels | **0,870** | 0,866 | 0,768 | 0,779 | 0,761 |
| Multi3WOZ | FR | oracle | 0,923 | **0,928** | 0,760 | 0,764 | 0,737 |
| Multi3WOZ | FR | réels | **0,599** | 0,595 | 0,473 | 0,486 | 0,473 |

**C2 − LLM** (graine 13 ; 8 réglages × 3 LLM × 3 métriques = 72 tests, Holm ; `results/comparaisons_c2_llm_phase2_dev.csv`) : **les 72 différences sont significatives, toutes en faveur de C2.**
- F1 : +0,20 à +0,25 (MTOP oracle), +0,10 à +0,12 (MTOP réels), +0,09 à +0,19 (Multi3WOZ oracle), +0,10 à +0,13 (Multi3WOZ réels).
- Faux positifs : −0,13 à −0,19 sur MTOP oracle, −0,04 à −0,06 sur MTOP réels, −0,01 à −0,02 sur Multi3WOZ.
- Choix|présent : +0,05 à +0,18.

**Constats**
- **QR1 tient sur les quatre couples jeu × langue, avec trois LLM récents de deux familles.** Aucun LLM n'approche Laya fine-tuné ; le meilleur est le plus souvent Gemma-4-31B. Le MoE de 180 B ne fait pas mieux que les modèles de 27 et 31 B.
- **Sur ces jeux, l'écart vient aussi du choix, pas seulement de l'abstention.** Sur MultiATIS++, les LLM choisissaient aussi bien que Laya (0,97). Sur MTOP (113 intentions, 75 slots) et Multi3WOZ (historique, valeurs de base de données), leur choix|présent tombe à 0,72–0,88, contre 0,80–0,97 pour Laya.
- **Multi3WOZ est plus favorable aux LLM**, avec peu de faux positifs (1,6 à 2,6 %) grâce au filtre par domaine. **MTOP leur est défavorable** (15 à 20 % de faux positifs avec l'oracle).
- **Les nouveaux LLM font nettement mieux que Qwen3-32B sur Multi3WOZ** (oracle EN : 0,82–0,85 contre 0,68), à peu près pareil sur MTOP. Les runs Qwen3-32B de la phase 2 sont conservés, sous leur nom de modèle.
- **Coût** (p95 par requête, par lot) : Laya 5 à 12 ms. Qwen3.8-Flash-Next 8 à 13 ms sur deux cartes ; Gemma 8 à 28 ms ; Qwen3.8-27B 20 à 39 ms. Sur Multi3WOZ avec l'oracle, le débit de Laya est de 360 à 370 req/s, contre 28 à 74 req/s pour les LLM (5 à 13 fois moins).
**Décision sur candidats réels** (graine 13 ; 2 plis sur le dev ; AURC, puis couverture à 2 % et 5 % de risque)

| Jeu, langue | C2 | C3 v2 | Qwen3.8-27B | Gemma-4-31B | Qwen3.8-Flash-Next |
|---|---|---|---|---|---|
| MTOP EN | 0,075 · 0,14 · 0,38 | **0,068 · 0,22 · 0,52** | 0,220 · 0,00 · 0,00 | 0,224 · 0,00 · 0,00 | 0,245 · 0,00 · 0,00 |
| MTOP FR | 0,047 · 0,35 · **0,67** | **0,046 · 0,39** · 0,66 | 0,166 · 0,01 · 0,01 | 0,201 · 0,00 · 0,00 | 0,207 · 0,01 · 0,01 |
| Multi3WOZ EN | **0,003 · 1,00 · 1,00** | 0,004 · 1,00 · 1,00 | 0,009 · 0,89 · 1,00 | 0,027 · 0,00 · 1,00 | 0,027 · 0,30 · 1,00 |
| Multi3WOZ FR | **0,003 · 0,99 · 1,00** | 0,004 · 0,99 · 1,00 | 0,024 · 0,51 · 0,94 | 0,056 · 0,00 · 0,00 | 0,103 · 0,00 · 0,82 |

- **Laya fine-tuné trie ses erreurs bien mieux que les trois LLM** : AURC 3 à 40 fois plus faible.
  - Sur MTOP, aucun LLM ne peut décider seul plus de 1 % des requêtes à 5 % de risque, alors que Laya en décide 38 à 67 %.
  - Sur Multi3WOZ EN, Qwen3.8-27B est le seul LLM proche (AURC 0,009 contre 0,003).
- **Les LLM restent surconfiants** (T de 1,6 à 12,5). Gemma, le meilleur LLM en F1, est le plus surconfiant (T de 3,7 à 12,5), et le moins utilisable pour décider seul.
- Entre C2 et C3 v2 : avantage faible à C3 v2 sur MTOP, à C2 sur Multi3WOZ.

#### 30 octobre 2026 — transfert interlingue anglais → français (QR3), **dev** — **valide**

136 runs (`scripts/transfert.sh`, du 29/10 21 h au 30/10 12 h 30), code `16746567cf3c…`, graines 13 à 15 (C2) et 13 à 17 (C3 v2), sauf l'entraînement bilingue de Multi3WOZ (graine 13 seulement). Candidats oracle (k = 5, `hard_no_overlap`) et réels. « EN → FR » : checkpoint entraîné sur l'anglais seul (sélection sur le dev anglais), évalué sur le dev français. « EN+FR » : entraînement sur les deux langues (sélection sur le dev anglais). Les références FR → FR et EN → EN viennent des phases 1 et 2 (MultiATIS++ FR → FR : nouveau).

**F1 des slots sur le dev français** (moyenne ± écart-type sur les graines)

| Jeu | Candidats | Juge | FR → FR | **EN → FR** (sans données FR) | EN+FR → FR |
|---|---|---|---|---|---|
| MultiATIS++ | oracle | C2 | 0,949 ± 0,005 | 0,864 ± 0,012 | **0,957 ± 0,004** |
| MultiATIS++ | oracle | C3 v2 | 0,949 ± 0,005 | 0,785 ± 0,065 | **0,957 ± 0,003** |
| MultiATIS++ | réels | C2 | 0,861 ± 0,008 | 0,808 ± 0,017 | **0,862 ± 0,008** |
| MultiATIS++ | réels | C3 v2 | 0,857 ± 0,006 | 0,803 ± 0,022 | **0,868 ± 0,005** |
| MTOP | oracle | C2 | 0,937 ± 0,002 | 0,892 ± 0,003 | **0,947 ± 0,002** |
| MTOP | oracle | C3 v2 | 0,932 ± 0,004 | 0,882 ± 0,003 | **0,940 ± 0,004** |
| MTOP | réels | C2 | **0,626 ± 0,006** | 0,619 ± 0,004 | 0,623 ± 0,004 |
| MTOP | réels | C3 v2 | **0,639 ± 0,008** | 0,623 ± 0,009 | 0,632 ± 0,018 |
| Multi3WOZ | oracle | C2 | 0,923 ± 0,002 | 0,840 ± 0,001 | **0,924** (1 graine) |
| Multi3WOZ | oracle | C3 v2 | **0,928 ± 0,001** | 0,810 ± 0,009 | **0,928** (1 graine) |
| Multi3WOZ | réels | C2 | **0,599 ± 0,008** | 0,563 ± 0,008 | 0,583 (1 graine) |
| Multi3WOZ | réels | C3 v2 | **0,595 ± 0,011** | 0,563 ± 0,007 | 0,507 (1 graine, anomalie) |

**Effet de l'entraînement bilingue sur le dev anglais** (EN → EN contre EN+FR → EN, F1) :
- MultiATIS++ : C2 0,957 → 0,963 (oracle), 0,876 → 0,877 (réels) ; C3 v2 0,952 → 0,963, 0,878 → 0,884.
- MTOP : C2 0,945 → 0,949 et 0,692 → 0,682 ; C3 v2 0,941 → 0,944 et 0,695 → 0,693.
- Multi3WOZ (1 graine) : C2 0,944 → 0,944 et 0,870 → 0,871 ; C3 v2 0,951 → 0,954 et 0,866 → **0,722** (anomalie).

**Constats**
- **Sans données françaises (EN → FR), le juge perd 0,05 à 0,08 de F1 avec l'oracle** pour C2 : −0,085 sur MultiATIS++, −0,045 sur MTOP, −0,083 sur Multi3WOZ. Le checkpoint multilingue de Laya transfère donc en partie, sans égaler un entraînement en français.
- **Avec les candidats réels, la perte du transfert est plus faible** (−0,007 à −0,054) : le plafond vient alors des générateurs français, pas du juge.
- **L'entraînement bilingue est le meilleur choix** : avec l'oracle, il égale ou dépasse le français seul (+0,000 à +0,010) et l'anglais seul (+0,000 à +0,011). Il ne coûte rien en anglais, sauf l'anomalie ci-dessous. Avec les candidats réels, les écarts sont de l'ordre de l'écart-type.
- **C3 v2 transfère moins bien que C2 sans données françaises** (oracle : 0,785 ± 0,065 contre 0,864 sur MultiATIS++, avec une forte variance entre graines). L'augmentation, réglée sur des distracteurs anglais, n'aide pas le passage à une autre langue.
- **Anomalie, une seule graine.** C3 v2 bilingue sur Multi3WOZ, avec candidats réels, s'abstient à tort sur 2 029 paires gold présentes (≈ 450 pour les autres checkpoints), et son choix|présent tombe à 0,69. Avec l'oracle, le même checkpoint est bon (0,954 EN, 0,928 FR), et ses métriques d'entraînement sont normales (F1 dev 0,965). À vérifier avec d'autres graines avant toute conclusion.
- **Coût** : l'entraînement bilingue de Multi3WOZ porte sur 98 789 exemples (377 033 questions), soit ≈ 17 min par époque et ≈ 1 h par graine.

#### 30 octobre 2026 — point d'avancement, décisions et feuille de route

**Décisions de l'auteur (30/10)**
- **Grille des runs finaux** : C1, C2, **C3 v2**, et C4 avec Qwen3.8-Flash-Next (référence), Gemma-4-31B et Qwen3.8-27B. C3 v1 en ablation ; C0 et C5 dès qu'ils sont prêts.
- ~~Entraînement bilingue EN+FR comme condition principale~~ → **remplacé le 31/10 : modèles monolingues comme condition principale** (option A, après vérification de l'anomalie Multi3WOZ). Bilingue et transfert EN → FR en analyse (QR3).
- **Candidats** : oracle `hard_no_overlap`, k = 5, et candidats réels. `hard` et les balayages k × p_absent en analyse.
- **Entraînement** : 3 époques, checkpoint sur le F1 dev, `unexpressed_ratio` 2,0. **Graines** : 3 (C2), 5 (C3 v2), 3 (LLM, seul le tirage des candidats change).
- **Décision (QR4)** : température réglée sur le dev ; seuils à 2 % et 5 % de risque.
- **Test** : un seul passage par configuration figée.
- **Calendrier** : soumission au plus tôt. **C5 et la robustesse sont gardés.**

**Feuille de route** (✅ fait · ⏳ en cours · ☐ à faire)
1. ✅ Vérification de l'anomalie (Multi3WOZ bilingue, graines 14-15) : confirmée en partie ; retour aux modèles monolingues (31/10).
2. ⏳ **Latence requête par requête** pour H1 : `scripts/latence.sh` prêt (lots de taille 1, `settings.mesure = latence`), à lancer cartes libres.
3. ✅ **Figures générées par le code** : `experiments/figures.py` (pgfplots `.tex` + `.dat`, sans dépendance ; Qwen3-32B exclu par défaut). Sur le dev : 6 figures. La figure coût / F1 attend les runs de latence.
4. ☐ **C0 : tagger BIO** (mDeBERTa-v3), baseline d'extraction et générateur de candidats (meilleurs candidats français).
5. ☐ Balayages k × p_absent sur MTOP et Multi3WOZ (C2 / C3 v2), au moins k = 20 et p_absent = 0,5.
6. ☐ C4 few-shot.
7. ☐ Mode « deux temps » fine-tuné (H4).
8. ☐ C5 : LLM extracteur génératif (sortie JSON).
9. ☐ Robustesse : permutation des options, renommage des slots, paraphrase des descriptions.
10. ☐ Domaines non vus (QR3).
11. ☐ Échantillons pour l'analyse d'erreurs (200 cas par jeu et langue), classement manuel par l'auteur.
12. ☐ Protocole interne daté, puis **runs finaux sur le test** (grille bilingue), intervalles, Holm, tableaux et figures.
13. ✅ Hors code, préparé le 30/10 (sources consultées ce jour) :
    - `docs/etat_de_l_art.md` et `docs/references.bib` : 28 références du plan vérifiées, 3 corrigées (années), 43 entrées BibTeX, 10 travaux récents.
    - `docs/publication.md` : guide aux auteurs (ScienceDirect inaccessible : version archivée de 2023, DOAJ 2026) et licences.
    - `docs/LICENSE.proposition.md` : Apache-2.0 proposé pour le code.
    - **Décisions de l'auteur en attente** : titulaire du copyright, licence du code, licence des checkpoints MTOP (données CC BY-SA), accord LDC et checkpoint MultiATIS++, APC, déclaration IA générative.
    - Puis rédaction.


#### 31 octobre 2026 — vérification de l'anomalie Multi3WOZ bilingue, et retour aux modèles monolingues — **valide**

Graines supplémentaires (`scripts/transfert.sh 0 C multi3woz "14" "14 15"`, 30/10 21 h 45 → 31/10 4 h 40, code `16746567…`) : C2 graine 14, C3 v2 graines 14 et 15, entraînés sur EN+FR (sélection sur le dev anglais, candidats oracle), évalués sur le dev EN et FR.

| Multi3WOZ, F1 | Candidats | Langue | Monolingue (phase 2) | Bilingue EN+FR, par graine |
|---|---|---|---|---|
| C2 | oracle | EN | 0,944 ± 0,001 | 0,944 · 0,948 |
| C2 | oracle | FR | 0,923 ± 0,002 | 0,924 · 0,924 |
| C3 v2 | oracle | EN | 0,951 ± 0,001 | 0,954 · 0,952 · 0,953 |
| C3 v2 | oracle | FR | 0,928 ± 0,001 | 0,928 · 0,933 · 0,932 |
| C2 | réels | EN | 0,870 ± 0,004 | 0,871 · 0,862 |
| C2 | réels | FR | 0,599 ± 0,008 | 0,583 · 0,564 |
| C3 v2 | réels | EN | 0,866 ± 0,005 | **0,722 · 0,861 · 0,815** |
| C3 v2 | réels | FR | 0,595 ± 0,011 | **0,507** · 0,585 · 0,578 |

- **L'anomalie est confirmée en partie.** Avec l'oracle, l'entraînement bilingue est stable et au niveau du monolingue. Avec les candidats réels, C3 v2 bilingue est **instable d'une graine à l'autre** : 650 à 2 029 abstentions à tort, contre ≈ 450 en monolingue.
- **C2 bilingue perd aussi un peu en français** sur candidats réels : 0,574 en moyenne, contre 0,599.
- **Cause probable : le critère de sélection.** Le checkpoint est choisi sur le dev **anglais** avec les candidats **oracle**, qui ne voit ni les listes réelles ni le français. Piste d'amélioration à citer dans l'article : choisir le checkpoint sur un dev qui inclut des candidats réels et les deux langues.
- **Bilan, candidats réels, dev FR (bilingue − monolingue)** :
  - C2 : +0,001 (MultiATIS++), −0,003 (MTOP), −0,025 (Multi3WOZ) ;
  - C3 v2 : +0,011, −0,007, −0,038 (et instable).

**Décision de l'auteur (31/10) : option A, les modèles monolingues redeviennent la condition principale** (un modèle par jeu et par langue). L'entraînement bilingue et le transfert EN → FR sont présentés dans l'analyse du transfert (QR3). Cela remplace la décision du 30/10. Les entraînements monolingues existent déjà pour toutes les graines sur le dev.

### 5. Constats

Tous **provisoires** : sauf mention contraire, ils viennent des runs biaisés du 22/10 et sont à reconfirmer après relance.

- **Valide (tout le test, une graine) : Laya fine-tuné bat le LLM-juge zero-shot de 32 B sur les deux sous-tâches.** C2 : F1 0,919, choix|présent 0,956, 0,6 % de faux positifs. Qwen3-32B : F1 0,276, choix|présent 0,892, 25,7 % de faux positifs. L'écart de F1 vient surtout de l'abstention : le LLM répond sur un quart des slots non exprimés. À confirmer avec la référence 235B, en few-shot, et avec plusieurs graines. Attention : la comparaison oppose un juge entraîné sur la distribution du test et un LLM sans exemple ; c'est le cadre de H1, mais il faudra aussi C4 few-shot.
- **Latence (mesure par lot, non définitive) :** à lots pleins, C2 est environ 8 fois plus rapide que Qwen3-32B par requête (p50 1,1 ms contre 9,0 ms ; débit 823 contre 102 req/s), sur une seule H200 chacun. H1 vise ≥ 10× face au 235B : la mesure requête par requête reste à faire.
- **Exactitude de trame** (tous les slots d'un énoncé justes, aucun en trop) : C2 0,703, C4-dev 0,035, C1 0. Elle rend visible le coût des faux positifs.
- **Le fine-tuning règle d'un coup le choix et l'abstention** (essai C2, valide mais préliminaire). Avec 1 000 exemples et une époque, Laya passe d'une précision ≈ 0,01 à 0,74 et d'un choix|présent ≈ 0,07 à 0,86. C'est cohérent avec le diagnostic zero-shot ci-dessous (AUC 0,53) : l'information manquait au modèle, pas à la règle de décision.
- **Limite de cet essai** : C2 est entraîné et testé avec la même distribution de candidats (oracle k = 5, `hard`). La robustesse à un contexte qui change est l'objet de C3 (QR5), à mesurer sur les balayages k × p_absent et sur les candidats réels.

- **Laya zero-shot ne sépare pas exprimé et non exprimé.** En `two_step` (`choice`), P(exprimé) a une médiane de 0,598 sur les slots exprimés (n = 727) et de 0,554 sur les non exprimés (n = 11 628). AUC 0,534. Aucun seuil de 0,5 à 0,99 ne dépasse F1 0,012. Ni la calibration ni le seuil ne peuvent compenser : c'est un argument pour le fine-tuning (C2, C3). Ce balayage a été fait sur le test, à titre de diagnostic ; le seuil devra être réglé sur le dev.
- **Qwen3-32B choisit bien, mais s'abstient mal.** En `hard`, choix|présent ≈ 0,89, mais encore ≈ 30 % de faux positifs. Laya : ≈ 87 % de faux positifs.
- **Erreurs typiques de Laya.**
  - Il préfère des n-grammes qui englobent la gold : « on april » pour « april », « before 7 », « flight from charlotte ».
  - Il confond des types : « april » pour un jour du mois ou pour une ville.
  - Il s'abstient parfois à tort.
  - Test d'intégration sur 4 requêtes : `single` 4/4, `two_step` `noul` 2/4, `two_step` `choice` 3/4. Les probabilités restent proches de 0,5, même quand la réponse est juste.
- **Latence : la mesure actuelle ne teste pas H1.** Le p95 par requête est amorti par lot (Laya lot 128, LLM lot 256). Sur 200 exemples, le facteur LLM/Laya est ≈ 3 (8–10 ms contre 3–3,5 ms), pas 30. Sur 20 exemples, le LLM affichait 284 ms (lots mal remplis). H1 exige une mesure requête par requête, à taille de lot égale, avec le modèle 235B.
- **Coïncidence pilote / Laya en `hard`.** Les deux ont 33 bonnes réponses. Leurs choix ne sont identiques que sur 177 requêtes exprimées sur 727, et seules 4 bonnes réponses sont communes.

### 6. Incidents techniques et corrections

Pour l'annexe de reproductibilité.

| # | Symptôme | Cause | Correction |
|---|---|---|---|
| 1 | `uv` introuvable depuis le Makefile | uv installé dans `~/.local/bin`, absent du `PATH` | `PATH` complété dans le Makefile |
| 2 | Triton : « Failed to find C compiler » | torch sur GPU compile avec un compilateur C | `sudo apt install build-essential` ; gcc ajouté aux prérequis de `scripts/preparer.sh` |
| 3 | vLLM : « Cannot re-initialize CUDA in forked subprocess » | CUDA déjà initialisé par les graines torch avant le fork | `VLLM_WORKER_MULTIPROC_METHOD=spawn` (`llm_judge.py` et Makefile) |
| 4 | FlashInfer : « Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist » | FlashInfer compile ses noyaux d'échantillonnage avec `nvcc`, absent | `VLLM_USE_FLASHINFER_SAMPLER=0`. Sans effet sur les résultats : le juge ne lit que les logprobs. Installer `cuda-toolkit-13-0` sera sans doute nécessaire pour le 235B FP8 |
| 5 | `.env.exemple` absent sur la machine | les fichiers commençant par un point n'ont pas été copiés lors du premier envoi manuel | copie par `rsync` |
| 6 | P = R = 1,0 sur des runs en échec | convention quand 0 exemple est jugé | `make tableaux` ignore les runs `status: error` |
| 7 | Listes oracle vides ou appauvries | `candidates.fit: false` dans les configs oracle | `fit: true` et garde-fou dans le sampler (§ 4) |
| 8 | `LocalEntryNotFoundError` au chargement du modèle de base, hors ligne | commande lancée sans le Makefile : `.env` non lu, `HF_HOME` absent, cache par défaut vide | passer par `make` (qui lit `.env`), ou `set -a; source .env; set +a` |
| 9 | Modifications de `laya_slots/data/` et `tests/data/` jamais copiées sur la machine | `rsync --exclude data` exclut tout dossier nommé `data` | `--exclude /data` (seulement le dossier racine) |
| 10 | `metrics.json` indique `git_commit = 143f163` (`dirty`) pour des runs lancés avec un code bien plus récent | le dossier `.git` de la machine n'est pas mis à jour avec le code | `code_sha256` dans `metrics.json` (indépendant de `.git`) ; `make version` sur le poste et sur la machine avant chaque série |
| 11 | Grille lancée en `hard` au lieu de `hard_no_overlap` (22/10, 22 h) | code non recopié sur la machine après le changement de `oracle.yaml` | runs reclassés en ablation `hard` ; vérifier `make version` des deux côtés avant de lancer |
| 12 | Qwen3.8-Flash-Next : `CUDA out of memory` sur la carte 0 avec les candidats réels seulement (27/10) | GLiNER (≈ 1,9 Gio) chargé sur la carte 0 dans le processus principal, non compté par vLLM quand il dimensionne son cache | avec `CAND=reels`, les cibles LLM passent `gpu_memory_utilization=0.88` |
| 13 | Runs de la cible `c4` sur le dev nommés `c4-dev-…`, confondus avec ceux de la cible `c4-dev` | préfixe de cible + suffixe de split | la cible `c4` écrit désormais `results/c4ref-…` (runs du 27/10 22 h : `c4-dev-multiatis-…` = Flash-Next) |

### 7. Décisions ouvertes

- ~~Distracteurs `hard` contenant la gold~~ : tranché le 23/10/2026, `hard_no_overlap` pour la grille, `hard` en ablation (§ 3).
- **Relance** de tous les runs oracle (pilote, C1, C4-dev, en `easy` et `hard`) avec `fit: true`.
- **Seuil d'abstention** de Laya `two_step` : à régler sur le dev, pas sur le test.
- **Mesure de latence pour H1** : requête par requête, à taille de lot égale, avec Qwen3-235B.
- **CUDA toolkit** sur la machine pour le 235B FP8 (incident 4).
- **Objectif d'entraînement** : log score supervisé (choix actuel) ou RLCD comme Laya d'origine, si le module `laya.train` devient disponible. À justifier dans l'article.
- **`k = 20` dans l'augmentation C3** : ramené à 15 par `max_options`. Relever `max_options` (et `head_max_len`) ou retirer 20 de la liste.
- **Ratio de requêtes non exprimées à l'entraînement** (2,0) : à fixer, ou à traiter comme hyperparamètre réglé sur le dev.
- **Discipline dev / test.** Le test MultiATIS++ EN a déjà été ouvert plusieurs fois pendant la mise au point. Pour l'article : figer les configs (dont `select_on`, `unexpressed_ratio`, distracteurs) sur le dev, puis relancer une seule fois le test par configuration gelée, avec 3 graines (5 pour C3).
- **Traçabilité du commit** sur la machine (incident 10), à régler avant les runs de l'article.
- **Mistral-Small-4** (second LLM d'une autre famille) : à réessayer avec vLLM ≥ 0.31 si nécessaire (essais du 27/10).
- **Calendrier** : pas d'urgence selon l'auteur.

#### 31 octobre – 1er novembre 2026 — relance dev, C0, latence, puis évaluation finale sur le **test** — **valide**

- **Relance sur le dev** (`scripts/relance_dev.sh`) des résultats explorés sur le test : C1 (grille 0,027, réels 0,303), C3 v1 réentraîné avec `select_on f1` (grille 0,933, réels 0,836), ablation `hard` (C2 0,950, C3 v2 0,893). Anciens checkpoints C3 v1 et `*-hard` déplacés dans `checkpoints/_ancien-pre-gel/`.
- **C0, tagger BIO** mDeBERTa-v3-base (`laya_slots/candidates/tagger.py`, `experiments/train_tagger.py`, `scripts/c0.sh`), 3 jeux × 2 langues × 3 graines. Piège : poids fp16 chargés par transformers 5 → AdamW diverge (nan) ; corrigé en chargeant en fp32. Dev : bat le pipeline sur MultiATIS++ et MTOP, perd sur Multi3WOZ (valeurs non littérales : 27 % EN, 63 % FR).
- **Latence requête par requête** (`scripts/latence.sh`, dev) : Laya ≈ 8 ms (p50), LLM 25–50 ms, rapport 3 à 6 ; la 1re occurrence de chaque longueur d'entrée coûte ≈ 80 ms à Laya. H1 non soutenue en latence.
- **Protocole figé** (`docs/protocole_test.md`, commit `ba631bb`, `code_sha256 d80bc7ce…`), runs exploratoires antérieurs sur le test marqués `OBSOLETE` (139).
- **Test final** (`scripts/test_final.sh`, 31/10 11 h → 1/11 21 h 21) : **570 runs, 0 échec** (grille Laya et C0, analyses, 3 LLM). Analyse : `scripts/analyse_finale.py` → `results/analyse/` (tableau, familles de Holm 1–3, décisions).
  - F1 test, listes oracle : Laya 0,92–0,95, meilleur LLM 0,38–0,84 ; famille 1 : 96 tests sur 108 en faveur de C2 (les 72 sur F1 et faux positifs) ; sur MultiATIS++ les LLM choisissent aussi bien ou un peu mieux (jusqu'à +0,02).
  - C0 contre pipeline (famille 3) : +0,09 à +0,22 sur MultiATIS++ et MTOP, −0,12 (EN) et −0,21 (FR) sur Multi3WOZ.
  - C3 v2 ≈ C2 (famille 2), sauf MultiATIS++ FR réels (−0,022) ; C3 v1 réels 0,784 contre 0,828.
  - Décision (dev → test) : AURC de Laya 3 à 40 fois plus faible ; seuils à 5 % dépassés sur MultiATIS++ (risque réalisé 6,2–7,0 %).
  - Bilingue Multi3WOZ réels : instabilité confirmée sur le test (C3 EN 0,714 / 0,845 / 0,802).
- Article et README publics mis à jour avec les chiffres du test ; dépôt GitHub `propose-then-judge-slots` (privé).

## Points d'attention

- Le code Laya suit l'API documentée sur la fiche Hugging Face (`convaiinnovations/laya`, version 0.3.22, épinglée dans `uv.lock`). Il tourne sur GPU depuis le 22 octobre 2026.
- `labo.sh` suppose un dépôt GitHub et n'est pas utilisé : le dépôt reste local et le code est copié sur la machine par `rsync` (voir le journal d'expérimentation).
- Les benchmarks publics peuvent figurer dans les données d'entraînement du LLM de référence (contamination). Ce point sera signalé comme limite, et les splits non vus serviront de contrôle.
- Les références de l'état de l'art du plan de recherche sont à vérifier sur l'ACL Anthology avant citation.

## Contribuer

Les règles communes (format unifié, interfaces, répertoires de chacun, absence de fuite des valeurs gold vers les juges) sont dans [`CONTRAT.md`](CONTRAT.md). Code et identifiants en anglais, commentaires et documentation en français.
