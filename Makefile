# Commandes du projet : installation, téléchargements, tests et expériences.
# `make` ou `make aide` liste les cibles. Toutes passent par l'environnement uv (.venv).
#
# Variables (en ligne de commande : make c1 DATASET=mtop LANGUE=fr GPUS=1 ARGS="data.limit=200")
#   CONFIG   config d'expérience (cible run)
#   SORTIE   dossier du run (défaut : results/<nom>-<horodatage>)
#   GPUS     cartes visibles (CUDA_VISIBLE_DEVICES)
#   DATASET  multiatis | mtop | multi3woz      LANGUE  en | fr      SEED  graine
#   ARGS     surcharges OmegaConf supplémentaires (cle.sous_cle=valeur ...)
#
# Les réglages locaux (HF_HOME, LLM_MODEL…) viennent de .env (modèle : .env.exemple).
# Lancé par labo.sh depuis un instantané, .env est lu dans le dépôt du serveur ($LABO_REPO).

ENV_FICHIER := $(firstword $(wildcard .env) $(if $(LABO_REPO),$(wildcard $(LABO_REPO)/.env)))
ifneq ($(ENV_FICHIER),)
include $(ENV_FICHIER)
export HF_HOME HF_TOKEN LLM_MODEL LLM_MODEL_DEV LLM_TP LLM_MODEL_ALT LLM_MODEL_ALT2 LLM_TP_ALT LLM_TP_ALT2
endif

SHELL     := bash
# uv s'installe dans ~/.local/bin, absent du PATH d'un shell qui n'a pas relu ~/.bashrc.
export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)
UV        := uv run --no-sync
HORODATE  := $(shell date +%Y%m%d-%H%M%S)
GPUS      ?= $(or $(CUDA_VISIBLE_DEVICES),0)
# Cartes d'une cible C4 selon son nombre de cartes TP : 1 -> GPUS, 2 -> 0,1.
GPUS_BASE := $(GPUS)
gpus_tp    = $(if $(filter 1,$(1)),$(GPUS_BASE),0,1)
DATASET   ?= multiatis
LANGUE    ?= en
SEED      ?= 13
LLM_TP    ?= 2
# LLM-juges de robustesse (autre famille) : Mistral Small 4 sur 2 cartes, Gemma 4 31B sur une.
LLM_TP_ALT  ?= 2
LLM_TP_ALT2 ?= 1
ARGS      ?=
DONNEES   := data.dataset=$(DATASET) data.lang=$(LANGUE) seed=$(SEED)
# Multi3WOZ (décision du 26/10) : slots du domaine gold du tour, historique et état déjà rempli.
DONNEES   += $(if $(filter multi3woz,$(DATASET)),requests.intent=domain requests.filled=state_before)
# DISTR=hard : ablation des distracteurs (grille : hard_no_overlap, voir configs/candidats/oracle.yaml).
# Change les candidats de tous les runs, et le dossier des checkpoints C2/C3 (suffixe -hard).
DISTR     ?=
DONNEES   += $(if $(DISTR),candidates.generators.0.params.distractors=$(DISTR))
# CAND=reels : candidats réels (configs/conditions/*_reels.yaml) au lieu des candidats oracle.
CAND      ?=
SUF_CAND  := $(if $(filter reels,$(CAND)),_reels)
# SPLIT=dev : évaluer sur le dev (décisions de protocole) ; le test ne sert qu'aux runs gelés.
SPLIT     ?= test
NOM_CAND  := $(if $(filter reels,$(CAND)),-reels)$(if $(filter-out test,$(SPLIT)),-$(SPLIT))
ifneq ($(and $(DISTR),$(filter reels,$(CAND))),)
$(error DISTR ne s'applique qu'aux candidats oracle, pas avec CAND=reels)
endif
DONNEES   += $(if $(filter-out test,$(SPLIT)),data.split=$(SPLIT))
# Candidats réels : GLiNER occupe ~2 Gio de la carte 0 dans le processus principal, que vLLM ne compte pas
# en dimensionnant son cache (OOM de Qwen3.8-Flash-Next, 27/10). On lui réserve donc un peu moins de mémoire.
LLM_MEM   := $(if $(filter reels,$(CAND)),judge.params.backend_kwargs.llm_kwargs.gpu_memory_utilization=0.88)
# ENTR : langues d'entraînement du checkpoint (défaut : LANGUE). Transfert : make c2 LANGUE=fr ENTR=en ;
# entraînement bilingue : make entrainer-c2 LANGUE=en ENTR=en+fr (le dev de sélection reste LANGUE).
ENTR      ?= $(LANGUE)
CKPT      := $(DATASET)-$(ENTR)-s$(SEED)$(if $(DISTR),-$(DISTR))
RUN_ID    := $(DATASET)-$(LANGUE)$(if $(filter-out $(LANGUE),$(ENTR)),-entr-$(ENTR))-s$(SEED)$(if $(DISTR),-$(DISTR))

# Runs hors ligne : les modèles doivent être en cache (make modeles).
# vLLM : moteur démarré par spawn (CUDA déjà initialisé par les graines) ; échantillonneur PyTorch
# au lieu de FlashInfer, qui compile ses noyaux avec nvcc (absent sans CUDA toolkit). Le juge ne lit
# que les logprobs : l'échantillonneur ne change pas les résultats.
EXEC_RUN  = CUDA_VISIBLE_DEVICES=$(GPUS) HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 $(UV) python -m experiments.run

.DEFAULT_GOAL := aide
.PHONY: c0 entrainer-c0 entrainer-c3v2 c3v2 intervalles comparer graines balayage version aide preparer prerequis config installer modeles donnees verifier verifier-gpu \
        tests tests-integration lint run pilote c1 c1-reels c4 c4-dev c4-alt c4-alt2 tester-llm tableaux \
        entrainer-c2 entrainer-c3 c2 c3

aide: ## liste des cibles
	@grep -hE '^[a-z0-9 -]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 20

# ---------------------------------------------------------------- installation
preparer: ## tout : prérequis, .env, environnement, modèles, données, vérifications
	bash scripts/preparer.sh

prerequis config installer modeles donnees verifier: ## une seule étape de scripts/preparer.sh
	bash scripts/preparer.sh --etape $@

version: ## code exécuté : empreinte (à comparer entre le poste et la machine), commit, distracteurs de la grille
	@$(UV) python -c "from experiments.run import code_sha256; print('code_sha256', code_sha256())"
	@git log -1 --format="commit %h %s" 2>/dev/null || echo "commit : pas de .git"
	@grep -o "distractors: [a-z_]*" configs/candidats/oracle.yaml

verifier-gpu: ## GPU vus par le pilote et par PyTorch
	nvidia-smi
	CUDA_VISIBLE_DEVICES=$(GPUS) $(UV) python -c "import torch; n = torch.cuda.device_count(); \
	print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, {n} GPU'); \
	[print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(n)]"

# ---------------------------------------------------------------- qualité
tests: ## tests CPU, sans réseau ni modèle
	$(UV) pytest -q

tests-integration: ## tests avec les vrais modèles (cache local)
	HF_HUB_OFFLINE=1 $(UV) pytest -q -m integration -s

lint: ## ruff (mêmes chemins que la CI)
	$(UV) ruff check experiments laya_slots/evaluation tests/experiments tests/evaluation

# ---------------------------------------------------------------- expériences
# C5 (extracteur LLM) n'a pas encore de cible.

run: ## un run : make run CONFIG=configs/... [SORTIE=...] (commande de labo.sh)
	@test -n "$(CONFIG)" || { echo "CONFIG manquant : make run CONFIG=configs/pilote.yaml"; exit 2; }
	$(EXEC_RUN) --config $(CONFIG) --output $(or $(SORTIE),results/$(basename $(notdir $(CONFIG)))-$(HORODATE)) $(ARGS)

pilote: ## pilote MultiATIS++ EN, oracle k=5, juge baseline (200 exemples)
	$(EXEC_RUN) --config configs/pilote.yaml --output $(or $(SORTIE),results/pilote-$(HORODATE)) $(ARGS)

c0: ## C0 tagger BIO seul (candidats du tagger, meilleur score) (après make entrainer-c0)
	$(EXEC_RUN) --config configs/conditions/c0_tagger.yaml \
	  --output $(or $(SORTIE),results/c0$(if $(filter-out test,$(SPLIT)),-$(SPLIT))-$(RUN_ID)-$(HORODATE)) $(DONNEES) \
	  candidates.generators.0.params.checkpoint=checkpoints/tagger/$(CKPT) $(ARGS)

c1: ## C1 Laya zero-shot, candidats oracle
	$(EXEC_RUN) --config configs/conditions/c1_laya_zeroshot$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c1$(NOM_CAND)-$(DATASET)-$(LANGUE)-s$(SEED)-$(HORODATE)) $(DONNEES) $(ARGS)

c1-reels: ## C1 Laya zero-shot, candidats réels
	$(EXEC_RUN) --config configs/conditions/c1_laya_zeroshot_reels.yaml \
	  --output $(or $(SORTIE),results/c1-reels-$(DATASET)-$(LANGUE)-s$(SEED)-$(HORODATE)) $(DONNEES) $(ARGS)

c4: GPUS = $(if $(filter 1,$(LLM_TP)),0,0,1)
c4: ## C4 LLM-juge (LLM_MODEL sur LLM_TP cartes), candidats oracle
	@test -n "$(LLM_MODEL)" || { echo "LLM_MODEL vide : le renseigner dans .env"; exit 2; }
	$(EXEC_RUN) --config configs/conditions/c4_llm_juge$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c4ref$(NOM_CAND)-$(DATASET)-$(LANGUE)-s$(SEED)-$(HORODATE)) $(DONNEES) \
	  judge.params.model=$(LLM_MODEL) judge.params.backend_kwargs.llm_kwargs.tensor_parallel_size=$(LLM_TP) $(LLM_MEM) $(ARGS)

c4-dev: ## C4 avec le LLM de mise au point (LLM_MODEL_DEV, une carte)
	@test -n "$(LLM_MODEL_DEV)" || { echo "LLM_MODEL_DEV vide : le renseigner dans .env"; exit 2; }
	$(EXEC_RUN) --config configs/conditions/c4_llm_juge$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c4-dev$(NOM_CAND)-$(DATASET)-$(LANGUE)-s$(SEED)-$(HORODATE)) $(DONNEES) \
	  judge.params.model=$(LLM_MODEL_DEV) judge.params.backend_kwargs.llm_kwargs.tensor_parallel_size=1 $(LLM_MEM) $(ARGS)

# Robustesse (décision du 27/10) : même prompt, même lecture des logprobs, LLM d'une autre famille.
# Raisonnement coupé selon la famille (chat_template_kwargs: auto). judge_model sépare les lignes
# de make tableaux.
c4-alt: GPUS = $(call gpus_tp,$(LLM_TP_ALT))
c4-alt: ## C4 avec le LLM de robustesse (LLM_MODEL_ALT, LLM_TP_ALT cartes)
	@test -n "$(LLM_MODEL_ALT)" || { echo "LLM_MODEL_ALT vide : le renseigner dans .env"; exit 2; }
	$(EXEC_RUN) --config configs/conditions/c4_llm_juge$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c4-alt$(NOM_CAND)-$(DATASET)-$(LANGUE)-s$(SEED)-$(HORODATE)) $(DONNEES) \
	  judge.params.model=$(LLM_MODEL_ALT) judge.params.backend_kwargs.llm_kwargs.tensor_parallel_size=$(LLM_TP_ALT) $(LLM_MEM) $(ARGS)

c4-alt2: GPUS = $(call gpus_tp,$(LLM_TP_ALT2))
c4-alt2: ## C4 avec le LLM de robustesse léger (LLM_MODEL_ALT2, LLM_TP_ALT2 cartes)
	@test -n "$(LLM_MODEL_ALT2)" || { echo "LLM_MODEL_ALT2 vide : le renseigner dans .env"; exit 2; }
	$(EXEC_RUN) --config configs/conditions/c4_llm_juge$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c4-alt2$(NOM_CAND)-$(DATASET)-$(LANGUE)-s$(SEED)-$(HORODATE)) $(DONNEES) \
	  judge.params.model=$(LLM_MODEL_ALT2) judge.params.backend_kwargs.llm_kwargs.tensor_parallel_size=$(LLM_TP_ALT2) $(LLM_MEM) $(ARGS)

# Essai rapide d'un LLM sur la machine (20 exemples du dev MultiATIS++ EN, candidats oracle) :
#   make tester-llm MODELE=google/gemma-4-31B-it CARTES=1        (CARTES=0,1 : TP=2 par défaut)
tester-llm: ## essai rapide d'un LLM-juge : make tester-llm MODELE=... CARTES=0 [TP=1]
	@test -n "$(MODELE)" || { echo "MODELE manquant : make tester-llm MODELE=Qwen/Qwen3.8-27B CARTES=0"; exit 2; }
	bash scripts/tester_llm.sh $(MODELE) $(or $(CARTES),$(GPUS)) $(TP)

# ---------------------------------------------------------------- fine-tuning (C2, C3)
# Le checkpoint va dans checkpoints/<c2|c3>/<jeu>-<langue>-s<graine>, le dossier que lisent les
# configs d'évaluation (judge.finetuned). Une graine = un entraînement : SEED=13, 14, 15 (C3 : 13..17).
EXEC_TRAIN = CUDA_VISIBLE_DEVICES=$(GPUS) HF_HUB_OFFLINE=1 $(UV) python -m experiments.train

entrainer-c0: ## fine-tuning du tagger BIO de C0 (mDeBERTa-v3) -> checkpoints/tagger/...
	CUDA_VISIBLE_DEVICES=$(GPUS) HF_HUB_OFFLINE=1 $(UV) python -m experiments.train_tagger --config configs/train/c0.yaml \
	  --output checkpoints/tagger/$(CKPT) $(DONNEES) train.langs=$(ENTR) $(ARGS)

entrainer-c2: ## fine-tuning C2 (sans augmentation) -> checkpoints/c2/...
	$(EXEC_TRAIN) --config configs/train/c2.yaml \
	  --output checkpoints/c2/$(CKPT) $(DONNEES) train.langs=$(ENTR) $(ARGS)

entrainer-c3: ## fine-tuning C3 (augmentation par options aléatoires) -> checkpoints/c3/...
	$(EXEC_TRAIN) --config configs/train/c3.yaml \
	  --output checkpoints/c3/$(CKPT) $(DONNEES) train.langs=$(ENTR) $(ARGS)

c2: ## C2 Laya fine-tuné, candidats oracle (après make entrainer-c2)
	$(EXEC_RUN) --config configs/conditions/c2_laya_ft$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c2$(NOM_CAND)-$(RUN_ID)-$(HORODATE)) $(DONNEES) judge.finetuned=checkpoints/c2/$(CKPT) $(ARGS)

entrainer-c3v2: ## fine-tuning C3 v2 (augmentation centrée sur la confusion entre slots) -> checkpoints/c3v2/...
	$(EXEC_TRAIN) --config configs/train/c3v2.yaml \
	  --output checkpoints/c3v2/$(CKPT) $(DONNEES) train.langs=$(ENTR) $(ARGS)

c3v2: ## C3 v2, candidats oracle (CAND=reels : réels) (après make entrainer-c3v2)
	$(EXEC_RUN) --config configs/conditions/c3v2_laya_ft_aug$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c3v2$(NOM_CAND)-$(RUN_ID)-$(HORODATE)) $(DONNEES) judge.finetuned=checkpoints/c3v2/$(CKPT) $(ARGS)

c3: ## C3 Laya fine-tuné + augmentation, candidats oracle (après make entrainer-c3)
	$(EXEC_RUN) --config configs/conditions/c3_laya_ft_aug$(SUF_CAND).yaml \
	  --output $(or $(SORTIE),results/c3$(NOM_CAND)-$(RUN_ID)-$(HORODATE)) $(DONNEES) judge.finetuned=checkpoints/c3/$(CKPT) $(ARGS)

# Balayage QR2/QR5 : k x taux de valeur absente pour une condition (c1, c2, c3, c4-dev, c4, c4-alt, c4-alt2).
#   make balayage COND=c3            (12 runs ; max_options=20 pour que k=20 ne soit pas tronqué)
BALAYAGE_K      ?= 2 5 10 20
BALAYAGE_ABSENT ?= 0.0 0.2 0.5
COND            ?= c2
balayage: ## k x p_absent pour COND (QR2, QR5), à la suite sur GPUS
	@for k in $(BALAYAGE_K); do for p in $(BALAYAGE_ABSENT); do \
	  echo "== $(COND) k=$$k p_absent=$$p"; \
	  $(MAKE) --no-print-directory $(COND) ARGS="candidates.generators.0.params.k=$$k candidates.generators.0.params.p_absent=$$p judge.params.max_options=20 $(ARGS)" || exit 1; \
	done; done

# Graines supplémentaires (protocole : 3 pour les conditions entraînées, 5 pour C3) :
# pour chaque graine, fine-tuning puis balayage k x p_absent (qui contient le point de grille k=5, p_absent=0).
#   make graines COND=c2 GRAINES="14 15"        make graines COND=c3 GRAINES="14 15 16 17" GPUS=1
# Pour c1 / c4-dev / c4-alt / c4-alt2 (sans entraînement) : seulement le balayage, ou ARGS pour un point.
GRAINES ?= 14 15
graines: ## entraînement + balayage pour chaque graine de GRAINES (COND=c2 | c3 | c1 | c4-dev)
	@for s in $(GRAINES); do \
	  echo "==== $(COND) graine $$s"; \
	  case "$(COND)" in c2|c3|c3v2) $(MAKE) --no-print-directory entrainer-$(COND) SEED=$$s || exit 1;; esac; \
	  $(MAKE) --no-print-directory balayage COND=$(COND) SEED=$$s || exit 1; \
	done

TABLEAU_GROUPES := condition,dataset,lang,settings.mesure,settings.train_data,split,settings.generators,settings.judge_model,settings.judge_mode,settings.judge_checkpoint,settings.judge_rotations,settings.judge_expressed_qtype,settings.distractors,settings.k,settings.p_absent,settings.fit,settings.limit
intervalles: ## IC bootstrap à 95 % (grappes = dialogues ou énoncés) : ci.json dans chaque run de results/
	$(UV) python -m experiments.intervalles results

comparer: ## comparaison appariée + Holm : make comparer PAIRES="runA runB runC runD" (A−B, C−D…)
	@test -n "$(PAIRES)" || { echo 'PAIRES manquant : make comparer PAIRES="results/a results/b"'; exit 2; }
	$(UV) python -m experiments.intervalles $$(set -- $(PAIRES); while [ $$# -ge 2 ]; do printf -- '--comparer %s %s ' "$$1" "$$2"; shift 2; done) $(ARGS)

tableaux: ## agrège results/ en results/tableau.{csv,tex} (graines regroupées, réglages séparés)
	$(UV) python -m experiments.tables results --group-by $(TABLEAU_GROUPES) \
	  --csv results/tableau.csv --latex results/tableau.tex
