#!/usr/bin/env bash
# =============================================================================
#  relance_dev.sh — refait sur le DEV les résultats exploratoires obtenus sur le test
#  (MultiATIS++ EN), avec les réglages figés le 26/10 (3 époques, select_on f1).
#
#  Parties :
#    1  C1 Laya zero-shot : balayage k x p_absent (oracle), candidats réels, ablation hard.
#    3  C3 v1 (ablation) : réentraînement puis balayage k x p_absent et candidats réels.
#    H  ablation des distracteurs « hard » : C2 et C3 v2 entraînés et évalués avec DISTR=hard.
#  Les checkpoints C3 v1 et *-hard d'avant le gel (22-23/10, select_on balanced ou ancien défaut)
#  sont déplacés dans checkpoints/_ancien-pre-gel/ (pas supprimés) avant réentraînement.
#
#  Usage (sur la machine GPU, à la racine du dépôt, dans tmux) :
#    scripts/relance_dev.sh <carte> <parties> [graines C3 v1] [graines hard]
#    scripts/relance_dev.sh 0 1H                        # C1 + ablation hard (graines 13-15)
#    scripts/relance_dev.sh 1 3 "13 14 15 16 17"        # C3 v1, 5 graines
#  Reprise : un checkpoint réentraîné ou un run dev complet déjà là est sauté.
#  Journal : results/relance-dev-gpu<carte>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CARTE="${1:?carte GPU (0 ou 1)}"
PARTIES="${2:?parties : 1, 3, H ou une combinaison (1H, 13H)}"
GRAINES_C3="${3:-13 14 15 16 17}"
GRAINES_HARD="${4:-13 14 15}"
K="2 5 10 20"
ABSENT="0.0 0.2 0.5"
COMMUN="DATASET=multiatis LANGUE=en SPLIT=dev GPUS=$CARTE"

fait() {   # préfixe de run (sans horodatage) -> vrai si un run complet existe
  local m
  for m in results/$1-*/metrics.json; do
    [[ -f "$m" ]] && grep -q '"status": "complete"' "$m" && return 0
  done
  return 1
}

run() {   # nom cible [variables make...] : un run dev, sauté s'il existe déjà
  local nom=$1 cible=$2; shift 2
  if fait "$nom"; then echo "  déjà fait : $nom"; return 0; fi
  echo "== $(date '+%T') $nom"
  make --no-print-directory "$cible" $COMMUN SORTIE="results/$nom-$(date +%Y%m%d-%H%M%S)" "$@" \
    || echo "  ÉCHEC : $nom"
}

balayage() {   # cond graine
  local cond=$1 s=$2 k p
  for k in $K; do for p in $ABSENT; do
    run "$cond-dev-multiatis-en-s$s-k$k-p$p" "$cond" SEED="$s" \
      ARGS="candidates.generators.0.params.k=$k candidates.generators.0.params.p_absent=$p judge.params.max_options=20"
  done; done
}

entrainer() {   # cond graine [DISTR] : réentraîne si le checkpoint date d'avant le gel
  local cond=$1 s=$2 distr=${3:-} d
  d="checkpoints/$cond/multiatis-en-s$s${distr:+-$distr}"
  if [[ -f $d/config.yaml ]] && ! grep -q "select_on: f1" "$d/config.yaml"; then
    mkdir -p "checkpoints/_ancien-pre-gel/$cond"
    echo "  checkpoint d'avant le gel déplacé : $d -> checkpoints/_ancien-pre-gel/$cond/"
    mv "$d" "checkpoints/_ancien-pre-gel/$cond/"
  fi
  [[ -f $d/model.safetensors ]] && return 0
  echo "== $(date '+%T') entraînement $d"
  make --no-print-directory "entrainer-$cond" $COMMUN SEED="$s" ${distr:+DISTR=$distr}
}

exec > >(tee -a "results/relance-dev-gpu$CARTE.log") 2>&1
echo "==== $(date '+%F %T') relance sur le dev, parties $PARTIES, carte $CARTE"

if [[ $PARTIES == *1* ]]; then
  balayage c1 13
  run c1-reels-dev-multiatis-en-s13 c1 SEED=13 CAND=reels
  run c1-dev-multiatis-en-s13-hard c1 SEED=13 DISTR=hard
fi

if [[ $PARTIES == *3* ]]; then
  for s in $GRAINES_C3; do
    entrainer c3 "$s" || { echo "  ÉCHEC entraînement c3 s$s"; continue; }
    balayage c3 "$s"
    run "c3-reels-dev-multiatis-en-s$s" c3 SEED="$s" CAND=reels
  done
fi

if [[ $PARTIES == *H* ]]; then
  for cond in c2 c3v2; do
    for s in $GRAINES_HARD; do
      entrainer "$cond" "$s" hard || { echo "  ÉCHEC entraînement $cond s$s hard"; continue; }
      run "$cond-dev-multiatis-en-s$s-hard" "$cond" SEED="$s" DISTR=hard
    done
  done
fi
echo "==== $(date '+%F %T') fin"
