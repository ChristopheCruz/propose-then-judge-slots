#!/usr/bin/env bash
# =============================================================================
#  evaluer_hf.sh — évaluation du modèle publié (docs/modele_hf.md), checkpoints/hf/multi-s<graine>.
#
#  Chaque (jeu, langue) est évalué avec listes oracle et listes réelles, sur le split demandé :
#    MTOP 6 langues, Multi3WOZ 4, MASSIVE 52 locales (dont sw, cy, is, km, am jamais vues), SGD en.
#  Le dev sert à choisir la graine ; le test se lance une seule fois, avec la graine retenue.
#
#  Usage (machine GPU, racine du dépôt, dans tmux) :
#    scripts/evaluer_hf.sh <carte> <split> "<graines>" [jeux]
#    scripts/evaluer_hf.sh 1 dev "13 14 15"
#    scripts/evaluer_hf.sh 0 test 14
#    scripts/evaluer_hf.sh 0 dev 14 "mtop sgd"        # sous-ensemble de jeux
#  Sorties : results/hf-<split>-…/ (sautées si un run complet existe déjà) ; journal results/hf/evaluer-<split>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CARTE="${1:?carte (0 ou 1)}"
SPLIT="${2:?split : dev ou test}"
GRAINES="${3:?graines, ex. \"13 14 15\"}"
JEUX="${4:-mtop multi3woz massive sgd}"
VARIANTE="${VARIANTE:-}"          # VARIANTE=_v2 scripts/evaluer_hf.sh … : checkpoints/hf/multi_v2-s<graine>

langues() {   # jeu -> langues préparées (data/processed/<jeu>/<langue>/)
  local d
  for d in data/processed/"$1"/*/; do [[ -d $d ]] && basename "$d"; done
}

fait() {   # préfixe de run (sans horodatage) -> vrai si un run complet existe
  local m
  for m in results/$1-*/metrics.json; do
    [[ -f "$m" ]] && grep -q '"status": "complete"' "$m" && return 0
  done
  return 1
}

run() {   # nom [variables make...]
  local nom=$1; shift
  if fait "$nom"; then echo "  déjà fait : $nom"; return 0; fi
  echo "== $(date '+%F %T') $nom"
  make --no-print-directory hf VARIANTE="$VARIANTE" GPUS="$CARTE" SPLIT="$SPLIT" SORTIE="results/$nom-$(date +%Y%m%d-%H%M%S)" "$@" \
    || echo "  ÉCHEC : $nom"
}

mkdir -p results/hf
exec > >(tee -a "results/hf/evaluer$VARIANTE-$SPLIT.log") 2>&1
echo "==== $(date '+%F %T') évaluation HF, split $SPLIT, graines $GRAINES, carte $CARTE, jeux $JEUX"
make --no-print-directory -s version 2>&1 | grep -i sha || true

for s in $GRAINES; do
  [[ -f checkpoints/hf/multi$VARIANTE-s$s/model.safetensors ]] || { echo "  checkpoint absent : multi$VARIANTE-s$s"; continue; }
  for jeu in $JEUX; do
    for langue in $(langues "$jeu"); do
      run "hf$VARIANTE-$SPLIT-$jeu-$langue-s$s" DATASET="$jeu" LANGUE="$langue" SEED="$s"
      run "hf$VARIANTE-$SPLIT-reels-$jeu-$langue-s$s" DATASET="$jeu" LANGUE="$langue" SEED="$s" CAND=reels
    done
  done
done
echo "==== $(date '+%F %T') fin"
