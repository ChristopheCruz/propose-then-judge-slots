#!/usr/bin/env bash
# =============================================================================
#  c0.sh — C0, tagger BIO seul (mDeBERTa-v3) : entraînement puis évaluation, sur le DEV.
#
#  Pour chaque jeu x langue x graine : make entrainer-c0, puis make c0 SPLIT=dev
#  (le tagger est le seul générateur, top_k=1, et le juge prend son candidat).
#
#  Usage (sur la machine GPU, à la racine du dépôt, dans tmux) :
#    scripts/c0.sh <carte> [jeux] [langues] [graines]
#    scripts/c0.sh 0                                   # tout : 3 jeux x en fr x 13 14 15
#    scripts/c0.sh 1 "mtop multi3woz" fr "13"
#  Reprise : un checkpoint présent ou un run dev complet déjà là est sauté.
#  Journal : results/c0-gpu<carte>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CARTE="${1:?carte GPU (0 ou 1)}"
JEUX="${2:-multiatis mtop multi3woz}"
LANGUES="${3:-en fr}"
GRAINES="${4:-13 14 15}"

fait() {   # préfixe de run (sans horodatage) -> vrai si un run complet existe
  local m
  for m in results/$1-*/metrics.json; do
    [[ -f "$m" ]] && grep -q '"status": "complete"' "$m" && return 0
  done
  return 1
}

exec > >(tee -a "results/c0-gpu$CARTE.log") 2>&1
echo "==== $(date '+%F %T') C0, carte $CARTE : $JEUX / $LANGUES / graines $GRAINES"
for jeu in $JEUX; do
  for langue in $LANGUES; do
    for s in $GRAINES; do
      id="$jeu-$langue-s$s"
      if [[ ! -f checkpoints/tagger/$id/model.safetensors ]]; then
        echo "== $(date '+%T') entraînement checkpoints/tagger/$id"
        make --no-print-directory entrainer-c0 DATASET="$jeu" LANGUE="$langue" SEED="$s" GPUS="$CARTE" \
          || { echo "  ÉCHEC entraînement $id"; continue; }
      fi
      if fait "c0-dev-$id"; then echo "  déjà fait : c0-dev-$id"; continue; fi
      echo "== $(date '+%T') c0-dev-$id"
      make --no-print-directory c0 DATASET="$jeu" LANGUE="$langue" SEED="$s" SPLIT=dev GPUS="$CARTE" \
        || echo "  ÉCHEC : c0-dev-$id"
    done
  done
done
echo "==== $(date '+%F %T') fin"
