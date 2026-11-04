#!/usr/bin/env bash
# =============================================================================
#  phase2.sh — MTOP et Multi3WOZ, EN et FR : entraînements et évaluations sur le DEV.
#
#  Usage (sur la machine GPU, à la racine du dépôt, dans tmux) :
#    scripts/phase2.sh <gpu> <jeu> [langues]
#    scripts/phase2.sh 0 mtop            # carte 0 : MTOP en puis fr
#    scripts/phase2.sh 1 multi3woz en    # carte 1 : Multi3WOZ anglais seulement
#
#  Pour chaque langue :
#    1. C2 (graines 13-15) et C3 v2 (graines 13-17) : fine-tuning, puis évaluation sur le dev,
#       candidats oracle (point de grille k=5, hard_no_overlap) et candidats réels ;
#    2. C1 et C4-dev (graine 13) : mêmes évaluations.
#  Réglages figés en phase 1 (3 époques, checkpoint sur F1 dev). Multi3WOZ : slots du domaine gold,
#  historique et état rempli (le Makefile les ajoute). Le test n'est PAS ouvert.
#  Reprise : une étape déjà faite est sautée (checkpoint présent, ou run dev complet présent).
#  Journal : results/phase2-<jeu>-gpu<carte>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:?carte GPU (0 ou 1)}"
JEU="${2:?jeu : mtop | multi3woz}"
LANGUES="${3:-en fr}"
[[ "$JEU" =~ ^(mtop|multi3woz)$ ]] || { echo "jeu inconnu : $JEU"; exit 2; }

fait_run() {   # prefixe de run (c2, c3v2-reels...) graine -> vrai si un run dev complet existe
  local pat="results/$1-dev-$JEU-$LANGUE-s$2-*/metrics.json"
  compgen -G "$pat" >/dev/null && grep -lq '"status": "complete"' $pat
}

evaluer() {    # cible graine
  local cible="$1" s="$2"
  fait_run "$cible" "$s" || make --no-print-directory "$cible" DATASET="$JEU" LANGUE="$LANGUE" SEED="$s" GPUS="$GPU" SPLIT=dev
  fait_run "$cible-reels" "$s" || make --no-print-directory "$cible" DATASET="$JEU" LANGUE="$LANGUE" SEED="$s" GPUS="$GPU" SPLIT=dev CAND=reels
}

exec > >(tee -a "results/phase2-$JEU-gpu$GPU.log") 2>&1
for LANGUE in $LANGUES; do
  echo "==== $(date '+%F %T') phase 2 : $JEU $LANGUE sur la carte $GPU"
  for cond in c2 c3v2; do
    graines="13 14 15"; [[ "$cond" == c3v2 ]] && graines="13 14 15 16 17"
    for s in $graines; do
      echo "== $cond graine $s"
      [[ -f "checkpoints/$cond/$JEU-$LANGUE-s$s/model.safetensors" ]] \
        || make --no-print-directory "entrainer-$cond" DATASET="$JEU" LANGUE="$LANGUE" SEED="$s" GPUS="$GPU"
      evaluer "$cond" "$s"
    done
  done
  echo "== C1 et C4-dev graine 13"
  evaluer c1 13
  evaluer c4-dev 13
  echo "==== $(date '+%F %T') fin $JEU $LANGUE"
done
