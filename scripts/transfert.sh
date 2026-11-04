#!/usr/bin/env bash
# =============================================================================
#  transfert.sh — transfert interlingue anglais -> français (QR3), sur le DEV.
#
#  Trois parties, pour C2 (graines 13-15) et C3 v2 (graines 13-17), candidats oracle et réels :
#    A  EN -> FR sans données françaises : checkpoints anglais (déjà entraînés) évalués sur le dev FR.
#    B  FR -> FR pour MultiATIS++ (référence manquante ; MTOP et Multi3WOZ l'ont déjà, phase 2).
#    C  EN+FR : entraînement bilingue (dev de sélection anglais), évalué sur le dev EN et le dev FR.
#
#  Usage (sur la machine GPU, à la racine du dépôt, dans tmux) :
#    scripts/transfert.sh <carte> <parties> [jeux] [graines C2] [graines C3 v2]
#    scripts/transfert.sh 0 A                          # rapide : aucun entraînement
#    scripts/transfert.sh 1 BC "multiatis mtop"        # entraînements sur la carte 1
#    scripts/transfert.sh 0 C multi3woz "13" "13"      # Multi3WOZ bilingue, graine 13 seulement (long)
#  Reprise : un checkpoint présent ou un run dev complet déjà là est sauté.
#  Journal : results/transfert-gpu<carte>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CARTE="${1:?carte GPU (0 ou 1)}"
PARTIES="${2:?parties : A, B, C ou une combinaison (AB, BC, ABC)}"
JEUX="${3:-multiatis mtop multi3woz}"
GRAINES_C2="${4:-13 14 15}"
GRAINES_C3="${5:-13 14 15 16 17}"

fait() {   # préfixe de run (sans horodatage) -> vrai si un run complet existe
  local m
  for m in results/$1-*/metrics.json; do
    [[ -f "$m" ]] && grep -q '"status": "complete"' "$m" && return 0
  done
  return 1
}

evaluer() {   # cond jeu langue entr graine : oracle puis réels, sur le dev
  local cond=$1 jeu=$2 langue=$3 entr=$4 s=$5 id
  id="$jeu-$langue$([[ $entr != "$langue" ]] && echo "-entr-$entr" || true)-s$s"
  fait "$cond-dev-$id" || make --no-print-directory "$cond" DATASET="$jeu" LANGUE="$langue" ENTR="$entr" \
    SEED="$s" SPLIT=dev GPUS="$CARTE" || echo "  ÉCHEC : $cond-dev-$id"
  fait "$cond-reels-dev-$id" || make --no-print-directory "$cond" DATASET="$jeu" LANGUE="$langue" ENTR="$entr" \
    SEED="$s" SPLIT=dev GPUS="$CARTE" CAND=reels || echo "  ÉCHEC : $cond-reels-dev-$id"
}

entrainer() {   # cond jeu langue_dev entr graine
  local cond=$1 jeu=$2 langue=$3 entr=$4 s=$5
  [[ -f "checkpoints/$cond/$jeu-$entr-s$s/model.safetensors" ]] && return 0
  make --no-print-directory "entrainer-$cond" DATASET="$jeu" LANGUE="$langue" ENTR="$entr" SEED="$s" GPUS="$CARTE"
}

graines() { if [[ $1 == c2 ]]; then echo "$GRAINES_C2"; else echo "$GRAINES_C3"; fi; }

exec > >(tee -a "results/transfert-gpu$CARTE.log") 2>&1
echo "==== $(date '+%F %T') transfert, parties $PARTIES, carte $CARTE : $JEUX"
for jeu in $JEUX; do
  for cond in c2 c3v2; do
    for s in $(graines $cond); do
      if [[ $PARTIES == *A* ]]; then
        echo "== A $cond $jeu graine $s : EN -> FR"
        if [[ -f "checkpoints/$cond/$jeu-en-s$s/model.safetensors" ]]; then evaluer "$cond" "$jeu" fr en "$s"
        else echo "  checkpoint anglais absent : checkpoints/$cond/$jeu-en-s$s"; fi
      fi
      if [[ $PARTIES == *B* && $jeu == multiatis ]]; then
        echo "== B $cond $jeu graine $s : FR -> FR"
        entrainer "$cond" "$jeu" fr fr "$s" && evaluer "$cond" "$jeu" fr fr "$s"
      fi
      if [[ $PARTIES == *C* ]]; then
        echo "== C $cond $jeu graine $s : EN+FR"
        entrainer "$cond" "$jeu" en en+fr "$s" && { evaluer "$cond" "$jeu" en en+fr "$s"; evaluer "$cond" "$jeu" fr en+fr "$s"; }
      fi
    done
  done
done
echo "==== $(date '+%F %T') fin"
