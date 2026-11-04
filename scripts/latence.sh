#!/usr/bin/env bash
# =============================================================================
#  latence.sh — latence requête par requête (H1) : chaque juge reçoit une requête à la fois.
#
#  Les runs par lots (taille 64 à 256) amortissent le coût fixe d'un appel et ne mesurent pas la
#  latence d'un système en ligne. Ici : judge.batch_size=1 (une requête par appel au juge, une
#  passe avant pour Laya, une génération vLLM pour les LLM), sur les N premiers exemples du dev,
#  candidats oracle (k=5, hard_no_overlap), graine 13. Les settings des runs portent mesure=latence.
#
#  Usage (sur la machine GPU, cartes libres, dans tmux) :
#    scripts/latence.sh [jeux] [n_exemples]          # défaut : "multiatis mtop multi3woz" 100
#  Ordre : Laya (C2, C3 v2) et les LLM d'une carte sur la carte 0, puis Qwen3.8-Flash-Next sur 0,1.
#  Rien d'autre ne doit tourner sur les cartes (la mesure serait faussée).
#  Journal : results/latence.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

JEUX="${1:-multiatis mtop multi3woz}"
N="${2:-100}"
UN="judge.batch_size=1 data.limit=$N"

fait() {   # préfixe de run -> vrai si un run de latence complet existe
  local m
  for m in results/$1-*/metrics.json; do
    [[ -f "$m" ]] && grep -q '"status": "complete"' "$m" && grep -q '"mesure": "latence"' "$m" && return 0
  done
  return 1
}

lancer() {   # cible cartes jeu [ENTR]
  local cible=$1 cartes=$2 jeu=$3 entr=${4:-} nom
  nom="lat-$cible-$jeu-en"
  if fait "$nom"; then echo "  déjà fait : $nom"; return 0; fi
  echo "== $(date '+%T') $nom (cartes $cartes)"
  make --no-print-directory "$cible" DATASET="$jeu" LANGUE=en SEED=13 SPLIT=dev GPUS="$cartes" \
    ${entr:+ENTR=$entr} SORTIE="results/$nom-$(date +%Y%m%d-%H%M%S)" ARGS="$UN" || echo "  ÉCHEC : $nom"
}

exec > >(tee -a results/latence.log) 2>&1
echo "==== $(date '+%F %T') latence requête par requête : $JEUX, $N exemples"
# Cartes libres avant la première mesure : on attend la fin des autres processus GPU.
while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; do
  echo "  $(date '+%T') cartes occupées, attente : $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd' ')"
  sleep 300
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  carte /'
for jeu in $JEUX; do
  lancer c2 0 "$jeu"
  lancer c3v2 0 "$jeu"
  lancer c4-dev 0 "$jeu"
  lancer c4-alt2 0 "$jeu"
  lancer c4 0,1 "$jeu"
done
echo "==== $(date '+%F %T') fin"
