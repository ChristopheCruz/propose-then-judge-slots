#!/usr/bin/env bash
# =============================================================================
#  phase2_llm.sh — LLM-juges (C4) sur MTOP et Multi3WOZ, EN et FR, sur le DEV.
#
#  Refait les runs C4 de la phase 2 avec les LLM retenus le 27/10 (les runs du 26-27/10
#  utilisaient Qwen3-32B) : candidats oracle (point de grille k=5, hard_no_overlap) et réels,
#  graine 13. Le test n'est PAS ouvert.
#
#  Usage (sur la machine GPU, à la racine du dépôt, dans tmux) :
#    scripts/phase2_llm.sh <cible> <cartes> [jeux] [langues]
#      cible   c4-dev (LLM_MODEL_DEV, 1 carte) | c4-alt2 (LLM_MODEL_ALT2, 1 carte) | c4 (LLM_MODEL, 2 cartes)
#    En parallèle :            scripts/phase2_llm.sh c4-dev 0      et      scripts/phase2_llm.sh c4-alt2 1
#    Puis, les deux libres :   scripts/phase2_llm.sh c4 0,1
#
#  Reprise : un run est sauté s'il existe déjà, complet, AVEC LE MÊME MODÈLE (les anciens runs
#  Qwen3-32B, qui portent les mêmes noms de dossier, ne comptent donc pas).
#  Journal : results/phase2-llm-<cible>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CIBLE="${1:?cible : c4-dev | c4-alt2 | c4}"
CARTES="${2:?cartes : 0, 1 ou 0,1}"
JEUX="${3:-mtop multi3woz}"
LANGUES="${4:-en fr}"

valeur_env() { grep -E "^$1=" .env | tail -1 | cut -d= -f2-; }
case "$CIBLE" in
  c4-dev)  MODELE="$(valeur_env LLM_MODEL_DEV)";  PREFIXE=c4-dev ;;
  c4-alt2) MODELE="$(valeur_env LLM_MODEL_ALT2)"; PREFIXE=c4-alt2 ;;
  c4)      MODELE="$(valeur_env LLM_MODEL)";      PREFIXE=c4ref ;;
  *) echo "cible inconnue : $CIBLE (c4-dev | c4-alt2 | c4)"; exit 2 ;;
esac
[[ -n "$MODELE" ]] || { echo "modèle de $CIBLE vide dans .env"; exit 2; }

fait() {   # nom de run sans horodatage -> vrai si un run complet de ce modèle existe
  local m
  for m in results/$1-*/metrics.json; do
    [[ -f "$m" ]] || continue
    python3 - "$m" "$MODELE" <<'EOF' && return 0
import json, sys
m = json.load(open(sys.argv[1]))
ok = m.get("status") == "complete" and (m.get("settings") or {}).get("judge_model") == sys.argv[2]
sys.exit(0 if ok else 1)
EOF
  done
  return 1
}

exec > >(tee -a "results/phase2-llm-$CIBLE.log") 2>&1
echo "==== $(date '+%F %T') $CIBLE ($MODELE) sur les cartes $CARTES : $JEUX / $LANGUES"
for JEU in $JEUX; do
  for LANGUE in $LANGUES; do
    for CAND in oracle reels; do
      # pas de « $([[ … ]] && …) » ici : sous set -e, un test faux dans une affectation arrête le script
      if [[ $CAND == reels ]]; then suffixe=-reels; opt_cand=CAND=reels; else suffixe=; opt_cand=; fi
      nom="$PREFIXE$suffixe-dev-$JEU-$LANGUE-s13"
      if fait "$nom"; then echo "  déjà fait : $nom"; continue; fi
      echo "== $(date '+%T') $nom"
      make --no-print-directory "$CIBLE" DATASET="$JEU" LANGUE="$LANGUE" SEED=13 SPLIT=dev GPUS="$CARTES" $opt_cand \
        || echo "  ÉCHEC : $nom (suite du script ; voir results/$nom-*/run.log)"
    done
  done
done
echo "==== $(date '+%F %T') fin $CIBLE"
