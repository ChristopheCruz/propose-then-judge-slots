#!/usr/bin/env bash
# =============================================================================
#  tester_llm.sh — essai rapide d'un LLM-juge (C4) sur la machine GPU.
#
#  Usage (à la racine du dépôt, modèle déjà en cache : make modeles) :
#    scripts/tester_llm.sh <modele> <cartes> [tp]
#    scripts/tester_llm.sh Qwen/Qwen3.8-27B 0
#    scripts/tester_llm.sh Qwen/Qwen3.8-Flash-Next-FP8 0,1          # tp = nombre de cartes
#    scripts/tester_llm.sh mistralai/Mistral-Small-4-119B-2603 0,1 2
#    FORCER=1 scripts/tester_llm.sh ...                              # relancer un essai réussi
#
#  Étapes :
#    1. architecture du modèle (config.json du cache) prise en charge par vLLM
#       (vllm.ModelRegistry.get_supported_archs()) ; versions de vLLM et transformers ;
#    2. run C4 sur 20 exemples du DEV de MultiATIS++ EN, candidats oracle (point de grille),
#       hors ligne, mêmes variables que le Makefile (make run) ;
#    3. résumé : statut, F1, choix|présent, taux de faux positifs, latence p50, taux de réponses
#       sans lettre (raisonnement mal coupé ?) ; en cas d'échec, lignes d'erreur du moteur.
#  Idempotent : un essai déjà complet (metrics.json "complete") n'est pas relancé.
#  Run : results/tester-llm-<modele>-tp<tp>/ ; journal : results/tester-llm-<modele>-tp<tp>.log
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

MODELE="${1:?modèle (id Hugging Face)}"
CARTES="${2:?cartes visibles, ex. 0 ou 0,1}"
TP="${3:-$(awk -F, '{print NF}' <<<"$CARTES")}"
[[ "$TP" =~ ^[0-9]+$ ]] || { echo "tp invalide : $TP"; exit 2; }

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
[[ -n "${HF_TOKEN:-}" ]] || unset HF_TOKEN
export HF_HUB_OFFLINE=1

NOM="tester-llm-$(tr '/:' '__' <<<"$MODELE")-tp$TP"
SORTIE="results/$NOM"
JOURNAL="results/$NOM.log"
mkdir -p results
DEBUT=$(( $( [[ -f "$JOURNAL" ]] && wc -l < "$JOURNAL" || echo 0) + 1 ))   # cet essai seulement
exec > >(tee -a "$JOURNAL") 2>&1
echo "==== $(date '+%F %T') $MODELE, cartes $CARTES, tp $TP"

# ---------------------------------------------------------------- 1. architecture
if ! CUDA_VISIBLE_DEVICES="" uv run --no-sync python - "$MODELE" <<'EOF'
import json, sys
from huggingface_hub import hf_hub_download
model = sys.argv[1]
try:
    cfg = json.load(open(hf_hub_download(model, "config.json")))
except Exception as e:  # noqa: BLE001
    sys.exit(f"  ÉCHEC   config.json absent du cache ({type(e).__name__}) : make modeles ?")
import transformers, vllm
from vllm import ModelRegistry
archs = cfg.get("architectures") or []
text = cfg.get("text_config") or {}
print(f"  vllm {vllm.__version__}, transformers {transformers.__version__}")
print(f"  architectures {archs}, model_type {cfg.get('model_type')}, texte {text.get('model_type', '-')}")
supported = set(ModelRegistry.get_supported_archs())
missing = [a for a in archs if a not in supported]
if missing or not archs:
    sys.exit(f"  ÉCHEC   architecture non prise en charge par vLLM {vllm.__version__} : {missing or archs}")
print("  ok      architecture prise en charge par vLLM")
EOF
then
  echo "==== $(date '+%F %T') arrêt : architecture"; exit 1
fi

# ---------------------------------------------------------------- 2. run
fait() { [[ -f "$SORTIE/metrics.json" ]] && grep -q '"status": "complete"' "$SORTIE/metrics.json"; }
if fait && [[ "${FORCER:-0}" != 1 ]]; then
  echo "  déjà fait : $SORTIE (FORCER=1 pour relancer)"
  CODE=0
else
  make --no-print-directory run CONFIG=configs/conditions/c4_llm_juge.yaml SORTIE="$SORTIE" GPUS="$CARTES" \
    ARGS="data.dataset=multiatis data.lang=en data.split=dev data.limit=20 seed=13 \
judge.params.model=$MODELE judge.params.backend_kwargs.llm_kwargs.tensor_parallel_size=$TP ${ARGS_LLM:-}"
  CODE=$?
fi

# ---------------------------------------------------------------- 3. résumé
if fait; then
  uv run --no-sync python - "$SORTIE" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.loads((d / "metrics.json").read_text())
dec = m.get("decomposition") or {}
def f(x): return "-" if x is None else f"{x:.3f}"
n = nl = 0
for line in (d / "predictions.jsonl").open():
    raw = json.loads(line).get("raw") or {}
    if "no_letter" in raw:
        n += 1; nl += bool(raw["no_letter"])
print(f"  statut {m['status']}  exemples {m['n_examples']}  requêtes {m['n_requests']}")
print(f"  F1 {f(m.get('slot_f1'))}  choix|présent {f(dec.get('choice_accuracy_given_present'))}  "
      f"faux positifs {f(dec.get('false_positive_rate'))}  latence p50 {f(m.get('latency_ms_p50'))} ms")
print(f"  sans lettre dans le top-k : {nl}/{n}" + ("  <- gabarit / raisonnement à vérifier" if nl > n * 0.05 else ""))
EOF
  echo "==== $(date '+%F %T') ok : $SORTIE"
  exit 0
fi
echo "  ÉCHEC   run (code $CODE)"
[[ -f "$SORTIE/metrics.json" ]] && grep -o '"status": "[a-z]*"' "$SORTIE/metrics.json"
echo "  -- cause (lignes d'erreur du moteur, journal $JOURNAL) :"
{ tail -n +"$DEBUT" "$JOURNAL"; cat "$SORTIE/run.log" 2>/dev/null; } \
  | grep -E "EngineCore.*Error|Error:|CUDA out of memory|OutOfMemory|not supported|ValueError|RuntimeError" \
  | tail -n 15
echo "==== $(date '+%F %T') échec : $MODELE"
exit 1
