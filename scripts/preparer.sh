#!/usr/bin/env bash
# =============================================================================
#  preparer.sh — installe l'environnement, télécharge les modèles et prépare les
#                données avant de lancer les expériences (sur la machine courante).
#
#  À exécuter à la racine du dépôt, sur le serveur GPU (ou en local pour les tests CPU).
#  Normalement appelé par `make preparer` ; chaque étape peut être relancée seule.
#
#  Usage
#    scripts/preparer.sh                    toutes les étapes, dans l'ordre
#    scripts/preparer.sh --etape modeles    une seule étape (option répétable)
#    scripts/preparer.sh --oui              ne pose pas de question (installe uv si absent)
#
#  Étapes
#    prerequis   outils, GPU, espace disque
#    config      crée .env depuis .env.exemple s'il manque (puis s'arrête pour le compléter)
#    installer   Python 3.12 et `uv sync --frozen --all-extras` (versions de uv.lock)
#    modeles     Laya, GLiNER, embeddings, LLM -> $HF_HOME ; écrit MODELES.lock.json
#    donnees     MTOP et Multi3WOZ (téléchargés), MultiATIS++ (copie LDC dans data/raw/multiatis)
#    verifier    tests CPU, puis test d'intégration Laya (modèle réel)
#
#  Chaque étape est idempotente : relancer le script ne refait pas ce qui est déjà fait
#  (les jeux déjà convertis sont sautés, les modèles déjà en cache ne sont pas retéléchargés).
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
RACINE="$(pwd)"

err()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
info() { printf '\033[36m→ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }

ETAPES_TOUTES=(prerequis config installer modeles donnees verifier)
ETAPES=()
OUI=0
while (( $# )); do
  case "$1" in
    --etape) [[ -n "${2:-}" ]] || err "--etape demande un nom : ${ETAPES_TOUTES[*]}"
             [[ " ${ETAPES_TOUTES[*]} " == *" $2 "* ]] || err "étape inconnue : $2 (${ETAPES_TOUTES[*]})"
             ETAPES+=("$2"); shift 2 ;;
    --oui)   OUI=1; shift ;;
    -h|--help|aide) sed -n '2,/^# =====/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) err "option inconnue : $1 (voir --help)" ;;
  esac
done
(( ${#ETAPES[@]} )) || ETAPES=("${ETAPES_TOUTES[@]}")

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
UV_RUN=(uv run --no-sync)

charger_env() {
  [[ -f .env ]] || return 0
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  [[ -n "${HF_TOKEN:-}" ]] || unset HF_TOKEN
}

# -----------------------------------------------------------------------------
etape_prerequis() {
  info "Prérequis"
  local manque=0
  # gcc : Triton (torch, vLLM) compile ses modules CUDA au premier appel sur GPU.
  for c in git python3 curl gcc; do
    if command -v "$c" >/dev/null; then printf '  ok      %s\n' "$c"; else printf '  ABSENT  %s\n' "$c"; manque=1; fi
  done
  (( manque == 0 )) || err "installer les outils absents (sudo apt install git python3 curl build-essential)"

  if command -v uv >/dev/null; then
    printf '  ok      uv (%s)\n' "$(uv --version | cut -d' ' -f2)"
  else
    if (( ! OUI )); then
      read -r -p "  uv absent : l'installer dans ~/.local/bin ? [o/N] " r
      [[ "$r" =~ ^[oOyY]$ ]] || err "uv est nécessaire : https://docs.astral.sh/uv/"
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    command -v uv >/dev/null || err "uv installé mais introuvable dans le PATH"
    ok "uv installé"
  fi

  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU /'
  else
    warn "aucun GPU NVIDIA visible : seuls les tests CPU fonctionneront ici"
  fi

  charger_env
  if [[ -n "${HF_HOME:-}" ]]; then
    mkdir -p "$HF_HOME"
    local libre_go besoin_go=20
    [[ "${TELECHARGER_LLM:-1}" == 1 ]] && besoin_go=550
    libre_go=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc '0-9')
    if (( libre_go < besoin_go )); then
      warn "$HF_HOME : ${libre_go} Go libres, ~${besoin_go} Go conseillés"
    else
      printf '  ok      %s : %s Go libres\n' "$HF_HOME" "$libre_go"
    fi
  fi
}

etape_config() {
  info "Configuration"
  if [[ -f .env ]]; then
    ok ".env présent"
  else
    cp .env.exemple .env
    ok ".env créé depuis .env.exemple"
    echo "  Complète .env (HF_HOME, LLM_MODEL…), puis relance : make preparer"
    exit 0
  fi
  charger_env
  [[ -n "${HF_HOME:-}" ]] || err ".env : HF_HOME est vide"
  echo "  HF_HOME=$HF_HOME"
  echo "  LLM_MODEL=${LLM_MODEL:-(aucun)}  LLM_MODEL_DEV=${LLM_MODEL_DEV:-(aucun)}  TELECHARGER_LLM=${TELECHARGER_LLM:-1}"
  echo "  LLM_MODEL_ALT=${LLM_MODEL_ALT:-(aucun)}  LLM_MODEL_ALT2=${LLM_MODEL_ALT2:-(aucun)}"
}

etape_installer() {
  info "Environnement Python (uv.lock, tous les extras)"
  uv python install 3.12
  uv sync --frozen --all-extras
  "${UV_RUN[@]}" python - <<'EOF'
import importlib
for m in ("laya", "torch", "transformers", "vllm", "gliner", "sentence_transformers", "datasets"):
    mod = importlib.import_module(m)
    print(f"  ok      {m} {getattr(mod, '__version__', '?')}")
import torch
n = torch.cuda.device_count()
print(f"  CUDA    {torch.version.cuda}, {n} GPU visible(s)" + "".join(
    f"\n          {i}: {torch.cuda.get_device_name(i)}" for i in range(n)))
EOF
  ok "Environnement prêt (.venv)"
}

etape_modeles() {
  charger_env
  [[ -n "${HF_HOME:-}" ]] || err "HF_HOME non défini : lancer d'abord l'étape config"
  info "Modèles -> $HF_HOME"
  local args=()
  [[ "${TELECHARGER_LLM:-1}" == 1 ]] || args+=(--sans-llm)
  env -u HF_HUB_OFFLINE "${UV_RUN[@]}" python -m experiments.telecharger_modeles "${args[@]}"
  ok "Modèles en cache ; versions dans MODELES.lock.json (à commiter)"
}

preparer_jeu() {   # dataset lang [--download]
  local ds="$1" lang="$2"; shift 2
  if [[ -f "data/processed/$ds/$lang/MANIFEST.json" ]]; then
    printf '  déjà prêt  %s/%s\n' "$ds" "$lang"
    return 0
  fi
  "${UV_RUN[@]}" python -m laya_slots.data.prepare --dataset "$ds" --lang "$lang" "$@"
}

etape_donnees() {
  info "Données (data/processed)"
  for ds in mtop multi3woz; do
    for lang in en fr; do preparer_jeu "$ds" "$lang" --download; done
  done
  if [[ -d data/raw/multiatis ]] && find data/raw/multiatis -name 'train_EN.tsv' | grep -q .; then
    for lang in en fr; do preparer_jeu multiatis "$lang"; done
  else
    warn "MultiATIS++ absent : copier les TSV de la copie LDC (LDC2021T04) sous data/raw/multiatis/,"
    warn "puis relancer : make donnees (voir laya_slots/data/multiatis.py)"
  fi
  ok "Données prêtes"
}

etape_verifier() {
  info "Tests CPU"
  "${UV_RUN[@]}" pytest -q
  info "Test d'intégration Laya (modèle réel, hors ligne)"
  charger_env
  HF_HUB_OFFLINE=1 "${UV_RUN[@]}" pytest -q -m integration tests/judges -s
  ok "Vérifications passées. Premier run : make pilote"
}

for e in "${ETAPES[@]}"; do "etape_$e"; done
