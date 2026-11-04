#!/usr/bin/env bash
# =============================================================================
#  labo.sh — lancer les expérimentations sur le serveur GPU (2×H200) par SSH
#            et sauvegarder le code et les résultats sur GitHub.
#
#  À exécuter depuis ton poste, à la racine du dépôt git du projet.
#
#  Principe
#    - Le code circule par GitHub : poste --push--> GitHub --pull--> serveur.
#    - Chaque expérience tourne sur un instantané figé du code (git worktree au
#      commit courant) : modifier le code ensuite ne perturbe pas un run en cours.
#    - Chaque run écrit dans results/<run_id>/ : log, meta.json (commit, config,
#      GPU, durée, code de sortie) et ce que produit ton script d'expérience.
#    - À la fin du run, les petits fichiers de results/ sont commités et poussés
#      sur GitHub depuis le serveur. Les gros fichiers (checkpoints…) restent sur
#      le serveur, inventoriés dans results/GROS_FICHIERS.txt ; `labo.sh recuperer`
#      les rapatrie par rsync.
#
#  Commandes
#    init                     crée labo.env (configuration) à compléter
#    setup                    prépare le serveur (clone, outils, environnement)
#    sync                     pousse le code local sur GitHub et met le serveur à jour
#    run <config> [--gpus 0] [--nom x] [--sans-sync]
#                             lance une expérience en arrière-plan (tmux)
#    file <fichier>           lance une file d'expériences ; une ligne = "gpu config [nom]"
#                             les lignes d'un même GPU s'exécutent l'une après l'autre
#    statut                   runs en cours, derniers runs, occupation des GPU
#    logs <run_id> [-f]       affiche (ou suit) le log d'un run
#    stop <run_id>            arrête un run
#    sauver [message]         commit + push des résultats depuis le serveur
#    recuperer [run_id]       rapatrie les résultats (gros fichiers compris) par rsync
#    shell                    ouvre un shell sur le serveur, dans le dépôt
#
#  Prérequis
#    - Poste : bash, ssh, rsync, git.
#    - Serveur : Ubuntu, git, tmux (sinon repli sur nohup), python3, uv, pilotes NVIDIA.
#    - Le serveur doit pouvoir pousser sur GitHub : clé de déploiement avec droit
#      d'écriture (recommandé), ou AGENT_FORWARD=1 pour utiliser ta clé via ssh -A.
#      Avec AGENT_FORWARD, la sauvegarde automatique en fin de run échoue (la connexion
#      ssh est fermée depuis longtemps) : lance alors './labo.sh sauver'.
#    - Les configs doivent être commitées : le run tourne sur le commit, pas sur ta copie.
#    - Les dossiers non versionnés listés dans LIENS_PARTAGES (données converties,
#      checkpoints) sont liés depuis le dépôt du serveur dans l'instantané de chaque run.
#    - Les résultats du serveur arrivent sur la branche BRANCHE : 'sync' les intègre
#      (git pull --rebase) avant de pousser.
# =============================================================================
set -euo pipefail

ICI="$(pwd)"
CONF="${LABO_CONF:-$ICI/labo.env}"

err()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
info() { printf '\033[36m→ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ecrire_conf_exemple() {
  cat > "$CONF" <<'EOF'
# ---- Serveur ----
SERVEUR="moi@mon-serveur.exemple.fr"     # utilisateur@hôte (ou un alias de ~/.ssh/config)
SSH_PORT=22
SSH_CLE=""                               # ex. ~/.ssh/id_ed25519 (vide = défaut ssh)
AGENT_FORWARD=0                          # 1 = ssh -A (le serveur pousse avec ta clé)
REMOTE_DIR="/home/moi/laya-slots"        # chemin ABSOLU du dépôt sur le serveur

# ---- GitHub ----
DEPOT_GIT="git@github.com:mon-compte/laya-slots.git"
BRANCHE="main"
GIT_NOM="Serveur H200"                   # auteur des commits de résultats
GIT_EMAIL="serveur-h200@users.noreply.github.com"

# ---- Expériences ----
# Commande lancée pour chaque run, depuis l'instantané du code.
# Variables disponibles : $CONFIG, $RUN_ID, $RUN_DIR (dossier de sortie), $CUDA_VISIBLE_DEVICES,
# $LABO_REPO (dépôt du serveur). --no-sync : l'environnement partagé .venv n'est pas modifié
# par les runs (seul 'labo.sh sync' le met à jour, quand aucun run ne tourne).
EXP_CMD='uv run --no-sync python -m experiments.run --config "$CONFIG" --output "$RUN_DIR"'
# Variante par le Makefile (lit .env du dépôt du serveur, runs hors ligne avec HF_HUB_OFFLINE=1) :
# EXP_CMD='make run CONFIG="$CONFIG" SORTIE="$RUN_DIR" GPUS="$CUDA_VISIBLE_DEVICES"'
GPUS_DEFAUT="0"
MAX_MO=20                                # au-delà, un fichier n'est pas poussé sur GitHub
AUTORISER_ARBRE_SALE=0                   # 1 = lancer malgré des modifications non commitées (elles ne sont PAS dans le run)
LIENS_PARTAGES="data/raw data/processed checkpoints"   # non versionnés, liés dans chaque instantané
EOF
}

if [[ "${1:-aide}" =~ ^(aide|-h|--help)$ ]]; then
  sed -n '2,/^# =====/p' "$0" | sed 's/^# \{0,1\}//'; exit 0
fi

if [[ "${1:-}" == "init" ]]; then
  [[ -e "$CONF" ]] && err "$CONF existe déjà."
  ecrire_conf_exemple
  ok "Configuration créée : $CONF — complète-la, puis lance : ./labo.sh setup"
  grep -qxF 'labo.env' .gitignore 2>/dev/null || echo "(conseil : ajoute labo.env à .gitignore)"
  exit 0
fi

[[ -f "$CONF" ]] || err "Pas de configuration. Lance d'abord : ./labo.sh init"
# shellcheck disable=SC1090
source "$CONF"
for v in SERVEUR REMOTE_DIR DEPOT_GIT BRANCHE EXP_CMD; do
  [[ -n "${!v:-}" ]] || err "Variable $v manquante dans $CONF"
done
[[ "$REMOTE_DIR" == /* ]] || err "REMOTE_DIR doit être un chemin absolu."
SSH_PORT="${SSH_PORT:-22}"; GPUS_DEFAUT="${GPUS_DEFAUT:-0}"; MAX_MO="${MAX_MO:-20}"
GIT_NOM="${GIT_NOM:-Serveur}"; GIT_EMAIL="${GIT_EMAIL:-serveur@localhost}"
LIENS_PARTAGES="${LIENS_PARTAGES-data/raw data/processed checkpoints}"

SSH_OPTS=(-p "$SSH_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
[[ -n "${SSH_CLE:-}" ]] && SSH_OPTS+=(-i "${SSH_CLE/#\~/$HOME}")
[[ "${AGENT_FORWARD:-0}" == 1 ]] && SSH_OPTS+=(-A)

rsh() { ssh "${SSH_OPTS[@]}" "$SERVEUR" "$@"; }
q()   { printf '%q ' "$@"; }
LABO_DISTANT="$REMOTE_DIR/.labo"
distant() { rsh "bash $(q "$LABO_DISTANT/remote.sh") $(q "$@")"; }

# -----------------------------------------------------------------------------
# Partie exécutée sur le serveur (installée dans $REMOTE_DIR/.labo/remote.sh)
# -----------------------------------------------------------------------------
script_distant() {
cat <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
LABO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$LABO/env"
REPO="$REMOTE_DIR"; RES="$REPO/results"
# Une session ssh non interactive ne lit pas la fin de ~/.bashrc : uv (~/.local/bin) serait introuvable.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"   # un seul environnement partagé
# Identité git pour tous les commits ET les rebase (git pull --rebase en a besoin aussi).
export GIT_AUTHOR_NAME="$GIT_NOM" GIT_COMMITTER_NAME="$GIT_NOM"
export GIT_AUTHOR_EMAIL="$GIT_EMAIL" GIT_COMMITTER_EMAIL="$GIT_EMAIL"
LIENS_PARTAGES="${LIENS_PARTAGES-data/raw data/processed checkpoints}"
mkdir -p "$RES" "$LABO/arbres" "$LABO/sessions"
cd "$REPO"
VERROU="$LABO/git.lock"   # sérialise les opérations git sur le dépôt du serveur

horodatage() { date '+%F %T'; }
slug() { basename "$1" | sed 's/\.[^.]*$//; s/[^A-Za-z0-9_-]/-/g' | cut -c1-40; }
nouvel_id() {  # identifiant unique : le dossier de résultats est créé atomiquement
  local base n=1 id
  base="$(date +%Y%m%d-%H%M%S)-$(slug "$1")"; id="$base"
  until mkdir "$RES/$id" 2>/dev/null; do
    n=$((n + 1)); id="$base-$n"
    (( n < 100 )) || { echo "Impossible de créer $RES/$id" >&2; exit 1; }
  done
  printf '%s' "$id"
}

ecrire_meta() {  # fichier cle=valeur...
  local f="$1"; shift
  python3 - "$f" "$@" <<'PY'
import json, sys, os
f = sys.argv[1]
d = json.load(open(f)) if os.path.exists(f) else {}
for kv in sys.argv[2:]:
    k, _, v = kv.partition("=")
    d[k] = int(v) if v.lstrip("-").isdigit() else v
json.dump(d, open(f, "w"), ensure_ascii=False, indent=2)
PY
}

session() {  # nom commande
  local nom="$1" cmd="$2"
  if command -v tmux >/dev/null; then
    tmux new-session -d -s "$nom" "$cmd"
  else
    setsid nohup bash -c "$cmd" > "$LABO/sessions/$nom.out" 2>&1 < /dev/null &
    echo $! > "$LABO/sessions/$nom.pid"
  fi
}

session_active() {
  local nom="$1"
  if command -v tmux >/dev/null; then tmux has-session -t "$nom" 2>/dev/null
  else [[ -f "$LABO/sessions/$nom.pid" ]] && kill -0 "$(cat "$LABO/sessions/$nom.pid")" 2>/dev/null
  fi
}

sauver() {  # un seul push à la fois ; le verrou est rendu à la sortie du bloc
  { flock 9; sauver_verrouille "$@"; } 9>"$VERROU"
}

sauver_verrouille() {
  local msg="${1:-résultats}"
  cd "$REPO"
  local seuil=$(( MAX_MO * 1024 ))
  # Inventaire des gros fichiers (non poussés)
  { echo "# Fichiers >= ${MAX_MO} Mo gardés sur le serveur ($(hostname), $(horodatage))"
    echo "# taille_octets  chemin"
    find "$RES" -type f -size +"$(( seuil - 1 ))"k -printf '%s  %P\n' | sort -k2
  } > "$RES/GROS_FICHIERS.txt"
  # Petits fichiers seulement, sans les marqueurs internes (.pid, .arrete) des runs en cours
  find "$RES" -type f -size -"${seuil}"k ! -name '.pid' ! -name '.arrete' -print0 | xargs -0 -r git add -f --
  if git diff --cached --quiet; then echo "Rien de nouveau à sauvegarder."; return 0; fi
  git commit -q -m "résultats : $msg"
  local i
  for i in 1 2 3 4 5; do
    if git pull -q --rebase --autostash origin "$BRANCHE" && git push -q origin "HEAD:$BRANCHE"; then
      echo "Résultats poussés sur GitHub ($(git rev-parse --short HEAD))."; return 0
    fi
    echo "Push refusé, nouvel essai dans $((i * 10)) s…"; sleep $((i * 10))
  done
  echo "ÉCHEC du push : commit local conservé, relance './labo.sh sauver'." >&2; return 1
}

preparer() {  # id config gpus
  local id="$1" config="$2" gpus="$3" dir="$RES/$1" arbre="$LABO/arbres/$1"
  mkdir -p "$dir"
  local commit; commit="$(git rev-parse HEAD)"
  # la config doit exister dans le commit : le run tourne sur l'instantané, pas sur la copie locale
  if ! git cat-file -e "$commit:$config" 2>/dev/null; then
    echo "Config absente du commit ${commit:0:7} : $config (oubli de git add / commit ?)" >&2
    rmdir "$dir" 2>/dev/null || true
    return 1
  fi
  flock "$VERROU" git worktree add -q --detach "$arbre" "$commit"
  lier_partages "$arbre"
  ecrire_meta "$dir/meta.json" run_id="$id" config="$config" gpus="$gpus" \
    commit="$commit" branche="$BRANCHE" hote="$(hostname)" debut="$(horodatage)" statut="en_cours"
}

lier_partages() {  # arbre : lie les dossiers non versionnés (données, checkpoints) du dépôt
  local arbre="$1" p
  for p in $LIENS_PARTAGES; do
    [[ -e "$REPO/$p" && ! -e "$arbre/$p" ]] || continue
    mkdir -p "$(dirname "$arbre/$p")"
    ln -s "$REPO/$p" "$arbre/$p"
  done
}

delier_partages() {  # retire les liens avant 'git worktree remove --force'
  local arbre="$1" p
  for p in $LIENS_PARTAGES; do
    if [[ -L "$arbre/$p" ]]; then rm -f "$arbre/$p"; fi
  done
}

executer() {  # id config gpus
  local id="$1" config="$2" gpus="$3" dir="$RES/$1" arbre="$LABO/arbres/$1"
  local t0; t0=$(date +%s)
  cd "$arbre"
  export CONFIG="$config" RUN_ID="$id" RUN_DIR="$dir" CUDA_VISIBLE_DEVICES="$gpus" LABO_REPO="$REPO"
  # code de l'instantané en tête du chemin ; journal non tamponné (logs -f) ;
  # uv run ne resynchronise pas l'environnement partagé pendant que d'autres runs tournent
  export PYTHONPATH="$arbre${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 UV_NO_SYNC=1
  echo "[$(horodatage)] run $id | config=$config | GPU=$gpus | commit=$(git rev-parse --short HEAD)" | tee "$dir/log.txt"
  set +e
  # l'expérience tourne dans son propre groupe de processus : 'stop' ne vise qu'elle
  setsid bash -c "$EXP_CMD" > >(tee -a "$dir/log.txt") 2>&1 &
  local pid=$! code
  echo "$pid" > "$dir/.pid"
  # HUP : tmux kill-session ou perte du terminal ; on arrête proprement l'expérience
  trap 'kill -TERM -- -"$pid" 2>/dev/null' INT TERM HUP
  while :; do wait "$pid"; code=$?; kill -0 "$pid" 2>/dev/null || break; done
  trap - INT TERM HUP
  rm -f "$dir/.pid"
  sleep 1   # laisse tee finir d'écrire le log
  set -e
  local statut="termine"; (( code != 0 )) && statut="echec"
  [[ -f "$dir/.arrete" ]] && statut="arrete" && rm -f "$dir/.arrete"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader > "$dir/gpus.txt" 2>/dev/null || true
  ecrire_meta "$dir/meta.json" fin="$(horodatage)" duree_s="$(( $(date +%s) - t0 ))" code_sortie="$code" statut="$statut"
  echo "[$(horodatage)] fin : $statut (code $code)" | tee -a "$dir/log.txt"
  cd "$REPO"
  delier_partages "$arbre"
  flock "$VERROU" git worktree remove --force "$arbre" 2>/dev/null || true
  sauver "$id ($statut)" || true
}

cmd="${1:-}"; shift || true
case "$cmd" in
  maj)
    { flock 9    # pas de pull pendant qu'un run sauvegarde ses résultats
      git fetch -q origin
      git checkout -q "$BRANCHE"
      avant="$(sha1sum uv.lock 2>/dev/null || true)"
      git pull -q --rebase --autostash origin "$BRANCHE"
      git worktree prune
    } 9>"$VERROU"
    echo "Serveur à jour : $(git log -1 --format='%h %s')"
    if [[ -f pyproject.toml ]] && [[ "$avant" != "$(sha1sum uv.lock 2>/dev/null || true)" || ! -d .venv ]]; then
      if compgen -G "$LABO/arbres/*" >/dev/null; then
        echo "ATTENTION : dépendances modifiées mais des runs tournent ; 'uv sync' reporté."
      else
        uv sync --frozen --all-extras && echo "Environnement synchronisé."
      fi
    fi ;;
  lancer)   # config gpus [nom]
    config="$1"; gpus="$2"; id="$(nouvel_id "${3:-$1}")"
    preparer "$id" "$config" "$gpus"
    session "exp-$id" "bash $(printf %q "$LABO/remote.sh") executer $(printf '%q ' "$id" "$config" "$gpus")"
    echo "$id" ;;
  executer) executer "$@" ;;
  file)     # lignes "gpu config [nom]" sur l'entrée standard
    ts="$(date +%Y%m%d-%H%M%S)-$$"; mkdir -p "$LABO/files"; n=0
    while IFS= read -r ligne || [[ -n "$ligne" ]]; do
      ligne="${ligne%$'\r'}"                       # fichier édité sous Windows
      [[ "$ligne" =~ ^[[:space:]]*(#|$) ]] && continue
      read -r gpu config nom <<< "$ligne"
      if [[ ! "$gpu" =~ ^[0-9]+(,[0-9]+)*$ || -z "${config:-}" ]]; then
        echo "Ligne ignorée (attendu : gpu config [nom]) : $ligne" >&2; continue
      fi
      echo "$config ${nom:-}" >> "$LABO/files/$ts-gpu${gpu//,/_}.txt"; n=$((n + 1))
    done
    (( n > 0 )) || { echo "File vide."; exit 1; }
    for f in "$LABO/files/$ts"-gpu*.txt; do
      g="${f##*-gpu}"; g="${g%.txt}"; g="${g//_/,}"
      nom_s="file-$ts-gpu${g//,/_}"
      r="$(printf %q "$LABO/remote.sh")"; gq="$(printf %q "$g")"
      # la file est lue sur le descripteur 3 : les commandes du run ne consomment pas ses lignes
      session "$nom_s" "while read -r c n <&3; do id=\$(bash $r preparer_id \"\$c\" $gq \"\$n\" </dev/null) || continue; bash $r executer \"\$id\" \"\$c\" $gq </dev/null; done 3< $(printf %q "$f")"
      echo "GPU $g : $(wc -l < "$f") run(s) en file (session $nom_s)"
    done ;;
  preparer_id)  # config gpus [nom] -> prépare et affiche l'id
    id="$(nouvel_id "${3:-$1}")"; preparer "$id" "$1" "$2"; echo "$id" ;;
  statut)
    echo "== GPU =="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi indisponible"
    echo; echo "== Sessions actives =="
    if command -v tmux >/dev/null; then tmux ls 2>/dev/null | grep -E '^(exp|file)-' || echo "aucune"
    else for p in "$LABO"/sessions/*.pid; do [[ -e "$p" ]] && kill -0 "$(cat "$p")" 2>/dev/null && basename "$p" .pid; done; fi
    echo; echo "== 15 derniers runs =="
    mapfile -t derniers < <(printf '%s\n' "$RES"/*/ | sort -r | head -15)
    for m in "${derniers[@]}"; do
      [[ -f "$m/meta.json" ]] || continue
      python3 - "$m/meta.json" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
g = lambda k, v='': str(d.get(k, v))
print(f"{g('run_id'):32s} {g('statut'):9s} GPU={g('gpus'):4s} {g('duree_s','-'):>7s} s  {g('config')}")
PY
    done ;;
  logs)
    f="$RES/$1/log.txt"; [[ -f "$f" ]] || { echo "Log introuvable : $1" >&2; exit 1; }
    if [[ "${2:-}" == "-f" ]]; then tail -n 50 -f "$f"; else tail -n 80 "$f"; fi ;;
  stop)
    id="$1"; f="$RES/$id/.pid"
    [[ -f "$f" ]] || { echo "Aucun processus actif pour $id." >&2; exit 1; }
    pid="$(cat "$f")"; touch "$RES/$id/.arrete"
    # SIGTERM (intercepte-le dans ton code pour sauvegarder proprement), SIGKILL 60 s plus tard
    # si le run tourne encore (.pid inchangé : on ne tue pas un groupe dont le numéro a été réutilisé)
    kill -TERM -- -"$pid" 2>/dev/null || true
    setsid nohup bash -c "sleep 60; [ \"\$(cat $(printf %q "$f") 2>/dev/null)\" = $pid ] && kill -KILL -- -$pid 2>/dev/null" >/dev/null 2>&1 < /dev/null &
    echo "Arrêt demandé pour $id ; le statut et la sauvegarde suivent." ;;
  sauver) sauver "${1:-sauvegarde manuelle}" ;;
  *) echo "Commande distante inconnue : $cmd" >&2; exit 2 ;;
esac
REMOTE
}

env_distant() {
  local v
  for v in REMOTE_DIR BRANCHE EXP_CMD MAX_MO GIT_NOM GIT_EMAIL LIENS_PARTAGES; do
    printf '%s=%q\n' "$v" "${!v}"
  done
}

installer_distant() {
  # Écriture dans un fichier temporaire puis mv : un run en cours (bash lit son script au fil
  # de l'exécution) garde l'ancien fichier au lieu de lire un script modifié sous ses pieds.
  local r e
  r="$(printf %q "$LABO_DISTANT/remote.sh")"; e="$(printf %q "$LABO_DISTANT/env")"
  { script_distant; } | rsh "mkdir -p $(q "$LABO_DISTANT") && cat > $r.tmp && chmod +x $r.tmp && mv -f $r.tmp $r"
  env_distant | rsh "cat > $e.tmp && mv -f $e.tmp $e"
}

# -----------------------------------------------------------------------------
# Commandes locales
# -----------------------------------------------------------------------------
verifier_git_local() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || err "Lance labo.sh depuis le dépôt git du projet."
  if [[ -n "$(git status --porcelain --untracked-files=no)" && "${AUTORISER_ARBRE_SALE:-0}" != 1 ]]; then
    git status --short --untracked-files=no
    err "Modifications non commitées : commite-les (le run doit correspondre à un commit précis)."
  fi
}

integrer_distant() {
  # Le serveur pousse ses résultats sur BRANCHE : sans les intégrer, le push serait refusé.
  git ls-remote --exit-code -q origin "refs/heads/$BRANCHE" >/dev/null 2>&1 || return 0   # dépôt vide
  git fetch -q origin "$BRANCHE"
  git merge-base --is-ancestor FETCH_HEAD HEAD && return 0
  local courante; courante="$(git symbolic-ref -q --short HEAD || true)"
  [[ "$courante" == "$BRANCHE" ]] ||
    err "origin/$BRANCHE contient des commits absents de ta branche (${courante:-HEAD détachée}) : intègre-les d'abord (git rebase origin/$BRANCHE)."
  info "Intégration des commits du serveur (résultats)…"
  git pull -q --rebase origin "$BRANCHE" || err "Échec du rebase : résous le conflit, puis relance."
}

cmd_sync() {
  verifier_git_local
  info "Push du code local sur GitHub ($BRANCHE)…"
  local i
  for i in 1 2 3; do   # un run peut pousser ses résultats entre notre fetch et notre push
    integrer_distant
    git push -q origin "HEAD:$BRANCHE" && break
    (( i < 3 )) || err "Push refusé trois fois."
    info "Push refusé (résultats poussés entre-temps ?), nouvel essai…"; sleep 2
  done
  info "Mise à jour du serveur…"
  installer_distant
  distant maj
  ok "Code synchronisé : $(git log -1 --format='%h %s')"
}

cmd_setup() {
  info "Vérification du serveur $SERVEUR…"
  rsh 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
       for c in git tmux python3 uv nvidia-smi rsync flock; do
         if command -v $c >/dev/null; then printf "  ok      %s\n" $c; else printf "  ABSENT  %s\n" $c; fi
       done
       nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed "s/^/  GPU /"'
  echo "  (tmux absent : repli sur nohup ; uv absent : curl -LsSf https://astral.sh/uv/install.sh | sh)"

  info "Accès GitHub depuis le serveur…"
  if rsh 'ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | grep -q "successfully authenticated"'; then
    ok "Le serveur est authentifié sur GitHub."
  else
    echo "  Le serveur ne s'authentifie pas sur GitHub. Deux options :"
    echo "   1. Clé de déploiement : sur le serveur, ssh-keygen -t ed25519 -f ~/.ssh/github_deploy,"
    echo "      ajoute la clé publique au dépôt (Settings → Deploy keys, cocher 'Allow write access')."
    echo "   2. AGENT_FORWARD=1 dans labo.env (ta clé locale est utilisée via ssh -A)."
  fi

  info "Clonage du dépôt (si besoin)…"
  rsh "if [ -d $(q "$REMOTE_DIR/.git") ]; then echo '  dépôt déjà présent'; else git clone -q $(q "$DEPOT_GIT") $(q "$REMOTE_DIR") && echo '  cloné'; fi
       cd $(q "$REMOTE_DIR") && git checkout -q $(q "$BRANCHE")
       grep -qxF '.labo/' .git/info/exclude || echo '.labo/' >> .git/info/exclude
       grep -qxF '.venv/' .git/info/exclude || echo '.venv/' >> .git/info/exclude"
  installer_distant
  distant maj
  ok "Serveur prêt. Essaie : ./labo.sh run configs/pilote.yaml --gpus 0"
}

cmd_run() {
  local config="" gpus="$GPUS_DEFAUT" nom="" sync=1
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus) gpus="$2"; shift 2 ;;
      --nom) nom="$2"; shift 2 ;;
      --sans-sync) sync=0; shift ;;
      -*) err "Option inconnue : $1" ;;
      *) config="$1"; shift ;;
    esac
  done
  [[ -n "$config" ]] || err "Usage : ./labo.sh run <config> [--gpus 0|1|0,1] [--nom x] [--sans-sync]"
  [[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || err "--gpus attend 0, 1 ou 0,1"
  if (( sync )); then cmd_sync; else installer_distant; fi
  local id; id="$(distant lancer "$config" "$gpus" "$nom" | tail -n 1)"
  ok "Run lancé : $id (GPU $gpus)"
  echo "   suivre :  ./labo.sh logs $id -f"
  echo "   arrêter : ./labo.sh stop $id"
}

cmd_file() {
  local f="${1:-}"; [[ -f "$f" ]] || err "Usage : ./labo.sh file <fichier>  (lignes : gpu config [nom])"
  cmd_sync
  distant file < "$f"
}

cmd_recuperer() {
  local id="${1:-}" dest="$ICI/resultats_serveur"
  mkdir -p "$dest"
  info "Rapatriement dans $dest/ …"
  # rsync découpe la commande -e lui-même : guillemets simples, pas d'échappements à la printf %q
  local a e="ssh"
  for a in "${SSH_OPTS[@]}"; do e+=" '$a'"; done
  rsync -az --info=progress2 -e "$e" \
    "$SERVEUR:$REMOTE_DIR/results/${id:+$id}" "$dest/"
  ok "Terminé."
}

cmd="${1:-aide}"; shift || true
case "$cmd" in
  setup)     cmd_setup ;;
  sync)      cmd_sync ;;
  run)       cmd_run "$@" ;;
  file)      cmd_file "$@" ;;
  statut)    distant statut ;;
  logs)      [[ -n "${1:-}" ]] || err "Usage : ./labo.sh logs <run_id> [-f]"
             if [[ "${2:-}" == "-f" ]]; then ssh -t "${SSH_OPTS[@]}" "$SERVEUR" "bash $(q "$LABO_DISTANT/remote.sh") logs $(q "$1") -f"
             else distant logs "$1"; fi ;;
  stop)      [[ -n "${1:-}" ]] || err "Usage : ./labo.sh stop <run_id>"; distant stop "$1" ;;
  sauver)    distant sauver "${*:-sauvegarde manuelle}" ;;
  recuperer) cmd_recuperer "${1:-}" ;;
  shell)     ssh -t "${SSH_OPTS[@]}" "$SERVEUR" "cd $(q "$REMOTE_DIR") && exec \$SHELL -l" ;;
  aide|-h|--help) sed -n '2,/^# =====/p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) err "Commande inconnue : $cmd (./labo.sh aide)" ;;
esac
