#!/usr/bin/env bash
# =============================================================================
#  test_final.sh — runs finaux sur le TEST, selon docs/protocole_test.md (31/10/2026).
#
#  Parties :
#    L        grille Laya et C0 : C0 (réels), C1, C2, C3 v2 (oracle grille et réels), 6 couples
#    A        analyses MultiATIS++ EN : balayages k x p_absent, C3 v1, ablation hard ; transfert EN->FR
#             et EN+FR (checkpoints bilingues existants)
#    D        runs dev complémentaires de C1 sur MultiATIS++ FR (calibration, tableau dev)
#    c4-dev | c4-alt2 | c4   un LLM-juge (Qwen3.8-27B | Gemma-4-31B-it | Qwen3.8-Flash-Next, 2 cartes) :
#             dev MultiATIS++ FR (graine 13), puis grille test, 6 couples x oracle/réels x graines 13-15
#
#  Usage (sur la machine GPU, à la racine du dépôt, dans tmux) :
#    scripts/test_final.sh <cartes> <partie>
#    scripts/test_final.sh 1 L ; scripts/test_final.sh 0 c4-dev ; scripts/test_final.sh 0,1 c4
#  Les runs finaux vont dans results/final-* ; reprise : un run complet déjà là est sauté.
#  Garde-fou : refuse de tourner si le code n'a pas le code_sha256 du protocole.
#  Journal : results/test-final-<partie>.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

CARTES="${1:?cartes (0, 1 ou 0,1)}"
PARTIE="${2:?partie : L, A, D, c4-dev, c4-alt2 ou c4}"
SHA_PROTOCOLE=d80bc7ce063551bc01533296775fdada56634d8b6a53360eb604e4941ba46856
JEUX="multiatis mtop multi3woz"
LANGUES="en fr"
G3="13 14 15"
G5="13 14 15 16 17"

sha=$(.venv/bin/python -c "from experiments.run import code_sha256; print(code_sha256())")
[[ $sha == "$SHA_PROTOCOLE" ]] || { echo "code_sha256 $sha différent du protocole ($SHA_PROTOCOLE)"; exit 3; }

fait() {   # préfixe de run (sans horodatage) -> vrai si un run complet existe
  local m
  for m in results/$1-*/metrics.json; do
    [[ -f "$m" ]] && grep -q '"status": "complete"' "$m" && return 0
  done
  return 1
}

run() {   # nom cible [variables make...] : un run, sauté s'il existe déjà
  local nom=$1 cible=$2; shift 2
  if fait "$nom"; then echo "  déjà fait : $nom"; return 0; fi
  echo "== $(date '+%F %T') $nom"
  make --no-print-directory "$cible" GPUS="$CARTES" SORTIE="results/$nom-$(date +%Y%m%d-%H%M%S)" "$@" \
    || echo "  ÉCHEC : $nom"
}

grille() {   # cond graines : oracle (grille) et réels pour les 6 couples, test
  local cond=$1 graines=$2 jeu langue s
  for jeu in $JEUX; do for langue in $LANGUES; do for s in $graines; do
    run "final-$cond-$jeu-$langue-s$s" "$cond" DATASET="$jeu" LANGUE="$langue" SEED="$s"
    run "final-$cond-reels-$jeu-$langue-s$s" "$cond" DATASET="$jeu" LANGUE="$langue" SEED="$s" CAND=reels
  done; done; done
}

balayage() {   # cond graine : k x p_absent, MultiATIS++ EN, test
  local cond=$1 s=$2 k p
  for k in 2 5 10 20; do for p in 0.0 0.2 0.5; do
    run "final-$cond-multiatis-en-s$s-k$k-p$p" "$cond" DATASET=multiatis LANGUE=en SEED="$s" \
      ARGS="candidates.generators.0.params.k=$k candidates.generators.0.params.p_absent=$p judge.params.max_options=20"
  done; done
}

exec > >(tee -a "results/test-final-$PARTIE.log") 2>&1
echo "==== $(date '+%F %T') test final, partie $PARTIE, cartes $CARTES, code $sha"

case "$PARTIE" in
  L)
    for jeu in $JEUX; do for langue in $LANGUES; do for s in $G3; do
      run "final-c0-$jeu-$langue-s$s" c0 DATASET="$jeu" LANGUE="$langue" SEED="$s"
    done; done; done
    grille c2 "$G3"
    grille c3v2 "$G5"
    grille c1 "$G3"
    ;;
  A)
    balayage c1 13
    for s in $G3; do balayage c2 "$s"; done
    for s in $G5; do balayage c3v2 "$s"; done
    for s in $G5; do
      balayage c3 "$s"
      run "final-c3-reels-multiatis-en-s$s" c3 DATASET=multiatis LANGUE=en SEED="$s" CAND=reels
    done
    run final-c1-multiatis-en-s13-hard c1 DATASET=multiatis LANGUE=en SEED=13 DISTR=hard
    for s in $G3; do
      run "final-c2-multiatis-en-s$s-hard" c2 DATASET=multiatis LANGUE=en SEED="$s" DISTR=hard
      run "final-c3v2-multiatis-en-s$s-hard" c3v2 DATASET=multiatis LANGUE=en SEED="$s" DISTR=hard
    done
    for jeu in $JEUX; do
      for cond in c2 c3v2; do
        graines=$G3; [[ $cond == c3v2 ]] && graines=$G5
        for s in $graines; do
          # EN -> FR sans données françaises
          run "final-$cond-$jeu-fr-entr-en-s$s" "$cond" DATASET="$jeu" LANGUE=fr ENTR=en SEED="$s"
          run "final-$cond-reels-$jeu-fr-entr-en-s$s" "$cond" DATASET="$jeu" LANGUE=fr ENTR=en SEED="$s" CAND=reels
          # EN+FR -> EN et -> FR, si le checkpoint bilingue existe
          [[ -f checkpoints/$cond/$jeu-en+fr-s$s/model.safetensors ]] || continue
          for langue in $LANGUES; do
            run "final-$cond-$jeu-$langue-entr-en+fr-s$s" "$cond" DATASET="$jeu" LANGUE="$langue" ENTR=en+fr SEED="$s"
            run "final-$cond-reels-$jeu-$langue-entr-en+fr-s$s" "$cond" DATASET="$jeu" LANGUE="$langue" ENTR=en+fr SEED="$s" CAND=reels
          done
        done
      done
    done
    ;;
  D)
    run c1-dev-multiatis-fr-s13 c1 DATASET=multiatis LANGUE=fr SEED=13 SPLIT=dev
    run c1-reels-dev-multiatis-fr-s13 c1 DATASET=multiatis LANGUE=fr SEED=13 SPLIT=dev CAND=reels
    ;;
  c4-dev|c4-alt2|c4)
    run "$PARTIE-dev-multiatis-fr-s13" "$PARTIE" DATASET=multiatis LANGUE=fr SEED=13 SPLIT=dev
    run "$PARTIE-reels-dev-multiatis-fr-s13" "$PARTIE" DATASET=multiatis LANGUE=fr SEED=13 SPLIT=dev CAND=reels
    grille "$PARTIE" "$G3"
    ;;
  *) echo "partie inconnue : $PARTIE"; exit 2 ;;
esac
echo "==== $(date '+%F %T') fin de la partie $PARTIE"
