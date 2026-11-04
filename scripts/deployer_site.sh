#!/usr/bin/env bash
# =============================================================================
#  deployer_site.sh — déploie le site de promotion (site/) dans le site personnel GitHub Pages.
#
#  À LANCER SEULEMENT APRÈS LA SOUMISSION (ordre : soumission à la revue, puis HAL hal-05850913 [-> arXiv], puis GitHub, Hugging Face et
#  ce site en public). Un site GitHub Pages est toujours public.
#
#  Cible : dépôt ChristopheCruz/mypage.github.io (sortie MkDocs, branche master), dossier slotjudge/
#    -> https://christophecruz.github.io/mypage.github.io/slotjudge/
#  Le site est régénéré (site/build.py), puis seules les pages et assets/ sont copiés (pas src/, build.py).
#
#    scripts/deployer_site.sh              # prépare et montre le diff, rien n'est poussé
#    scripts/deployer_site.sh --envoyer    # commit et push
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ENVOYER="${1:-}"
DEPOT=ChristopheCruz/mypage.github.io
DOSSIER=slotjudge
PREP="$HOME/site-staging/mypage.github.io"

if grep -rlq 'XXXX.XXXXX' site/src docs/hf docs/github; then
  echo "ATTENTION : identifiant arXiv encore en attente (XXXX.XXXXX) dans :"; grep -rl 'XXXX.XXXXX' site/src docs/hf docs/github
fi
curl -s -o /dev/null -f --max-time 20 https://hal.science/hal-05850913 || echo "ATTENTION : la notice HAL hal-05850913 n'est pas encore en ligne
for d in propose-then-judge-slots; do
  v=$(gh repo view "ChristopheCruz/$d" --json visibility -q .visibility)
  [[ $v == PUBLIC ]] || echo "ATTENTION : le dépôt $d est encore $v (les liens du site seront cassés)"
done
python3 -c "from huggingface_hub import HfApi; import sys; sys.exit(HfApi().model_info('ChristopheCruz/laya-slotjudge-m46').private)" \
  || echo "ATTENTION : le modèle Hugging Face est encore privé"

python3 site/build.py
if [[ -d $PREP/.git ]]; then
  git -c credential.helper= -c credential.helper='!gh auth git-credential' -C "$PREP" pull -q --ff-only
else
  mkdir -p "$(dirname "$PREP")" && git -c credential.helper= -c credential.helper='!gh auth git-credential' clone -q "https://github.com/$DEPOT.git" "$PREP"
fi
rm -rf "${PREP:?}/$DOSSIER" && mkdir -p "$PREP/$DOSSIER"
rsync -a --exclude src --exclude build.py --exclude __pycache__ site/ "$PREP/$DOSSIER/"
git -C "$PREP" add -A "$DOSSIER"
git -C "$PREP" status --short | head -40

[[ $ENVOYER == --envoyer ]] || { echo "préparé dans $PREP (rien poussé ; ajouter --envoyer)"; exit 0; }
git -C "$PREP" commit -q -m "laya-slotjudge-m46: promotion site (6 languages)"
git -c credential.helper= -c credential.helper='!gh auth git-credential' -C "$PREP" push -q origin HEAD
echo "déployé : https://christophecruz.github.io/mypage.github.io/$DOSSIER/ (GitHub Pages met 1 à 2 minutes)"
