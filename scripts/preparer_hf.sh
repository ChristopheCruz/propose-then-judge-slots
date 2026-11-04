#!/usr/bin/env bash
# =============================================================================
#  preparer_hf.sh — données du modèle publié (docs/modele_hf.md) : téléchargement et conversion.
#
#  MTOP (6 langues), Multi3WOZ (4 langues), MASSIVE (toutes les locales), SGD (anglais).
#  Réseau requis (sans HF_HUB_OFFLINE). Reprise : une langue déjà préparée (MANIFEST.json) est sautée.
#  Usage : scripts/preparer_hf.sh            Journal : results/preparer-hf.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
exec > >(tee -a results/preparer-hf.log) 2>&1

prep() {   # jeu langue [--download]
  if [[ -f data/processed/$1/$2/MANIFEST.json ]]; then echo "  déjà prêt : $1/$2"; return 0; fi
  echo "== $(date '+%T') $1/$2"
  $PY -m laya_slots.data.prepare --dataset "$1" --lang "$2" ${3:-} | tail -5 || echo "  ÉCHEC : $1/$2"
}

echo "==== $(date '+%F %T') préparation des données du modèle publié"
prep mtop en --download
for l in de es fr hi th; do prep mtop "$l"; done
prep multi3woz en --download
for l in fr ar tr; do prep multi3woz "$l"; done
prep sgd en --download
prep massive fr --download
for l in $($PY -c "from laya_slots.data import get_loader; print(' '.join(get_loader('massive').languages()))"); do
  prep massive "$l"
done
echo "==== $(date '+%F %T') fin ; langues prêtes :"
for d in mtop multi3woz massive sgd; do echo "  $d : $(ls data/processed/$d 2>/dev/null | tr '\n' ' ')"; done
