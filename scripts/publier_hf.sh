#!/usr/bin/env bash
# =============================================================================
#  publier_hf.sh — publie le modèle choisi sur le dépôt Hugging Face privé (docs/modele_hf.md).
#
#  Tourne sur le poste local (connecté par `hf auth login`) : rapatrie le checkpoint depuis la machine
#  GPU dans un dossier de préparation hors Qsync, y ajoute la fiche et la calibration, vérifie le
#  contenu, puis envoie et pose l'étiquette de version.
#
#    scripts/publier_hf.sh <graine> [version] [--envoyer]
#    scripts/publier_hf.sh 14               # prépare et liste seulement (aucun envoi)
#    scripts/publier_hf.sh 14 v1.0 --envoyer
#
#  Fichiers publiés : model.safetensors, rl_agent_config.json, encoder/, tokenizer/ (format Laya),
#  calibration.json (si présent dans le checkpoint), training_config.yaml, train_metrics.json, README.md,
#  assets/logo-lamae.png.
#  Jamais : train.log, prédictions, données (licences des jeux ; MultiATIS++ absent par construction).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

GRAINE="${1:?graine du checkpoint (13, 14 ou 15)}"
VERSION="${2:-v1.0}"
ENVOYER="${3:-}"
DEPOT=ChristopheCruz/laya-slotjudge-m46
DISTANT=ubuntu@caire-gpu2:Workspace/laya-vs-llm-4slots/checkpoints/hf/multi-s$GRAINE
PREP="$HOME/hf-staging/laya-slotjudge-m46"
export RSYNC_RSH="ssh -i $HOME/.ssh/id_ed25519_caire -o IdentitiesOnly=yes -o BatchMode=yes"

hf auth whoami | grep -q "ChristopheCruz" || { echo "pas connecté à Hugging Face : hf auth login"; exit 2; }

rm -rf "$PREP" && mkdir -p "$PREP"
rsync -a --include 'model.safetensors' --include 'rl_agent_config.json' --include 'calibration.json' \
  --include 'train_metrics.json' --include 'config.yaml' --include 'encoder/***' --include 'tokenizer/***' \
  --exclude '*' "$DISTANT/" "$PREP/"
mv "$PREP/config.yaml" "$PREP/training_config.yaml"
# Température des questions à choix (temperature[0] du runtime Laya), ajustée sur le dev par scripts/hf/temperature.py.
rsync -a "${DISTANT%/checkpoints/*}/results/hf/temperature-s$GRAINE.json" "$PREP/calibration_fit.json" \
  || { echo "température absente : lancer scripts/hf/temperature.py $GRAINE sur la machine GPU"; exit 3; }
python3 - "$PREP" <<'PY'
import json, sys
from pathlib import Path
prep = Path(sys.argv[1])
t = json.loads((prep / "calibration_fit.json").read_text())["temperature_choice"]
cfg_path = prep / "rl_agent_config.json"
cfg = json.loads(cfg_path.read_text())
cfg["temperature"][0] = t
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
print(f"temperature[0] (choice) = {t}")
PY
mkdir -p "$PREP/assets" && cp site/assets/logo-lamae.png "$PREP/assets/"   # logo LAMAE de la fiche
# La fiche, sans les commentaires de brouillon de l'en-tête YAML.
grep -v -E '^# (Fiche de modèle|Les valeurs entre|Les commentaires de cet)' docs/hf/README.md > "$PREP/README.md"

# Contrôles : format Laya complet, pas de TODO dans la fiche avant la v1.0, pas de chemin local.
for f in model.safetensors rl_agent_config.json encoder/config.json tokenizer/tokenizer.json README.md; do
  [[ -f "$PREP/$f" ]] || { echo "manquant : $f"; exit 3; }
done
python3 -c "import json,sys; c=json.load(open(sys.argv[1])); assert c['laya_slots']['train_config']['seed']==int(sys.argv[2]), 'graine'" \
  "$PREP/rl_agent_config.json" "$GRAINE"
if grep -rlE '/home/|/mnt/c|C:\\' "$PREP" --include='*.json' --include='*.yaml' --include='*.md'; then
  echo "chemin local dans un fichier ci-dessus"; exit 3
fi
if grep -q 'TODO' "$PREP/README.md"; then
  echo "ATTENTION : la fiche contient encore des [TODO]"
  [[ $ENVOYER == --envoyer && $VERSION != *-draft ]] && { echo "refus d'étiqueter $VERSION avec des TODO (utiliser une version *-draft)"; exit 4; }
fi
( cd "$PREP" && find . -type f -printf '%10s  %p\n' | sort -k2 )

[[ $ENVOYER == --envoyer ]] || { echo "préparé dans $PREP (rien envoyé ; ajouter --envoyer)"; exit 0; }
hf upload "$DEPOT" "$PREP" . --private --commit-message "laya-slotjudge-m46 $VERSION (seed $GRAINE)"
hf repos tag create "$DEPOT" "$VERSION" -m "laya-slotjudge-m46 $VERSION (seed $GRAINE)"
echo "envoyé : https://huggingface.co/$DEPOT (étiquette $VERSION)"
