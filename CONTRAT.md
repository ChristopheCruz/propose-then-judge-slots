# Contrat de développement

Ce document fixe ce que tous les contributeurs (humains ou agents) doivent respecter.
Le plan scientifique est dans `docs/plan_de_recherche.docx`.

## 1. Fichiers partagés (ne pas modifier sans accord du coordinateur)

| Fichier | Contenu |
|---|---|
| `laya_slots/schema.py` | Format unifié : `Example`, `SlotValue`, `Ontology`, `Candidate`, `JudgeRequest`, `Judgment` |
| `laya_slots/interfaces.py` | Protocoles : `DatasetLoader`, `CandidateGenerator`, `Judge` |
| `pyproject.toml` | Dépendances. Besoin d'une nouvelle dépendance → la signaler dans le rapport final, ne pas l'ajouter |
| `CONTRAT.md`, `README.md` | Ce document, la présentation |

Si le contrat ne convient pas, **ne pas le contourner** : implémenter au mieux et décrire le changement proposé dans le rapport.

## 2. Répartition des répertoires

| Responsable | Répertoires |
|---|---|
| Données | `laya_slots/data/`, `laya_slots/normalize/`, `tests/data/`, `tests/normalize/` |
| Candidats | `laya_slots/candidates/`, `tests/candidates/` |
| Juges | `laya_slots/judges/`, `tests/judges/` |
| Infra | `experiments/`, `configs/`, `scripts/`, `.github/`, `tests/experiments/` |
| Entraînement (vague 2) | `laya_slots/train/` |
| Décision / évaluation (vague 2) | `laya_slots/decision/`, `laya_slots/evaluation/` |

Chacun n'écrit que dans ses répertoires. Le dossier s'appelle `evaluation/` (et non `eval/`) pour ne pas masquer la fonction `eval` de Python.

## 3. Conventions

- **Valeurs** : toute valeur de slot (gold ou candidate) est normalisée par `laya_slots.normalize` avant comparaison. Une valeur absente vaut `NONE_VALUE` (`"__none__"`).
- **Niveau tour** : `Example.slots` contient les slots exprimés dans le tour courant, pas l'état cumulé.
- **Jeux convertis** : `data/processed/<dataset>/<lang>/{train,dev,test}.jsonl` + `ontology.json` + `MANIFEST.json` (sha256 de chaque fichier, nombre d'exemples, version source). `data/raw/` et `data/processed/` ne sont pas versionnés.
- **Runs** : chaque run écrit dans `$RUN_DIR` (fourni par `labo.sh`) : `config.yaml` résolu, `predictions.jsonl` (une ligne par `JudgeRequest` : `example_id`, `slot`, `candidates`, `choice_value`, `proba`, `gold_value`), `metrics.json`.
- **Métrique principale** : F1 micro des slots sur les paires (slot, valeur) par exemple, valeur prédite `NONE_VALUE` exclue des prédictions.
- **Graines** : toute source d'aléa prend une graine explicite issue de la config.
- **Tests** : `uv run pytest` doit passer **sur CPU, sans réseau et sans télécharger de modèle**. Les tests qui demandent un GPU, un modèle ou le réseau sont marqués `@pytest.mark.integration` (exclus par défaut).
- **Langue** : code et identifiants en anglais, commentaires et documentation en français.
- **Pas de fuite** : un juge ne lit jamais `example.slots` ; un générateur de candidats ne lit les gold que s'il s'appelle `oracle`.

## 4. Lancement

`labo.sh` lance pour chaque run :

```bash
uv run python -m experiments.run --config "$CONFIG" --output "$RUN_DIR"
```
