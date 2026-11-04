---
# Fiche de modèle Hugging Face (3/11/2026). Nom : laya-slotjudge-m46 ; version v1.0 = étiquette git du dépôt HF.
# Les commentaires de cet en-tête sont retirés par scripts/publier_hf.sh.
license: cc-by-sa-4.0
base_model: convaiinnovations/laya
library_name: laya
pipeline_tag: text-classification
language:
  - multilingual
  - af
  - ar
  - az
  - bn
  - ca
  - da
  - de
  - el
  - en
  - es
  - fa
  - fi
  - fr
  - he
  - hi
  - hu
  - hy
  - id
  - it
  - ja
  - jv
  - ka
  - kn
  - ko
  - lv
  - ml
  - mn
  - ms
  - my
  - nb
  - nl
  - pl
  - pt
  - ro
  - ru
  - sl
  - sq
  - sv
  - ta
  - te
  - th
  - tl
  - tr
  - ur
  - vi
  - zh
datasets:
  - AmazonScience/massive
tags:
  - slot-filling
  - dialogue-state-tracking
  - task-oriented-dialogue
  - multiple-choice
  - calibration
  - selective-prediction
  - laya
---

# laya-slotjudge-m46 — a multilingual slot judge (46 languages)

A small, calibrated **slot judge** for task-oriented dialogue. Given the user's last message, the dialogue
context and a short list of **candidate values** for one slot, it returns a probability for each candidate
**and for "not specified"**, in a single forward pass of a ~322 M-parameter encoder.

It is the second half of a *propose, then judge* pipeline: cheap proposers (a lexicon, a NER model, a
catalogue, the dialogue history, an LLM…) suggest values; the judge chooses one or abstains, and its
probability is calibrated, so that a threshold can decide whether to **act**, **ignore** or **ask the user**
(refit the threshold, and ideally the temperature, on your own data).

This model is a fine-tuned version of the `multilingual` checkpoint of
[Laya](https://huggingface.co/convaiinnovations/laya) (mmBERT-base encoder with a decision head), trained on
MTOP, Multi3WOZ, MASSIVE and SGD in many languages. It accompanies the paper *Propose, then judge: a small
calibrated decision model versus large language model judges for multilingual slot filling* (Cruz, Ghanem
and Cherifi, preprint [hal-05850913](https://hal.science/hal-05850913)); code:
[ChristopheCruz/propose-then-judge-slots](https://github.com/ChristopheCruz/propose-then-judge-slots).

- **Developed by:** Christophe Cruz, Hussam Ghanem, Hocine Cherifi — ICB, UMR 6303 CNRS, Université Bourgogne Europe, Dijon, France
- **Model type:** multiple-choice decision model (bidirectional encoder + option-scoring head)
- **Fine-tuned from:** `convaiinnovations/laya`, checkpoint `multilingual`, revision `7b928d82`
- **Languages:** 46 languages seen in training (list above); see [Languages](#languages)
- **Licence:** CC BY-SA 4.0 (share-alike, inherited from MTOP and SGD)
- **Version:** v1.0 (git tag of this repository; pin it with `revision="v1.0"`)
- **Funding:** French National Research Agency (ANR), LabCom LAMAE (ANR-24-LCV1-0008) and CAIRE (ANR-23-CMAS-0031, computing infrastructure)

## What it does

For each slot of the current turn, the judge receives one question:

```text
STATE
  last_user_message     : "a flight from Lyon to Paris on Thursday"
  intent                : find_flight
  slots_already_filled  : (known values, in dialogue)
  history               : (previous turns, in dialogue)

QUESTION
  Which value does the user's last message give for the slot "from_city" (departure city)?
  A: Lyon
  B: Paris
  C: Thursday
  D: not specified: the last message gives no value for this slot
```

and answers with a distribution such as `A 0.97 · B 0.01 · C 0.01 · D 0.01`. The "not specified" option is
always the last one: most slots of a schema are **not** mentioned in a given turn, and the judge is trained
to say so. All the questions of a turn share one forward pass.

The model does **not** extract spans or generate text: the values must come from somewhere else.

## Intended use

- Filling or updating slots in a task-oriented assistant when candidate values are available (database
  entries, catalogue items, NER or regex matches, values from earlier turns, LLM suggestions).
- Verifying the output of a generative extractor: pass the LLM's value and its alternatives as candidates
  and use the judge's probability to accept, reject or ask.
- Selective prediction: thanks to calibration, a threshold on the probability trades coverage for risk.

**Out of scope**: open-ended extraction without candidates; long documents (the input is truncated to 1,024
tokens, each option to 48 tokens); more than 15 candidates per question without splitting them (see the
`tournament` mode of the code); safety-critical decisions without human review.

## How to use

Install the Laya runtime (`pip install "laya>=0.3,<0.4"`) and download this repository:

```python
from huggingface_hub import snapshot_download
from laya import Router

path = snapshot_download("ChristopheCruz/laya-slotjudge-m46", revision="v1.0")
router = Router(models={"multilingual": path})   # the fine-tuned weights replace the published checkpoint

state = {
    "last_user_message": "un vol de Lyon à Paris jeudi",
    "intent": "find_flight",
}
options = ["Lyon", "Paris", "jeudi"]
criteria = {chr(65 + i): v for i, v in enumerate(options)}
criteria[chr(65 + len(options))] = "not specified: the last message gives no value for this slot"

question = {
    "type": "choice",
    "instructions": 'Which value does the user\'s last message give for the slot "from_city" (departure city)?',
    "criteria": criteria,
}
[result] = router.predict_batch([{"state": state, "questions": {"from_city": question},
                                  "model": "multilingual"}])
print(result["answers"]["from_city"]["probabilities"])
# {'A': 0.9797, 'B': 0.0104, 'C': 0.0018, 'D': 0.0081}   -> Lyon
```

Keep the exact wording of the instructions and of the "not specified" option: the model was trained with
these templates (in English, whatever the language of the dialogue). Keys are neutral letters; **shuffle the
candidates** before building the question, because Laya, like LLM judges, has a position bias. The class
`laya_slots.judges.laya_judge.LayaJudge` of the code repository builds the questions, shuffles the options,
groups the slots of a turn into one call and reads the answers back; use it with `model_path=path`.

**Decision.** Take the option with the highest probability `p`. In the paper we act when `p ≥ τ`, ask the
user when the probability is intermediate, and ignore otherwise; τ is chosen on development data for a
target risk. Indicative thresholds for **filling a slot** (the judge proposes a value), fitted on the
multilingual development set with the shipped temperature; risk = share of wrong fills among those performed:

| Target risk | Oracle lists: τ | correct fills kept | Real lists: τ | correct fills kept |
|---|---|---|---|---|
| 1 % | 0.967 | 60 % | 0.991 | 3 % |
| 5 % | 0.909 | 92 % | 0.986 | 26 % |
| 10 % | any (risk 9.5 % at τ = 0) | 100 % | 0.976 | 55 % |

With the real proposers of the paper, 34 % of the values the judge proposes are wrong before thresholding, and
only high thresholds keep the risk low. These thresholds depend on your proposers and data: **refit them on
your own development set**.

## Files

| File | Content |
|---|---|
| `model.safetensors` | weights of the encoder and the decision head (Laya format) |
| `rl_agent_config.json` | Laya configuration (encoder `jhu-clsp/mmBERT-base`, input lengths, temperatures) and training metadata |
| `encoder/`, `tokenizer/` | encoder configuration and tokenizer |
| `training_config.yaml`, `train_metrics.json` | full training configuration and per-epoch metrics |
| `calibration_fit.json` | the fitted temperature and the data it was fitted on |

## Training

**Data.** Questions are built by the same code as at inference, one per (turn, slot), with the gold value
among shuffled candidates, or "not specified" as the target when the slot is not expressed (two unexpressed
questions kept per expressed one).

| Dataset | Languages | Examples per language (cap) | Licence |
|---|---|---|---|
| MTOP v2 (Li et al., 2021) | en, de, es, fr, hi, th | 8,000 | CC BY-SA 4.0 |
| Multi3WOZ (Hu et al., 2023) | en, fr, ar, tr | 8,000 turns | MIT |
| MASSIVE 1.1 (FitzGerald et al., 2023) | 47 locales (sw, cy, is, km, am held out) | 1,500 | CC BY 4.0 |
| Schema-Guided Dialogue (Rastogi et al., 2020) | en | 15,000 turns | CC BY-SA 4.0 |

Caps per (dataset, language) are drawn with the seed, so that English and MASSIVE do not dominate.
MultiATIS++, used in the paper, is **not** part of the training data (LDC licence).

**Candidate lists.** Recipe C3 v2 of the paper: list sizes vary (2 to 20, short lists over-represented), the
gold value is removed from 0, 20 or 50 % of the lists (the target is then "not specified"), and values of
**other slots of the same utterance** are added as distractors, which teaches the judge not to confuse slots.

**Objective and hyper-parameters.** Log score (cross-entropy on the correct option), 3 epochs, AdamW, learning
rate 2e-5 (encoder) and 1e-4 (head), weight decay 0.01, 5 % warm-up, micro-batch 32, bf16, gradient
checkpointing, one H200 GPU, about 1.1 hours of training (3 epochs over 604,175 questions) after about 1 hour of data
preparation. The checkpoint is selected on slot F1 over a multilingual
development set that contains both oracle lists and **real** lists from the proposers of the paper. Three
seeds (13, 14, 15) were trained (selection F1 0.857, 0.866, 0.857); **seed 14**, the best on development
data, is published (epoch 2). The test sets were evaluated once, for this seed only, after this choice.

**Calibration.** A single temperature **T = 1.65** for multiple-choice questions, fitted by negative
log-likelihood on 458,412 development questions (oracle and real lists), is written in `rl_agent_config.json`
(`temperature[0]`) and applied automatically by the Laya runtime. On these development data the expected
calibration error falls from 0.039 to 0.017 (in-sample). With real candidate lists on MTOP and MASSIVE the
judge remains over-confident (ECE ≈ 0.15–0.17): one temperature cannot fit every list distribution, so refit it
on your data (`laya.calibrate`).

## Evaluation

Test sets, seed 14, evaluated once. Slot F1 (micro), mean over the languages of each group; *oracle* lists
contain the gold value among distractors (k = 5), *real* lists come from the proposers of the paper (lexicon,
GLiNER, ontology, history; up to 20 candidates). Gold intents (MTOP, MASSIVE, SGD) or domains (Multi3WOZ) are
given.

| Dataset | Languages | Oracle lists | Real lists | Candidate recall (real) |
|---|---|---|---|---|
| MTOP | 6 | 0.953 | 0.488 | 0.578 |
| Multi3WOZ | 4 | 0.912 | 0.537 | 0.536 |
| MASSIVE, seen locales | 47 | 0.904 | 0.524 | 0.623 |
| MASSIVE, held-out locales (sw, cy, is, km, am) | 5 | 0.822 | 0.466 | 0.627 |
| SGD (15 test services unseen in training) | en | 0.848 | 0.515 | 0.526 |

Per language (test, slot F1, oracle / real lists):

| | en | fr | de | es | hi | th | ar | tr |
|---|---|---|---|---|---|---|---|---|
| MTOP | 0.960 / 0.632 | 0.959 / 0.550 | 0.942 / 0.573 | 0.964 / 0.505 | 0.949 / 0.495 | 0.943 / 0.173 | | |
| Multi3WOZ | 0.938 / 0.761 | 0.910 / 0.516 | | | | | 0.886 / 0.414 | 0.913 / 0.456 |
| MASSIVE | 0.915 / 0.664 | 0.894 / 0.584 | | | | | | |

Held-out MASSIVE locales (oracle / real): sw 0.810 / 0.483, cy 0.816 / 0.489, is 0.873 / 0.562,
km 0.834 / 0.288, am 0.779 / 0.508.

**Comparison with the paper's monolingual judges** (same test sets, same candidate lists; paper: mean over
3 seeds for C2 and 5 for C3):

| Pair | Oracle: C2 / C3 / this model | Real lists: C2 / C3 / this model |
|---|---|---|
| MTOP en | 0.953 / 0.951 / **0.960** | 0.690 / **0.695** / 0.632 |
| MTOP fr | 0.951 / 0.942 / **0.959** | 0.625 / **0.629** / 0.550 |
| Multi3WOZ en | 0.940 / **0.946** / 0.938 | **0.853** / 0.852 / 0.761 |
| Multi3WOZ fr | 0.917 / **0.919** / 0.910 | **0.592** / 0.588 / 0.516 |

With oracle lists, one multilingual model matches the specialised monolingual ones. **With real lists it is
0.06 to 0.09 F1 below them**: when the right value is proposed, it answers "not specified" more often (about
2 to 3 times more than the monolingual judges on development data). A variant trained with more MTOP and
Multi3WOZ data did not close the gap; the cause is not established. If you work on one of these datasets and
languages with similar proposers, a monolingual judge trained with the code repository will do better.

For reference, the monolingual models of the paper (same recipe, one dataset and one language at a time)
reach a slot F1 of 0.92–0.95 with oracle lists on the test sets of MultiATIS++, MTOP and Multi3WOZ in English
and French, against 0.38–0.84 for the best of three LLM judges (27 B to 180 B parameters); with oracle lists
they fill 0.4–0.5 % of the slots the user never mentioned on MultiATIS++, against 15–21 % for the LLMs.
These figures are those of the paper, not of this model.

## Languages

Seen in training: Afrikaans, Arabic, Azerbaijani, Bengali, Catalan, Chinese (simplified and traditional),
Danish, Dutch, English, Finnish, French, Georgian, German, Greek, Hebrew, Hindi, Hungarian, Indonesian,
Italian, Japanese, Javanese, Kannada, Korean, Latvian, Malay, Malayalam, Mongolian, Burmese, Norwegian Bokmål,
Persian, Polish, Portuguese, Romanian, Russian, Slovenian, Albanian, Spanish, Swedish, Tagalog, Tamil, Telugu,
Thai, Turkish, Urdu, Vietnamese, Armenian. Held out for evaluation: Swahili, Welsh, Icelandic, Khmer, Amharic.

The questions (instructions, slot descriptions, "not specified") are in English; the dialogue can be in any
of these languages. Most languages are covered by MASSIVE only (single-turn, virtual-assistant requests);
multi-turn dialogue is covered in English, French, Arabic and Turkish.

## Limitations and risks

- **The judge is only as good as the candidates.** If the right value is not proposed, the best answer is
  "not specified"; recall of the proposers bounds the pipeline.
- **Real candidate lists.** With noisy real lists, this multilingual model is 0.06–0.09 F1 below the
  paper's monolingual judges on MTOP and Multi3WOZ (see Evaluation): it abstains too often when the right
  value is proposed, and its confidence is too high on such lists.
- **Unsegmented scripts.** With the paper's real proposers (word n-gram lexicon, GLiNER), candidate recall is
  very low for Thai (MTOP), Khmer, Japanese, Chinese and Burmese, and so is F1 (MTOP Thai 0.17). This is a
  limit of those proposers, not of the judge (oracle F1 0.94 for MTOP Thai), but plan proposers adapted to
  these scripts.
- **Domain shift.** Training covers virtual-assistant commands, travel, restaurants, hotels, events and the
  SGD services. On new domains, use clear slot descriptions and fit the temperature and threshold on your own
  development data.
- **Position bias** remains: on hard or ambiguous questions the first option can attract probability. For
  example, for « un vol de Lyon à Paris jeudi » and the slot `to_city` described as "arrival city", the
  model answers *Lyon* in 2 of the 6 orders of the options, but *Paris* in all 6 with the description
  "destination city". Shuffle the options, write explicit slot descriptions, and, when it matters, average
  over circular rotations of the options (`rotations` in `LayaJudge`; they share the same forward pass).
- **Calibration** is fitted on the development data of the training datasets; it does not transfer exactly
  to other distributions.
- Translations in MASSIVE, MTOP and Multi3WOZ were produced from English sources and carry their cultural and
  lexical biases; low-resource languages have fewer and shorter examples.

## Licence

The weights are released under **CC BY-SA 4.0**: they are derived from MTOP and SGD (CC BY-SA 4.0), MASSIVE
(CC BY 4.0), Multi3WOZ (MIT) and Laya (Apache-2.0). If you distribute a model derived from this one, keep the
attribution and the same licence.

## Funding

<a href="https://christophecruz.github.io/mypage.github.io/projets/lamae/"><img src="https://huggingface.co/ChristopheCruz/laya-slotjudge-m46/resolve/main/assets/logo-lamae.png" alt="LabCom LAMAE" width="72" align="left"></a>

This work was supported by the French National Research Agency (ANR) through the LabCom LAMAE project
[grant number ANR-24-LCV1-0008] and the CAIRE project [grant number ANR-23-CMAS-0031], which funded the
computing infrastructure (a server with two NVIDIA H200 NVL GPUs) on which the model was trained.
<br clear="left">

## Citation

```bibtex
@unpublished{cruz2026proposejudge,
  author      = {Cruz, Christophe and Ghanem, Hussam and Cherifi, Hocine},
  title       = {Propose, then judge: a small calibrated decision model versus large language model
                 judges for multilingual slot filling},
  year        = {2026},
  note        = {Preprint},
  url         = {https://hal.science/hal-05850913},
  hal_id      = {hal-05850913},
  hal_version = {v1}
}
```

Please also cite Laya and the training datasets (MTOP, Multi3WOZ, MASSIVE, SGD).

Contact: Christophe Cruz (christophe.cruz@ube.fr).
