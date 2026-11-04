# Diagrams of the paper

| File | Figure | Content |
|---|---|---|
| `architecture.*` | Fig. 1 (Section 3.1) | the proposer–judge architecture on one request: input, proposers (real and oracle lists), options with “not specified”, the judge (Laya, or an LLM judge in C4), the calibrated three-zone decision; C0 as the tagger alternative |
| `decomposition.*` | Fig. 2 (Section 3.1) | error decomposition: every request ends in exactly one outcome (correct, recall error, choice error, wrong abstention, correct abstention, false positive) and the measures that isolate the proposers from the judge |
| `protocol.*` | Fig. 3 (Section 4) | experimental protocol: datasets and languages, oracle and real candidate lists, conditions C0–C4, families of measures, development/test discipline |

Each figure exists as SVG (source, here), PNG (2×, here, for the README) and PDF (vector, written to
`docs/article/figures/`, where the LaTeX article includes it).
The figures carry no numerical result, so they stay valid when the final test-set results replace the
provisional ones.

Regenerate (Pillow and Chromium; text is measured with the DejaVu fonts used for rendering, and an
overflowing box raises an error instead of being drawn):

```bash
python3 figures/schemas/make_figures.py   # SVG
bash figures/schemas/render.sh            # PDF (docs/article/figures/) and PNG
```
