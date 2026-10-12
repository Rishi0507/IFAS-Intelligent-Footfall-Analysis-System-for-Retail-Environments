# IFAS paper

The research paper for this project, in IEEE conference format.

| File | What it is |
| :--- | :--- |
| `ifas.pdf` | The built paper, ten pages |
| `ifas.tex` | The source. Diagrams and the algorithm are written in this file |
| `refs.bib` | The thirteen references |
| `figures/make_figures.py` | Draws all six plots as vector PDFs |
| `figures/runtime.json` | Stage timings measured by `scripts/benchmark_stages.py` |

## Building

Upload this folder to Overleaf and compile `ifas.tex`. Nothing else is needed,
since `IEEEtran` ships with every TeX distribution.

To build on your own machine, [Tectonic](https://tectonic-typesetting.github.io)
is a single download and fetches the packages it needs on the first run:

```bash
cd paper
tectonic ifas.tex
```

A full TeX Live or MiKTeX install works too: `latexmk -pdf ifas.tex`.

## Where each number comes from

Nothing in the paper is estimated. Every plotted value sits in the `DATA` table
at the top of `figures/make_figures.py` with one of four source tags.

| Tag | Source | Used for |
| :--- | :--- | :--- |
| `doc` | `docs/hyperparameter-analysis/` | Most of the two sensitivity figures |
| `plot` | Read back from the original plot images in that folder | About a dozen points that are not written in the notes. Reading error is about 0.0003 on the accuracy axes |
| `notebook` | Stored outputs of `age_classifier_v3.ipynb` | Age data counts, the split, validation and test mA, precision, recall and the confusion matrix |
| `runtime` | `scripts/benchmark_stages.py` | The timing figure and the numbers in Section IV-F |

To redraw the plots:

```bash
python paper/figures/make_figures.py
```

To time the stages on another machine and refresh the runtime figure:

```bash
python scripts/benchmark_stages.py --video clip.mp4 \
    --gender-ckpt models/hf/abhshkp__footfall-analysis-vit-stage1 \
    --age-bundle models/age --out paper/figures/runtime.json
```

Then copy the new values into the six `\t...` macros near the top of `ifas.tex`.

## Things to confirm before submitting

Each of these needs a decision from the authors.

1. **Headline gender accuracy.** The paper quotes 97.8%, from the linear probe
   study. The learning rate study reached 98.3% with the same learning rate and
   one more epoch of fine-tuning. The text says why the two differ. Quote
   whichever run matches the final model.
2. **Gender training schedule.** Table I lists 4 probe epochs, weight decay 0.3
   and warmup 0.2, the values the study selected. The defaults in
   `src/training/trainer.py` are 3 epochs, 0.01 and 0.1. If the released
   checkpoint was trained with the defaults, say so in the table.
3. **No gender test result.** The gender notebooks have no stored outputs, so the
   paper has validation accuracy only and lists that as a limitation. A held-out
   test number would strengthen it.
4. **Two age configurations.** The test results come from the released model
   (5 regions, 8 LBP points, 550 features), because that is what the notebook
   evaluated. The sensitivity study selects 4 regions and 16 points. The paper
   reports both and says which is which. Retraining and testing the selected
   configuration would let you drop the distinction.
5. **Interval density labels.** The paragraph on five intervals and six density
   labels is carried over from Draft 4. There is no code for it in this
   repository, and the paper says it is not evaluated.
6. **Stage 2.** The paper describes the second fine-tuning stage as part of the
   design and reports no result for it.
7. **Person detector for the age model.** Draft 4 named Faster R-CNN, which is
   what the age notebook uses for its own video test. The paper describes the
   integrated pipeline, where both classifiers share the YOLOv8n detections.
8. **The pooling layer.** `google/vit-base-patch16-224` ships without a pooler,
   so the dense layer between the class token and the head starts from random
   weights. It belongs to the backbone in `src/training/model.py` and is
   therefore frozen during Phase A. The linear probe then trains on a random
   projection of the class token, which may be why the probe alone stops near
   0.79. The paper does not mention this. Moving the pooler into the head, or
   reading the class token directly, is worth a test.
9. **Authors.** The author block follows Draft 4: four authors, one affiliation,
   the project guide on its own line and no email addresses. Adjust it to the
   venue's rules.
10. **The field check.** Section IV-G and its two tables come from one 15 second
    clip downloaded from the web (`vidssave.com 4MP PTZ Camera...720p.mp4`). The
    count of five people and where each one is were judged by eye from enlarged
    frames. Watch the clip and confirm them, and check that you may use that
    footage in a paper. The tracking rules were tuned on this clip, which the
    paper says.
11. **Length.** The paper is ten pages. Many conferences cap at six or eight.
    The field check, the runtime section and the algorithm are the easiest
    parts to shorten or move to an appendix.
