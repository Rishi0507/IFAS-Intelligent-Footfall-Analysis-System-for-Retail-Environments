# IFAS paper

The research paper for this project, in IEEE conference format.

| File | What it is |
| :--- | :--- |
| `ifas.pdf` | The built paper, ten pages |
| `ifas.tex` | The source. Diagrams and the algorithm are written in this file |
| `refs.bib` | The references, including the ones the research sections cite |
| `figures/make_figures.py` | Draws all six plots as vector PDFs, plus the RQ figures and `figures/rq3_numbers.tex` once their results exist |
| `figures/runtime.json` | Stage timings measured by `scripts/benchmark_stages.py` |
| `results/` | Research results written by `scripts/research/` (docs/research-plan.md); `make_figures.py` reads them under the `experiment` tag |
| `sections/` | Person A's sections of the research rewrite (task S3): System, RQ3, Ethics, Limitations |
| `draft_personA.tex`, `.pdf` | Builds those sections on their own. Every RQ3 number comes from `figures/rq3_numbers.tex` and prints as **??** until `rq3_pooling.json` exists |

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
| `experiment` | `paper/results/*.json`, written by `scripts/research/` | The RQ1-RQ3 figures, drawn only once their results files exist |

To redraw the plots:

```bash
python paper/figures/make_figures.py
```

To time the stages on another machine and refresh the runtime figure:

```bash
python scripts/benchmark_stages.py --video clip.mp4 \
    --gender-ckpt models/hf/abhshkp__footfall-analysis-vit-stage1 \
    --age-bundle abhshkp/footfall-analysis-age-svm --out paper/figures/runtime.json
```

Then copy the new values into the six `\t...` macros near the top of `ifas.tex`.

## Things to confirm before submitting

Each of these needs a decision from the authors.

0. **Check this first: which PETA images the gender model trained on.** The
   paper says "about 19,000". Two pieces of evidence say otherwise:
   * The `organize_peta` cell that `train_stage1.ipynb` had until now matched each
     `Label.txt` line only to a file named `<pid>` or `<pid>.<ext>`. On this
     project's `PETA.zip` that keeps CUHK alone (4,563 images). So the published
     checkpoint was not made by that cell as it stands.
   * The published Stage 1 checkpoint scores **100% on 200 images from the nine
     non-CUHK subsets and 92% (balanced 89%) on 200 CUHK images**
     (`results/stage1_training_data_check.json`). A fine-tuned ViT is close to
     perfect on its own training images, so it was trained on the nine subsets
     (about 14,400 images, the same ones the age notebook loads) and **not on
     CUHK**.

   Confirm with the Colab output line `[peta] wrote N images` or `n_total` in
   the Stage 1 `history.json` on Drive (about 1,440 means the nine subsets). If
   confirmed, change Section IV-A to "the 14,436 images of the nine subsets
   other than CUHK", and CUHK becomes a held-out test that neither model has
   seen: `results/rq1_gap.json` has both models on all 4,563 CUHK images. The
   notebook cell is fixed now and loads all 18,999 gender-labelled images;
   `run_rq2.py` uses `make_peta_split.py`, which was right from the start.

1. **Headline gender accuracy.** The paper quotes 97.8%, the best run of the
   linear probe study (4 probe + 3 fine-tune epochs), and now says it is not a
   run of the Table I schedule and was picked on the validation set it is
   reported on. The learning rate study reached 98.3% with one more epoch of
   fine-tuning. Quote whichever run matches the final model.
2. **Gender training schedule.** Table I lists 4 probe epochs, weight decay 0.3
   and warmup 0.2, the values the study selected. `TrainConfig` and both
   training notebooks now use these too (ported from the team's commit
   35c900f). Still confirm which settings the released Hub checkpoint was
   trained with; if it predates that change, say so in the table.
3. **No gender test result.** The gender notebooks have no stored outputs, so the
   paper has validation accuracy only and lists that as a limitation.
   `scripts/research/run_rq2.py` produces a held-out result on the
   identity-disjoint split as part of RQ2; it needs a GPU (task B6).
4. **Two age configurations.** The test results come from the released model
   (5 regions, 8 LBP points, 550 features), because that is what the notebook
   evaluated. The sensitivity study selects 4 regions and 16 points. The paper
   reports both and says which is which. Retraining and testing the selected
   configuration would let you drop the distinction.
5. **Interval density labels.** The paragraph on five intervals and six density
   labels (carried over from Draft 4) has been removed, because there is no
   code for it in this repository. Put it back only if it gets implemented.
6. **Stage 2.** The paper describes the second fine-tuning stage as part of the
   design and says it is not evaluated. RQ2 condition 4 in `run_rq2.py`
   evaluates it once labelled store crops exist.
7. **Person detector for the age model.** Draft 4 named Faster R-CNN, which is
   what the age notebook uses for its own video test. The paper describes the
   integrated pipeline, where both classifiers share the YOLOv8n detections.
8. **The pooling layer.** `google/vit-base-patch16-224` ships without a pooler,
   so the dense layer between the class token and the head starts from random
   weights. It belongs to the backbone in `src/training/model.py` and is
   therefore frozen during Phase A. The linear probe then trains on a random
   projection of the class token, which may be why the probe alone stops near
   0.79. The paper now says this in Section III-D. Moving the pooler into the
   head, or reading the class token directly, is worth a test.
9. **Authors.** The author block follows Draft 4: four authors, one affiliation,
   the project guide on its own line and no email addresses. Adjust it to the
   venue's rules.
10. **The field check.** Section IV-G and its two tables come from one 15 second
    clip downloaded from the web (`vidssave.com 4MP PTZ Camera...720p.mp4`). The
    count of five people and where each one is were judged by eye from enlarged
    frames. Watch the clip and confirm them, and check that you may use that
    footage in a paper. The tracking rules were tuned on this clip, which the
    paper says. The gender numbers here were re-run with the strict loader fix
    and came out identical (`results/b1_rebaseline.json`), so the old loader bug
    did not affect them.
11. **Identity overlap in the splits.** Both PETA splits are random at the image
    level, and PETA has 19,000 images of only 8,705 people, so the same person
    can sit in train and in validation or test. The paper now says so and cites
    Jia et al. (arXiv:2107.03576). `scripts/research/make_peta_split.py` builds
    an identity-disjoint split; rerunning on it would give numbers that hold for
    new customers.
12. **Length.** The paper is ten pages. Many conferences cap at six or eight.
    The field check, the runtime section and the algorithm are the easiest
    parts to shorten or move to an appendix.
13. **Field-check gender sentence.** The last paragraph of Section IV-G ("Gender
    was stable for the three people who are clearly visible ... confidence of
    0.56") came from the pipeline before the gender loader fix. Whether the old
    loader dropped the fine-tuned encoder depends on the installed transformers
    version. On the 97 s retail clip it did: the fix moved 15 of 46 people to a
    different label and the share below 0.7 confidence from 22 to 6
    (`results/b1_gender_rebaseline.json`). The 15 s field-check clip was made on
    a machine with transformers 5.13.0, whose weight names match the
    checkpoint, and re-running it with the strict loader gave identical results
    for every person (`results/b1_rebaseline.json`, item 10). So the sentence
    stands as written. The detection and tracking numbers in Section IV-G do
    not depend on the gender model.
