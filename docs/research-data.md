# Research data

Where each dataset in the research plan comes from, its terms, how it is split,
and how its labels map onto ours. Datasets live under `data/`, which git
ignores: none of them may be committed or redistributed.

## PETA (training source, RQ1 and RQ2)

Deng et al., ACM MM 2014. 19,000 images from ten surveillance datasets. Already
used by the training notebooks on Colab.

- **Split.** `src/research/peta.py` builds two splits in the standard
  9,500 / 1,900 / 7,600 proportions, written together to one CSV
  (`peta_split.csv`, kept on Drive and shared by every run):
  - `split_identity` (**primary**): every image of a person in one split, gender
    balanced. The test set holds only unseen people.
  - `split_image`: an image-level random split like the original notebooks use,
    in which people leak between train and test. Kept only to measure how much
    that leak inflates accuracy (one baseline run in `research_rq2_training.ipynb`).

  The released models were trained on image-level splits, and the stage-1 gender
  notebook held out only a 10 % validation set with **no test split**, which is
  why the paper has no gender test result. Retrain the age model on
  `split_identity` too, so gender and age are tested on the same people.

## MARS with MARS-Attribute labels (RQ1 and RQ3)

- **Images.** MARS, Zheng et al., ECCV 2016: 1,261 people, about 20,000
  tracklets cut by a detector (DPM) and a tracker (GMMCP) from six campus
  cameras. Research use only, distributed by its authors. Unzip `bbox_train`
  and `bbox_test` into `data/mars/`.
- **Labels.** `mars_attributes.csv` from
  [github.com/yuange250/MARS-Attribute](https://github.com/yuange250/MARS-Attribute),
  32 tracklet-level attributes in the Market-1501 definitions. The repository has
  no licence file; cite it and the MARS paper, and do not redistribute. The
  IRIP Beihang page that the original MARS attribute paper points to returned
  404 on 2026-10-18.
- **Size.** 16,360 labelled tracklets of 1,259 people. Gender code 0 / 1:
  9,589 / 6,771. Age young / teenager / adult / old: 104 / 13,161 / 2,837 / 258.
  Five people carry different labels on different tracklets.
- **Gender code (to confirm).** The source does not say which code is male.
  `src/research/mars.py` assumes 0 = male (59 %, as expected on a campus).
  Confirm with `python scripts/research/get_mars.py sheet` before reporting any
  MARS number, and record the check here.
- **Age.** Market-1501 counts college-age people as teenagers, so teenager
  (80 % of tracklets) roughly covers PETA's 16–30 group, adult covers 31–60 and
  old covers over 61. Only the old / not-old split is reliable enough to report.
- **Split.** MARS's own train / test split. The pooling study (A5) fits its
  quality model and attention baseline on train and reports on test.
- **Pipeline.**

  ```bash
  python scripts/research/get_mars.py labels
  python scripts/research/get_mars.py index     # data/mars/tracklets.jsonl
  python scripts/research/get_mars.py sheet     # data/mars/gender_check.jpg
  ```

## RAP (optional target domain, RQ1 and RQ2)

Li et al. Indoor surveillance in a shopping mall, with gender and age bands
(under 16, 17–30, 31–45, 46–60, over 60) that match PETA's almost exactly.
Access needs a signed research agreement from the authors. Not requested yet.

## Retail clips (RQ1 to RQ3)

Recorded by Person A (task A3) with written permission. Record the source,
consent, the test / FDA-target split and the labelling here once they exist.
The 97 s web clip used so far (`vidssave.com ... retail store 720p.mp4`) has no
confirmed licence and must not appear in the paper.
