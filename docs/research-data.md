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

**Measurements and what the released models saw.** `scripts/research/make_peta_split.py` builds the same kind of identity split and produced the numbers below and `paper/results/rq1_gap.json`.

The paper's numbers came from **image-level** random splits, which put many people on both sides
(Jia et al., arXiv:2107.03576). Every new result uses the identity-disjoint split: whole people go
to train, val or test in the PETA protocol's 50 / 10 / 40 proportions by image count, stratified by
gender. The script writes `<out>.summary.json` with `people_in_train_and_test`, which must be 0.

To measure what the overlap was worth, evaluate the same model on `--mode image` and `--mode
identity` test sets and report both. Measured on this release: in an image-level split with the
protocol's proportions, 74 to 75% of the test images from the nine identified subsets show a
person who is also in training (five seeds). TownCentre alone, with about 30 images per person,
accounts for two thirds of that.

**CUHK** (4,563 images) labels files, not people (`0606.png <tags>`), and gives no person ids. The
identity split leaves it out by default (`--unidentified exclude`), since its people could
otherwise sit on both sides; `--unidentified keep` puts each image in as its own unit. Earlier
loaders got CUHK wrong in opposite ways: the age notebook matched on `<pid>` and skipped CUHK
(its 14,241 images are the other nine subsets), while the gender notebook's cell matched on file
names and would keep *only* CUHK. The published gender checkpoint, though, behaves as if trained
on the nine non-CUHK subsets (100% on them against 92% on CUHK,
`paper/results/stage1_training_data_check.json`), so it was made some other way. Either way,
**CUHK is held out for both released models**, which makes it the one clean PETA test set for
them. `train_stage1.ipynb` itself is unchanged; the RQ2 runs train from raw PETA through
`src/research/peta.py`, which does not have this problem. See `paper/README.md`, item 0.

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

`eval_frames.py --domain mars` and `extract_features.py --mars-index` read
`data/mars/tracklets.jsonl` through `read_mars_index` in `src/research/data.py`,
which turns the frame paths into absolute ones and keeps age as adult or old.

## RAP (optional target domain, RQ1 and RQ2)

Li et al. Indoor surveillance in a shopping mall, with gender and age bands
(under 16, 17–30, 31–45, 46–60, over 60) that match PETA's almost exactly.
Access needs a signed research agreement from the authors. Not requested yet.

## Retail clips (RQ1 to RQ3)

Recorded by Person A (task A3) with written permission. Record the source,
consent, the test / FDA-target split and the labelling here once they exist.
The 97 s web clip used so far (`vidssave.com ... retail store 720p.mp4`) has no
confirmed licence and must not appear in the paper.

Record 3 to 5 clips from different cameras or stores, about 150 to 300 people in total, **with
written permission** from the store. Keep them apart from the stage-2 training data on Drive: a
clip used to train stage 2 can never be test data.

One row per clip. `check_data.py` reads this table, so keep the column names, write the file name
exactly as it is on disk, and put `test` or `FDA frames` in the Use column.

| Clip | Store and camera | Recorded | Length | Permission (who, date, where the signed copy is) | Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | test / FDA frames |

**Use:** *test* clips are processed with `--export-crops` and labelled. *FDA frames* clips only
give unlabelled frames for Fourier Domain Adaptation (`--fda-target-dir`), and no person in them is
labelled or scored.

The steps, in order:

1. **Permission.** Print the letter in [research-permission.md](research-permission.md), get it
   signed, and keep the scan on Drive. Record nothing before that.
2. **Record and sort.** Put test clips in `footage/test/` and FDA clips in `footage/fda/`, never
   the same clip in both. Add a row above for each.
3. **Check.** Compare against everything stage 2 was trained on. The check compares content as
   well as file names, so a cut or re-encoded copy of a training clip is caught too.

   ```bash
   python scripts/research/check_data.py --test footage/test --fda footage/fda \
       --stage2 <folder of stage-2 training clips> --report data/check.json
   ```

4. **Process the test clips.**

   ```bash
   python scripts/process_folder.py footage/test --site "<store>" --camera "<camera>" \
       --gender-repo abhshkp/footfall-analysis-vit-stage1 \
       --age-bundle abhshkp/footfall-analysis-age-svm --export-crops
   ```

   Then run the check again with `--runs outputs/runs/<the new runs>`. It confirms that each run
   comes from a registered test clip, has crops, and reports how many people the runs hold.

5. **Hand the FDA frames to B.** This writes evenly spread frames and a manifest of which clip
   each came from:

   ```bash
   python scripts/research/fda_frames.py footage/fda --per-clip 200 --out data/fda_frames
   ```

   Upload `data/fda_frames/` to Drive for `run_rq2.py --fda-target-dir`.

6. **Features for the attention baseline.** After labelling (Section 4):

   ```bash
   python scripts/research/extract_features.py --runs outputs/runs/<test runs> \
       --gender-repo abhshkp/footfall-analysis-vit-stage1 --age-bundle abhshkp/footfall-analysis-age-svm
   ```

## Labelling protocol (task S2)

* Each annotator labels every test person independently in the dashboard's labelling page
  (`label.html`, or the **Label** button for a run with crops). Their labels go to
  `labels_<name>.json` in the run folder.
* Gender: female, male or unsure. Age: one of the four PETA groups or unsure. Label what the crops
  show; when they do not show it, choose unsure. The page hides the model's guesses.
* `python scripts/research/agreement.py --runs ... --a <name1> --b <name2>` gives Cohen's kappa and
  the list of disagreements. Resolve them together and save the agreed labels as
  `labels_final.json` in each run folder; the research scripts prefer that file.
* Unsure people are left out of accuracy figures, and the number left out is reported.

## Ethics notes for the paper

* Written permission from every store ([research-permission.md](research-permission.md)), kept
  with the clip record above.
* CCTV of members of the public: crops stay on the project machines and are not published; figures
  show no identifiable person.
* Gender is recorded as two categories because the datasets label it that way. This is a
  limitation of the data, not a claim about people.
* No children's data: PETA's under-15 group is excluded and no child is labelled in the retail clips.
