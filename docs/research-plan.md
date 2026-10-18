# Research plan: from product to paper

This is the plan for turning IFAS from a system description into a research paper. Nothing in
it is implemented yet. The work is split evenly between two people, **Person A** and
**Person B**, with a few shared tasks. Fill in the names below before starting.

| Role | Owns | Name |
| :--- | :--- | :--- |
| Person A | Retail data and temporal pooling (RQ3) | |
| Person B | Domain gap and adaptation (RQ1, RQ2) | |

---

## 1. Why

`paper/ifas.tex` describes a system. Its contributions are a pipeline, a sensitivity study, an
age test, timings and a tracker heuristic. None of them is a question tested against a baseline,
which is what reviewers at a research venue look for. The paper also lists its own gaps
(see *Limitations* and `paper/README.md`):

- There is no held-out test result for gender, only validation accuracy.
- Every training result comes from a single run (one seed).
- The only real-footage check is one 15 s web clip, labelled by eye.
- Accuracy per person (after pooling several frames) has never been measured.

One more issue affects any number the pipeline produced before the loader fix that ships with
this plan ("Load the fine-tuned ViT encoder instead of silently skipping it"). Until that fix,
the inference loader silently skipped the fine-tuned ViT encoder, so every gender prediction
from `src/inference/` ran on ImageNet weights. That includes the field check in Section IV-G.
Training-time validation numbers (from the notebooks) are not affected. All pipeline-based
gender results must be re-run.

Target: an **IEEE conference paper, at most 8 pages**.

## 2. Research framing

**Working title:** *From PETA to the Shop Floor: Measuring and Closing the Domain Gap in
Pedestrian Attribute Recognition for Retail CCTV*

**Central question.** Attribute models are trained on curated pedestrian crops (PETA) and
deployed on CCTV that is lower resolution, blurred, cut off by the frame edge and occluded.
How much accuracy is lost per person, why, and what recovers it?

### RQ1: How large is the gap, and what drives it?

Measure frame-level accuracy of the gender ViT and the age SVM on four domains: the PETA test
split, PETA with controlled degradations, a public video attribute dataset (MARS), and our own
labelled retail clips.

Use the **FFT** to describe the gap. The radially averaged power spectrum of the crops from each
domain shows how much high-frequency detail CCTV loses. Then test whether a crop's
high-frequency energy predicts whether the model gets it wrong.

*Hypothesis H1:* the ViT degrades less than the hand-crafted SVM when resolution drops, but
more when the person is truncated.

### RQ2: Can the gap be closed without labelling target data?

Train the gender ViT (stage 1) under five conditions, three seeds each:

| # | Condition | Labels needed from the store |
| :- | :--- | :--- |
| 1 | Baseline (current stage 1) | none |
| 2 | + degradation augmentation | none |
| 3 | + **Fourier Domain Adaptation** (FDA, Yang & Soatto, CVPR 2020) | none, only unlabelled frames |
| 4 | Stage 2, supervised fine-tuning on store data | yes |
| 5 | FDA + stage 2 | yes |

FDA swaps the low-frequency part of the FFT amplitude spectrum of each PETA training image with
that of a random unlabelled retail frame and keeps the phase. Training images then take on the
store's colour and lighting while keeping their content. For the age SVM, add one cheap
condition: train on features extracted from degraded PETA images.

*Hypothesis H2:* FDA recovers a large share of what supervised stage 2 recovers, at no
labelling cost.

### RQ3: Does pooling over a track help, and which frames should count?

Each person is classified on up to 12 frames, and the votes are pooled. Compare per-person
accuracy for:

1. one random frame (no pooling),
2. majority vote (what the pipeline does now),
3. mean probability,
4. confidence-weighted mean,
5. **quality-weighted** mean.

Quality weights come from cues measured on each crop:

- **Sharpness:** FFT high-frequency energy ratio (Laplacian variance as a sanity check).
- **Motion blur:** **Farneback** dense optical flow (`cv2.calcOpticalFlowFarneback`), the mean
  flow magnitude inside the box minus the background median.
- **Truncation:** the box touches the frame border.
- **Occlusion:** the box's largest IoU with any other box in the frame.
- **Size and detector confidence.**

A logistic regression maps the cues to P(this frame is classified correctly), and that
probability is the frame's weight. Also report accuracy against the number of frames pooled
(k = 1 … 12).

*Hypothesis H3:* quality-weighted pooling beats majority vote, with the largest gain on
truncated and occluded people.

Crowd-level "density flow" from optical flow is out of scope. It does not answer any of the
three questions. Farneback is used where it measures something testable: motion blur per crop.

## 3. Shared interfaces

Agree on these formats first. Then A and B can work in parallel against fake data.

**`crops.jsonl`** (written by A1 next to `analytics.json`, one line per classified crop)

```json
{"track": 12, "frame": 431, "t": 32.9, "box": [412, 88, 501, 330], "det_conf": 0.71,
 "frame_w": 1270, "frame_h": 720, "crop": "crops/12/000431.jpg",
 "flow": {"box_mag": 1.8, "bg_mag": 0.3}}
```

**`labels_<annotator>.json`** (written by the labelling page, one file per annotator so
agreement can be measured)

```json
{"run": "store-cam1-2026-10-20", "annotator": "A",
 "labels": {"12": {"gender": "female", "age": "Age31-45", "note": ""}}}
```

`gender` is `female`, `male` or `unsure`. `age` is one of the four PETA groups or `unsure`.
`unsure` people are excluded from accuracy, and the number excluded is reported.

**Per-crop predictions** (`paper/results/frames_<domain>.jsonl`, written by B4)

```json
{"domain": "retail", "condition": "fda", "model": "gender-fda-seed1", "item": "store-cam1/12/000431",
 "track": 12, "frame": 431, "probs": {"female": 0.82, "male": 0.18}, "label": "female"}
```

**Results** (`paper/results/rq1_gap.json`, `rq2_adaptation.json`, `rq3_pooling.json`): each
file holds `{"meta": {"commit", "date", "models", "seeds"}, "rows": [...]}`.
`paper/figures/make_figures.py` reads them under a new `experiment` source tag, which keeps the
rule in `paper/README.md` that no plotted number is estimated.

## 4. Tasks

Sizes: **S** ≈ 1 day, **M** ≈ 2–3 days, **L** ≈ 4–5 days. Each person carries 13 points
(S = 1, M = 2, L = 3).

### Person A: retail data and temporal pooling (RQ3)

| ID | Task | Size | Done when |
| :- | :--- | :-: | :--- |
| A1 | **Crop export.** Add `export_crops` to `process_video` in `src/inference/pipeline.py` and `--export-crops` to the CLI and `scripts/process_folder.py`. Reuse the existing vote sampling (`vote_every_sec`, `max_votes`) so experiments see what the product sees. Compute the Farneback stats at export time, where consecutive frames are at hand. | M | On the sample clip, the number of crops equals `classified_frames` summed over tracks; the `crops.jsonl` schema matches Section 3. |
| A2 | **Labelling page.** `app/static/label.html` + `label.js` in the dashboard's style. Show one person at a time as a strip of their crops, with keyboard shortcuts for gender and age. Add `GET /api/runs/{id}/crops` and `POST /api/runs/{id}/labels` to `app/server.py`, reusing the `_run_dir` path guard. | M | A headless-Chrome test labels two people and the file appears with the right content. |
| A3 | **Retail clips.** Record 3–5 clips (different cameras or stores, about 150–300 people in total) **with written permission**. Keep them separate from the stage-2 training data on Drive. Split them: test clips for labelling, and other clips whose unlabelled frames go to B for FDA. Write down the source, consent and split in `docs/research-data.md`. | M | Clips processed with `--export-crops`; FDA frames handed to B; data note committed (footage itself stays out of git). |
| A4 | **Quality cues.** `src/research/quality.py`: FFT sharpness, Laplacian variance, Farneback motion, truncation, occlusion, size. | M | Unit tests: sharpness falls monotonically as blur grows; motion is ~0 on a static frame pair and positive on a shifted one. |
| A5 | **Pooling study.** `src/research/pooling.py` (strategies as pure functions over `[(probs, cues)]`) and `scripts/research/pooling_eval.py`. Fit the logistic quality model on MARS (from B2) or with cross-validation over retail persons. Report per-person accuracy for each strategy, accuracy vs k, and the gain on truncated/occluded subsets. Include a **learned temporal-attention baseline** (a small attention head over frozen per-frame features, trained on MARS train), because that is what video attribute papers use; reviewers will ask how hand-measured quality weights compare. | L | Writes `paper/results/rq3_pooling.json` with confidence intervals from B7; tests pass on synthetic tracks with known answers. |
| A6 | **Back into the product.** If quality weighting wins, add `pooling=` to `process_video` with the majority vote kept as an option. | S | Default changed only if A5 shows a significant gain; `pytest` stays green. |
| A7 | **Tests** for A1–A6 in `tests/test_research_pooling.py` and `tests/test_label_api.py`. | S | Full `pytest` green. |

### Person B: domain gap and adaptation (RQ1, RQ2)

| ID | Task | Size | Done when |
| :- | :--- | :-: | :--- |
| B1 | **Lock versions and re-baseline.** Pin `transformers` in `requirements.txt` and in both training notebooks, so checkpoint and loader key names cannot drift apart again. Re-run every pipeline-based gender number with the fixed loader. | S | **Done**: pinned to 5.6.2; `paper/results/b1_gender_rebaseline.json`. |
| B2 | **Public video dataset.** Use the tracklet-level MARS-Attribute labels from `github.com/yuange250/MARS-Attribute` (`mars_attributes.csv`; the IRIP link in the original paper is dead). They have no licence file, and MARS itself is for research use only, so cite both and do not redistribute. Write `scripts/research/get_mars.py`. Gender maps directly. Age is young / teenager / adult / old, so report age on MARS as adult vs old only. MARS crops come from a detector and tracker, which suits RQ3. If MARS is not usable, fall back to PETA-only for the large-scale side and say so. | M | Script downloads and verifies the data; mapping documented in `docs/research-data.md`. |
| B2b | **RAP (optional, strongly recommended).** RAP was filmed by indoor surveillance cameras in a shopping mall. Its age bands (under 16, 17–30, 31–45, 46–60, over 60) match PETA's almost exactly, and it labels gender. That makes it a labelled retail-like target domain for RQ1 and RQ2 at no labelling cost. Access needs a signed research agreement from its authors: request it in Phase 1. | S + wait | Access granted; RAP added as a domain in B4 and as an FDA target in B6. |
| B3 | **Degradation suite.** `src/research/degrade.py`: downscale (48 / 32 / 24 / 16 px wide), Gaussian and motion blur, JPEG quality, bottom truncation (25 / 50 %), synthetic occlusion. Seeded and deterministic. | S | Tests: same seed gives the same output; output size and range are valid. |
| B4 | **Gap measurement.** `scripts/research/eval_frames.py` runs `GenderModel.predict` and `AgeModel.predict` (`src/inference/models.py`, unchanged) on every domain. `src/research/spectrum.py` computes radially averaged power spectra and the per-crop high-frequency ratio. | M | Writes `frames_<domain>.jsonl` and `paper/results/rq1_gap.json`; spectrum and accuracy-vs-degradation figures render. |
| B5 | **Adaptation transforms.** `degrade_transform` and `fda_transform(target_frames, beta)` in `src/training/transforms.py`. Add a condition switch to `TrainConfig` and a notebook cell. | M | Tests: FDA keeps the shape and the phase spectrum; `beta = 0` returns the input unchanged. |
| B6 | **Training runs (Colab).** 5 conditions × 3 seeds for gender (Section 2, RQ2), plus the age SVM degraded-feature condition in `age_classifier_v3.ipynb`. Log every run to JSON. Push the best new checkpoint to a **new** Hub repo; do not overwrite `-stage1` / `-stage2`. | L | `paper/results/rq2_adaptation.json` with mean ± sd per condition; the held-out gender test result exists. |
| B7 | **Statistics helper.** `src/research/stats.py`: balanced accuracy, mA, ordinal age metrics (off-by-one accuracy, mean absolute group error), bootstrap 95 % CIs over persons, McNemar's test, Cohen's κ. | S | Unit tests against hand-computed examples. A5 and B4 both use it. |
| B8 | **Tests** for B3–B7 in `tests/test_research_gap.py`. | S | Full `pytest` green. |

### Shared

| ID | Task | Who |
| :- | :--- | :--- |
| S1 | Agree on the Section 3 formats before writing code. | Both, first day |
| S2 | Label every retail person independently (about 3–4 hours each). Report Cohen's κ; resolve disagreements together. | Both |
| S3 | **Paper rewrite**, at most 8 pages, organised by RQ. *A:* System (cut to about one column: pipeline figure + pooling definition), RQ3 experiments, Ethics (consent, CCTV privacy, binary gender labels as a dataset limitation, no children's data), Limitations. *B:* Introduction ending in RQ1–RQ3, Related Work (add FDA and video attribute recognition), RQ1 and RQ2 experiments, `make_figures.py`. *Together:* title, abstract, conclusion. The old sensitivity study, runtime section and algorithm move to an appendix or are cut. | Split as listed |
| S4 | Close the open items in `paper/README.md` that this work answers: gender test result (B6), field check (A3 + B1), stage 2 (B6), single runs (B6). | Both |

## 5. Order and dependencies

| Phase | Person A | Person B | Hand-offs |
| :- | :--- | :--- | :--- |
| 1 | S1, A1, A2, start A3 | S1, B1, B2, B3, B7 | B7 → A (stats) |
| 2 | Finish A3, S2 labelling, A4 | S2 labelling, B4 on PETA and MARS, B5 | A3 → B (unlabelled retail frames for FDA) |
| 3 | A5, A6, A7 | B4 on retail, B6 on Colab, B8 | A1 + S2 → B (retail crops and labels); B2 → A (MARS for fitting the quality model) |
| 4 | S3, S4 | S3, S4 | Results JSON → `make_figures.py` |

The long pole is A3 + S2: recording and labelling the retail clips. Start recording in Phase 1.

## 6. Before you start

- The gender loader fix, the age model loaded from the Hub with its own feature settings,
  the Flat view fix and stage 2 as the default gender model are merged (PR #1).
- Never use stage-2 training clips as test data. Results on them would be inflated.
- Footage and labels stay out of git (`outputs/` and `models/` are already ignored). Commit code,
  results JSON and figures only.

## 7. Literature check (2026-10-18)

A quick search for prior work on each claim. It is not a full survey; Related Work (S3) must
still read and cite these properly.

| Our claim | Closest prior work found | Gap we fill |
| :--- | :--- | :--- |
| RQ1: PAR models lose accuracy across domains | Cross-domain PAR is an established setting: a CD-PAR baseline with adversarial domain alignment, evaluated on transfers including PETA ([PMC12944462](https://pmc.ncbi.nlm.nih.gov/articles/PMC12944462/)); the MSP60K cross-domain benchmark ([AAAI 2025](https://dl.acm.org/doi/10.1609/aaai.v39i4.32434)) | Those works measure dataset-to-dataset transfer. We measure the gap to live retail CCTV and decompose it by cause (resolution, blur, truncation, occlusion) with controlled degradations and FFT spectra. |
| RQ2: close the gap without target labels | Adversarial alignment (CD-PAR above). FDA has been used for nighttime pedestrian *detection* ([ResearchGate](https://www.researchgate.net/publication/365334449_Fourier_domain_adaptation_for_nighttime_pedestrian_detection_using_Faster_R-CNN)). | No application of Fourier Domain Adaptation to attribute recognition found. FDA needs no extra network or adversarial training, and we compare it directly with supervised stage-2 fine-tuning. |
| RQ3: pool per-frame predictions over a track | Video PAR on MARS with learned temporal attention ([Chen et al.](https://www.researchgate.net/publication/330466115_Video-Based_Pedestrian_Attribute_Recognition)), occlusion-aware group-sparse attention ([PMC9460213](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460213/)), CLIP-guided fusion ([VTF](https://arxiv.org/pdf/2304.10091)), side tuning ([2404.17929](https://arxiv.org/pdf/2404.17929)). In face and re-ID video, quality-aware aggregation exists ([C-FAN](https://arxiv.org/pdf/1902.07327), [CSA-Net](https://arxiv.org/pdf/2207.02631)). | No video PAR method found that weights frames by *measured* image quality. Ours is training-free on top of any frame classifier, interpretable, and must be compared against a learned attention baseline (A5). |

Two reviewer-facing risks to address in the paper:

1. **"Why not just train a video PAR model?"** Answer with deployment constraints: the frame
   classifier is fixed and shared with the product, the quality weights need no video labels,
   and we report the learned-attention baseline (A5) side by side.
2. **"One store is not a benchmark."** Use public data for scale and statistics (MARS, and RAP
   if access is granted) and the retail clips for realism. Report confidence intervals
   throughout (B7).
