# Datasets

No image data is tracked in this repository. This file records what each dataset is,
where it comes from, what its licence permits, and what it is used for. Every claim
here was verified during the audit; nothing is assumed from a dataset's title.

## Why nothing is tracked

The original dataset is 357 MB of third-party stock and press photographs with an
upstream licence of "Unknown". Redistributing it from a public repository is a legal
exposure that has nothing to do with whether the machine learning works, so it was
untracked. The replacement dataset is properly licensed but 2.7 GB, which does not
belong in Git history either.

Consequence: a fresh checkout can run the test suite, but tests that genuinely need a
photograph will **skip** with a message pointing here rather than fail.

## 1. Legacy scraped dataset — *retired from training*

| | |
|---|---|
| Source | [Kaggle `numberfive/tire-tread-photos`](https://www.kaggle.com/datasets/numberfive/tire-tread-photos) |
| Size | 369 images (234 serviceable / 135 worn) |
| Licence | **Unknown** — not redistributable |
| Resolution | median 284 x 204 px; range 148–5000 px wide |
| Labels | binary folder labels, `good` / `bad` |
| Tread depth | none |

The dataset's own Kaggle description reads "Actually just 369 images", which matches
this project's data exactly. 266 of the 369 filenames are `images<N>.jpg`, the default
name Google Images assigns on download; others are identifiably from Adobe Stock,
iStock, Shutterstock, YouTube thumbnails and a news CDN.

**Known defects**, all measured (see [AUDIT.md](AUDIT.md)):

- Five pairs of **byte-identical images filed under both classes** — direct label
  contradictions that cap achievable accuracy.
- 45 near-duplicate pairs; the 369 files collapse to 331 distinct visual groups.
- Resolution correlates with class, which made the original TSCI feature a
  resolution proxy (exp001) and lets a classifier reading only the JPEG header reach
  68.6% accuracy.
- **Roughly 78% of it cannot be analysed correctly at all**: the images are too small
  to fill the analysis grid without upsampling, which is what caused the original
  confound (exp003).

**Current role.** Reproducing the originally published 74.8% result, and serving as an
out-of-domain robustness check where individual images have enough resolution. It is
not used for training.

### Obtaining it

Download from the Kaggle link above and arrange as:

```
data/good/    # serviceable tyres
data/bad/     # worn tyres
```

Because its licence is unknown, treat it as research-only and do not redistribute it.

## 2. Mendeley dataset — *adopted for training, with a corrected claim*

| | |
|---|---|
| Source | [Mendeley Data `10.17632/bn7ch8tvyp.1`](https://data.mendeley.com/datasets/bn7ch8tvyp/1) |
| Mirror | [Kaggle `warcoder/tyre-quality-classification`](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification) |
| Authors | Pathmanaban P, Abishek C, Kousik Muthayala Sai, Karthick S, Aakash S (Velammal Engineering College, 2023) |
| Size | **1,856 images** (1,028 defective / 828 good), 2.7 GB |
| Licence | **CC BY 4.0** — redistributable with attribution |
| Labels | binary: `defective` / `good condition` |
| Tread depth | none |

### What the dataset page claims, and what is actually true

Everything below was measured during this project's audit (`exp006_mendeley_audit`).
The dataset's description is inaccurate in three respects, and the errors matter
enough to record.

| Claim on the dataset page | Measured |
|---|---|
| "1854 digital tyre images" | **1,856** images |
| "high resolution (3000*3000)" | **666 distinct resolutions; not one image is 3000×3000.** Most common are 4000×1800 (26%) and 1800×4000 (18%) |
| "mobile phone camera ... good lighting environment" | Largely true for the bulk of it, but **309 images (17%) are small squares between 224 and 600 px** — the signature of web thumbnails, not phone originals |

### The acquisition confound

The two classes were photographed in separate sessions, and that leaks:

- **Metadata alone — no pixels at all — predicts the label at 0.655 balanced accuracy.**
- Four resolution groups are **100% single-class**. An exact sensor resolution
  fingerprints one camera in one session.
- Orientation splits sharply by class: 42% of `defective` images are landscape against
  19% of `good` ones.

The mitigation is to group cross-validation folds by exact pixel resolution, which
blocks a model from recognising the session instead of the tyre. Under that scheme the
metadata-only model **collapses to 0.534 — chance** — confirming it was pure
fingerprinting. **All reported metrics use resolution-grouped folds**, which are about
three points lower than ordinary folds.

### 🔴 The labels are about tyre *damage*, not tread depth

This is the most important finding, and it changes what the system can claim. Visual
inspection of 40 randomly sampled images, 20 per class:

- **`defective`** is dominated by **sidewall cracking, splits, perished rubber and bead
  damage**. Many images show the **sidewall rather than the tread at all**, and several
  show deep, healthy tread on a tyre that is damaged elsewhere.
- **`good`** is largely **new or nearly-new tyres in retail condition**.

So the dataset does not label tread wear. A model trained on it separates *visibly
damaged or degraded rubber* from *rubber in good condition* — a real and useful
distinction, but not the same thing as tread depth.

### But the signal is genuine

Under resolution-grouped folds:

| Input | Balanced accuracy |
|---|---|
| Metadata only (no pixels) | 0.534 |
| Trivial appearance — brightness, contrast, sharpness | 0.540 |
| Spectral + orientation descriptors | 0.651 |
| Edge / gradient | 0.761 |
| GLCM | 0.768 |
| LBP | 0.819 |
| **All image features** | **0.847** |

Trivial appearance features score near chance, so the model is not simply telling clean
tyres from dirty ones. The discriminative power sits in surface **micro-texture**, which
is what the pipeline was built to measure.

The quality gate passes about **91%** of this dataset, against 22% of the legacy one.
(It was 93.5% before `exp012` tightened the exposure, noise and compression checks;
the images it now turns away are mostly genuinely too dark.)

### Required attribution

> P, PATHMANABAN; C, Abishek; Sai, Kousik muthayala; S, Karthick; S, Aakash (2023),
> "Digital images of defective and good condition tyres", Mendeley Data, V1,
> doi: 10.17632/bn7ch8tvyp.1. Licensed CC BY 4.0.

### Obtaining and arranging it

```bash
mkdir -p data/external
curl -L -o data/external/mendeley-bn7ch8tvyp-1.zip \
  "https://data.mendeley.com/public-api/zip/bn7ch8tvyp/download/1"
unzip -q data/external/mendeley-bn7ch8tvyp-1.zip -d data/external/raw
```

Then arrange as, renaming away the spaces in the distributed folder name:

```
data/external/mendeley_tyres/worn/          # from "defective"
data/external/mendeley_tyres/serviceable/   # from "good"
```

## 3. Datasets considered and rejected

| Dataset | Why rejected |
|---|---|
| [Kaggle `samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset`](https://www.kaggle.com/datasets/samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset) | Simulated telemetry, not images. Not applicable. |
| [Roboflow `tire-tread-djsz9`](https://universe.roboflow.com/tire-yekl9/tire-tread-djsz9) | Advertises 6,316 images with three classes, which would be attractive. Automated inspection was blocked (HTTP 403) so licence, provenance and label definitions are **unverified**. Not used until checked by hand. |

## On tread-depth ground truth

No public dataset found provides tread depth in millimetres. Searching surfaced
several US patents on tread-depth measurement, all proprietary. This is a hard
constraint rather than a gap that more searching will close, and it is the reason the
system reports a hedged condition assessment with calibrated uncertainty rather than a
depth estimate. If labelled depth data becomes available, depth regression is a
natural extension — but it is not claimable today.
