# Tyre Condition Screening

**Assistive visual screening of tyre condition from a smartphone photograph.**

Upload or photograph a tyre. The system decides whether the image is good enough to
assess, and if it is, reports whether the surface resembles tyres with visible wear or
damage — with a calibrated probability, the measurements behind it, and an explicit
refusal when it cannot tell.

> **What this is not.** It cannot measure tread depth, cannot say whether a tyre is
> legal or roadworthy, and is not a certified inspection. It is a screening aid that
> tells you when a tyre is worth having looked at properly.

---

## Contents

- [What the system does](#what-the-system-does)
- [What it can and cannot determine](#what-it-can-and-cannot-determine)
- [Results](#results)
- [How it works](#how-it-works)
- [The research story](#the-research-story)
- [Datasets](#datasets)
- [Known failure modes](#known-failure-modes)
- [Running it](#running-it)
- [API](#api)
- [Project layout](#project-layout)
- [Status and roadmap](#status-and-roadmap)

---

## What the system does

```
photograph
   ↓  decode, apply EXIF orientation, reduce to 1600 px
   ↓  greyscale · CLAHE · Gaussian smoothing
   ↓  locate the tread band  (contour, or centre-band fallback)
   ↓  QUALITY GATE ──────── refuse ──▶  "Cannot assess" + specific retake advice
   ↓     blur · exposure · dynamic range · glare · noise · compression · resolution
   ↓  scale normalisation (downsample only — never upsample)
   ↓  46 texture and frequency features
   ↓  surface check ──▶ tread / sidewall / unclear   (reported, never used to refuse)
   ↓  calibrated RBF-SVM  ──▶  probability of a visible defect
   ↓  abstention band ────── inside ──▶  "Not conclusive"
   ▼
verdict + confidence + surface + measured evidence + visual evidence
```

Four outcomes, not three, and the fourth two are the point:

| Shown as | Verdict | Meaning |
|---|---|---|
| **Healthy** | `likely_serviceable` | No visible defects found |
| **Attention recommended** | `inconclusive` | Inside the abstention band — not confident either way |
| **Defect detected** | `defect_suspected` | Visible signs of wear, cracking or perished rubber |
| **Unable to assess** | `unable_to_assess` | The photograph does not support any assessment |

The positive label is deliberately not "High risk": risk depends on remaining tread
depth, which this system cannot measure and for which no ground truth exists.

## What it can and cannot determine

Being precise about this is the point of the project, so it is stated before the
results rather than after them.

**It can** distinguish tyre surfaces that resemble the *visibly defective* group of its
training data — worn tread, cracking, perished rubber — from those resembling the *good
condition* group, and it reports a calibrated probability for that judgement.

**It cannot:**

- **Measure tread depth.** No dataset available to this project contains tread-depth
  measurements. Searching surfaced only proprietary patented systems. Any millimetre
  figure would be fabricated.
- **Say which defect it is seeing.** Its training labels do not separate worn tread
  from sidewall cracking, so neither can it. `exp006` found the positive class is
  dominated by cracking and perished rubber.
- **Tell tread from sidewall.** Many training images are sidewall close-ups. The ROI
  extractor will happily return a "tread band" from a sidewall photograph.
- **Judge legality or roadworthiness.** Those depend on a measured depth against a
  jurisdiction's limit.
- **Generalise to a different imaging domain.** Measured, not assumed — see
  [the transfer result](#results).

## Results

All figures below were produced by scripts in `experiments/` and are reproducible.
None is quoted from the original paper.

**Validation methodology.** Balanced accuracy is the headline metric, because the data
is 55/45 and because a plain-accuracy figure once made an image-metadata confound look
like skill. Folds are **grouped by exact pixel resolution**, which prevents a model from
recognising a photo session rather than a tyre — this is the conservative estimate, and
it is about three points below ordinary folds.

### Model selection

Six calibrated candidates, all on the **same 20 resolution-grouped folds**, compared
with paired t-tests (`exp007`):

| Model | Balanced accuracy | vs best |
|---|---|---|
| **RBF-SVM, C=10** | **0.871 ± 0.034** | — |
| RBF-SVM, C=1 | 0.855 | p < 0.0001 |
| Linear SVM | 0.853 | p < 0.0001 |
| Logistic regression | 0.846 ± 0.016 | p < 0.0001 |
| Logistic regression (C=0.1) | 0.845 | p < 0.0001 |
| Random forest | 0.839 | p < 0.0001 |
| Majority class | 0.500 | — |

Gradient-boosted trees (XGBoost, CatBoost) were **not** tried. Tree ensembles are
already the weakest family here, which is the expected outcome for a few thousand
samples described by smooth, correlated, continuous features — adding a heavier
dependency to lose accuracy would be decoration.

This **reverses an earlier finding**. On the legacy dataset's 7 features and 369
samples, logistic regression beat the RBF-SVM significantly. At 46 features and 1,735
samples the ordering flips. That is a sample-size effect, not a contradiction: a kernel
method needs enough data to pay for its flexibility.

The RBF-SVM is served because the evidence supports it, and it carries a real cost:
being non-linear, its decision **cannot be decomposed** into per-feature contributions.
The system says so rather than substituting a surrogate. See
[explainability](#explainability-two-different-claims).

### Calibration and abstention

| | |
|---|---|
| Brier score | **0.072** |
| Expected calibration error | **0.070** |
| Decision threshold | 0.50 (chosen against a defect-recall target, not assumed) |
| Abstention band | ±0.20 |
| Abstention rate | **17.0%** |
| Balanced accuracy on answered cases | **0.957** |
| Defect recall on answered cases | 0.967 |

Threshold and band are both chosen from **out-of-fold** probabilities. The original
project chose its class boundaries from in-sample predictions, which is why they were
never a measurement of anything.

### Cross-domain transfer — a negative result

Trained on one dataset, tested on the other (`exp008`, 2000-sample bootstrap CIs):

| | Balanced accuracy | Within-dataset |
|---|---|---|
| Mendeley → legacy | 0.665 [0.559, 0.765] | 0.769 |
| Legacy → Mendeley | 0.595 [0.572, 0.617] | 0.878 |

**Transfer largely fails.** A linear baseline behaves the same way, so it is a property
of the data rather than of one model. Pooling the datasets changes nothing (−0.018) and
would make the training target a blend of two concepts.

The informative part is a check that did not go as expected: a classifier can tell
which dataset an image came from at only **0.627** balanced accuracy. Had that been
near 1.0, the transfer failure would be ordinary domain shift. It is not — the images
are only weakly distinguishable, yet a model trained on one is nearly useless on the
other. That points at the **labels**, not the appearance: the two datasets name
different properties.

**Consequence:** this system is validated only within its training domain, and there is
direct evidence it does not yet transfer.

## How it works

### Features (46)

| Family | What it measures |
|---|---|
| Scale-invariant spectral (5) | Power-law spectral slope and its goodness of fit; angular energy distribution — anisotropy, coherence, entropy |
| GLCM (20) | Contrast, dissimilarity, homogeneity, energy, correlation at 3 distances, averaged over 4 angles, plus per-property angular spread |
| LBP (18) | Rotation-invariant uniform local binary patterns |
| Edge / gradient (3) | Adaptive-Canny edge density, mean and 95th-percentile gradient magnitude |

Ablation under resolution-grouped folds (`exp006`): LBP alone 0.819, GLCM 0.768, edge
0.761, spectral 0.651, **all together 0.847**. Trivial appearance features — brightness,
contrast, sharpness — reach only **0.540**, so the model is not separating clean tyres
from dirty ones.

### The quality gate

Runs **before** the model, so a prediction for a rejected image is structurally
impossible. Checks blur (Laplacian variance), exposure, contrast, glare (saturated
fraction), aspect ratio, whether the tread was located, and **oversampling** — how many
native pixels are available per analysis pixel. Each refusal names the issue and gives
an action ("Hold the phone still and let the camera focus").

It passes 93.5% of the Mendeley data and 22% of the legacy data. That difference is a
measurement of the legacy dataset, not a defect in the gate.

### Explainability: two different claims

The report separates two things that are easy to conflate:

**Measured evidence** — what this tyre's surface measures, as a percentile of the
training distribution. A fact about the photograph, true whatever the model does.

**Model interpretation** — why the model decided as it did. Exact only for a linear
model. For the served RBF-SVM it is **unavailable**, and the report says so instead of
guessing.

When individual measurements disagree with the overall verdict — which happens, because
the classes are separated by the *combination* — the report states that plainly rather
than hiding it.

## The research story

This project's most useful output is a correction to its own earlier work.

**The original pipeline** computed a Tyre Surface Clarity Index, TSCI = E_high/E_total,
documented as decreasing with tread wear, and reported 74.8% five-fold accuracy.

**That result reproduces exactly.** It is also not what it appeared to be.

1. **The confound** (`exp001`). Downsampling a photograph *without changing the tyre*
   moves TSCI by **+0.266**, while the entire good-versus-worn signal is **0.074** — an
   artefact 3.6× the size of the signal. Cause: resizing every ROI to a fixed 256×128
   upsamples a thumbnail, whose upper octaves are already empty, while packing a large
   photograph's real detail into the same grid. A classifier reading **only the JPEG
   header** reached 68.6% accuracy on that dataset.

2. **The first conclusion was too strong** (`exp004`). "TSCI is broken" did not survive
   scrutiny. Its resolution sensitivity falls from 2.5× the class signal to **0.6×**
   once oversampling exceeds ~3×. The real fault is a **missing precondition** — the
   image must carry genuine detail to the analysis grid's Nyquist limit — which 78% of
   the legacy dataset violated.

3. **The fix** (`exp002`, `exp003`). Never upsample; normalise every ROI downward onto
   a common grid; refuse images that cannot fill it. **46 of 48 descriptors become
   resolution-stable.** The cost is real: no configuration keeps more than 80 of the
   legacy dataset's 369 images at a defensible floor, which is why that dataset was
   retired from training.

4. **And TSCI still does not earn its place** (`exp009`). Behind the gate, added to the
   production features, it changes balanced accuracy by **+0.0006, p = 0.60**. Its
   univariate AUC of 0.627 collapses to **0.508 — chance** — under session-blocked
   folds. Its direction also contradicts the original paper's physics on *both*
   datasets, now including one where the resolution confound cannot be blamed.

**TSCI is therefore reported as a diagnostic and excluded from the classifier** — valid
but redundant, which is a different finding from worthless, and the distinction is kept
rather than flattened.

## Datasets

Neither dataset is tracked in this repository. See **[docs/DATA.md](docs/DATA.md)** for
provenance, licensing, measured properties and download instructions.

| | Legacy scraped | Mendeley |
|---|---|---|
| Images | 369 | 1,856 |
| Licence | **Unknown** | **CC BY 4.0** |
| Source | Google Images, stock and press photos | Phone camera, plus ~17% web thumbnails |
| Labels | tread condition | tyre condition (damage-dominated) |
| Tread depth | none | none |
| Quality-gate pass rate | 22% | 93.5% |
| **Role** | Reproducing the published result only | **Training** |

The Mendeley dataset's page claims 1,854 images at a uniform 3000×3000. Measured: 1,856
images across **666 distinct resolutions, none of them 3000×3000**. Its labels are also
damage-dominated rather than tread-specific. It is still the right choice, for reasons
its description does not mention.

**Attribution (CC BY 4.0):** P, PATHMANABAN; C, Abishek; Sai, Kousik muthayala;
S, Karthick; S, Aakash (2023), "Digital images of defective and good condition tyres",
Mendeley Data, V1, doi: 10.17632/bn7ch8tvyp.1

## Known failure modes

Measured in `exp010` by degrading 60 images the system handles confidently, before and
after the gate work in `exp012`:

| Degradation | Refused **before** | Refused **after** | Silent flips before → after |
|---|---|---|---|
| Motion blur | 100% | 100% | — |
| Photographed too far away | 100% | 100% | — |
| Glare patch | 100% | 100% | — |
| **Underexposed** | 53% | **100%** | **20% → 0%** |
| **Sensor noise** | 0% | **100%** | 4% → **0%** |
| **Heavy JPEG (q12)** | 7% | **93%** | 5% → **0%** |
| Overexposed | 78% | 78% | 14% → **0%** |
| *(clean images answered confidently)* | 93% | 93% | — |

**The silent-flip rate — a confident verdict that changes under degradation with
nothing to warn the user — is now zero across every degradation tested**, and the rate
at which clean images are answered confidently is unchanged. The cost is a dataset pass
rate falling from 93.5% to 90.7%, chiefly images that are genuinely too dark.

The root cause of the underexposure blind spot is worth stating: exposure was measured
on the **CLAHE-equalised** image, and CLAHE exists to normalise local contrast, so it
hid the very defect the check was looking for. A photograph with a raw mean of 30 came
out of CLAHE at 57 and passed a threshold of 30.

Separating a genuinely dark tyre from an underexposed photograph needed a second
measurement: both have a low mean, but only the underexposed one has a compressed
histogram, so **dynamic range** is checked alongside mean intensity. Normal phone JPEGs
are explicitly protected — quality-85 measures about 1.19 blockiness and quality-60
about 1.52, against 3.40 at quality 12, with the threshold at 2.2.

### Tread versus sidewall

Hand-labelling 120 randomly sampled accepted images found that **47% are not clean
tread** — 39% sidewall close-ups, 8% mixed (`exp011`). Many are photographs of moulded
sidewall lettering containing no tread at all.

Two measurements decided what to do about it:

- The surface is **not** associated with the condition label (chi-square p = 1.00), so
  the model is not covertly learning "this is a sidewall photograph", and it performs
  comparably on both surfaces.
- A surface detector reaches **AUC 0.797 [0.698, 0.878]** — real signal, but at a
  threshold flagging three-quarters of non-tread images it also flags one genuine tread
  photograph in five.

So the surface is **reported, not enforced**. Refusing sidewall images would discard
working functionality to enforce a distinction the system cannot make confidently. Each
result states which surface it believes it assessed, and a non-tread result says
explicitly that it says nothing about remaining tread. The detector is fitted on 120
hand labels and should be refitted on a larger annotated set before being relied on
more heavily.

### Remaining limitations

- **Overexposure** is the weakest guard: 78% refused, 17% still answered confidently.
  Tightening further starts refusing real images. Its silent-flip rate is now zero, so
  the residual risk is a wrong answer offered rather than a changed one.
- **No transfer to other imaging domains** — measured, see [Results](#results).
- **Training labels carry an acquisition confound**; all reported figures use
  resolution-grouped folds to suppress it.
- **The legacy dataset contains five byte-identical cross-label pairs.**

## Running it

```bash
pip install -e ".[dev,api,research]"
pytest
```

Run the whole product locally — backend, then frontend:

```bash
python -m uvicorn tyretread.api.app:create_app --factory --port 8010
```

```bash
cp frontend/.env.example frontend/.env.local && npm install --prefix frontend && npm run dev --prefix frontend
```

See [docs/FRONTEND.md](docs/FRONTEND.md) for the camera flow, error handling and
privacy behaviour, and [docs/PHONE_TESTING.md](docs/PHONE_TESTING.md) for testing on a
real phone over HTTPS.

**Camera capture is implemented but not yet verified on physical hardware** — it needs
HTTPS, which `localhost` development cannot provide.

Inspect one photograph:

```bash
python -m tyretread inspect path/to/tyre.jpg
```

Run the API:

```bash
uvicorn tyretread.api.app:create_app --factory --reload
```

Reproduce an experiment (needs the datasets — see [docs/DATA.md](docs/DATA.md)):

```bash
python experiments/exp001_resolution_confound.py
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/inspect` | Assess one photograph. Multipart upload. |
| `GET /health` | Liveness plus whether a model is loaded |
| `GET /v1/model` | Which artifact is served and how it was validated |

A refused image is a **200 with an `unable_to_assess` verdict**, not an error — the
request succeeded, and "this cannot be assessed, here is why" is a result the client
must render. Errors are reserved for genuine faults: `413` oversized, `415` unsupported
type, `400` undecodable, `503` no model, `504` timeout.

Configuration is environment-based; see [.env.example](.env.example). The API refuses
to start with a wildcard CORS origin in production.

## Project layout

```
frontend/            React + TypeScript app — mobile-first, light theme, 57 KB gzipped
tyretread/           the package — one inference path, shared by API, CLI and tests
  imaging/           loading, EXIF, preprocessing, ROI, quality gate
  features/          spectral and texture descriptors, extraction orchestrator
  models/            artifact persistence, training, calibration, decision, evidence
  api/               FastAPI app, settings, response schemas
  inspect.py         the single production entry point
experiments/         numbered, self-describing, each writing a JSON record + LOG.md
tests/               131 backend tests, including regression tests on the findings
docs/                AUDIT.md · DATA.md · DEPLOYMENT.md · FRONTEND.md · PHONE_TESTING.md
src/                 the original pipeline, preserved and still reproducing 74.8%
```

`src/` is deliberately unchanged. It is what the published result was measured through,
and `tyretread/` is the corrected system — keeping both is what makes the comparison
checkable.

## Status and roadmap

**Now.** Corrected feature pipeline; quality gate covering blur, exposure, dynamic
range, glare, noise, compression and resolution; surface reporting; calibrated model
with abstention; persisted 0.72 MB artifact; FastAPI backend; mobile-first frontend
with camera capture; 130 backend + 33 frontend tests; 12 recorded experiments.

**Next.** Verify camera capture on a physical phone over HTTPS, then deploy to Vercel
and Render using a versioned release asset for the model.

**Later.** Tread-depth regression *if* expert-labelled depth data becomes available —
not before. Tread-versus-sidewall discrimination. Multi-image inspection.

---

### Responsible use

This is an assistive screening tool, not a safety certification. It works from a
photograph, cannot measure tread depth, and can be wrong. Any tyre flagged as possibly
defective, and any inconclusive result, should be inspected by a qualified professional.
