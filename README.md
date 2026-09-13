# TreadCheck — Tyre Condition Screening

TreadCheck is a computer vision project I built to explore whether a smartphone photo can
be used to screen a tyre for visible signs of poor condition — worn tread, cracking,
perished rubber.

**Live demo: [tyre-tread-analysis-one.vercel.app](https://tyre-tread-analysis-one.vercel.app)**

You upload or take a photo of a tyre. The app first decides whether the image is good
enough to assess at all, and if it is, tells you whether the surface looks like tyres in
good condition or tyres with visible damage — along with how confident it is.

It **cannot** measure tread depth in millimetres, and it can't tell you whether a tyre is
legal or roadworthy. It's a screening aid that flags tyres worth having looked at
properly.

*(The API is on a free Render instance that sleeps when idle, so the first request after
a gap can take 30–60 seconds to wake up.)*

## Why I built this

Tread depth is what lets a tyre clear water and grip in an emergency stop. In India the
legal minimum is 1.6 mm for cars and 0.8 mm for two- and three-wheelers, measured against
the tread wear indicator moulded into the tyre ([CMVR Rule 95](https://www.atmaindia.org.in/laws-and-regulations/)).

Checking that properly needs a gauge. My impression is that most drivers don't do it
between services — that's my own observation, not a statistic. But a photo is something
anyone can take, and visible wear and cracking are the kind of thing a camera can pick
up. So I wanted to find out how far ordinary computer vision could get with just a phone
photo, and what such a system could honestly claim.

## How it works

The pipeline is deliberately classical rather than deep learning — I wanted features I
could reason about and debug.

1. Decode the photo, fix EXIF rotation, scale down to 1600 px.
2. Convert to greyscale, apply CLAHE and light smoothing.
3. Find the tread band in the middle of the frame.
4. **Quality gate** — check blur, exposure, contrast, glare, noise, JPEG compression and
   resolution. If the photo fails, stop here and say why.
5. Normalise the region to a fixed analysis size (only ever downscaling).
6. Extract 46 texture and frequency features — GLCM, local binary patterns, edge density
   and spectral descriptors.
7. Feed them to a calibrated RBF-SVM, which outputs a probability.

There are four possible outcomes: **Healthy**, **Defect detected**, **Not conclusive**
(the probability is too close to the boundary to call), and **Unable to assess** (the
photo wasn't good enough). The last two matter — I'd rather the system admit it doesn't
know than guess.

## Tech used

**Backend** — Python, OpenCV, scikit-image, scikit-learn, FastAPI, pytest
**Frontend** — React, TypeScript, Vite, `getUserMedia` for camera capture, Vitest
**Deployment** — Vercel (frontend), Render (backend), model binary published as a GitHub
release asset and fetched at build time

## What I changed while building it

The most useful thing I got out of this project was finding a mistake in my own earlier
version of it.

That version was built around a feature I called TSCI — a ratio of high-frequency to
total energy in the image — and it reported 74.8% accuracy. When I went back and tested
it properly, I found the feature was mostly responding to **the resolution of the source
image, not to the condition of the tyre**. Shrinking a photo without changing the tyre at
all moved TSCI about 3.6× more than the entire difference between good and worn tyres in
the dataset. A classifier reading nothing but the JPEG header got 68.6% on that data.

The cause was that every region was being resized to a fixed grid, which stretched small
web images and packed real detail out of large ones. Fixing it meant three things:

- never upsample — if a photo doesn't carry enough real detail, refuse it instead
- add the quality gate, so bad input is rejected before the model ever sees it
- drop TSCI from the classifier and keep it only as a diagnostic

That last one took two attempts. My first conclusion was "TSCI is broken", which turned
out to be too strong — it's fine once the image carries enough detail, it just doesn't
add anything the other features don't already capture.

The other change worth mentioning: the original version picked its decision boundaries
using predictions the model had made on its own training data, so they weren't really
measuring anything. The current thresholds come from out-of-fold predictions.

I also had to retire the original dataset. It was scraped from the web, its licence was
unclear, and most of its images were too low-resolution to analyse honestly once the
upsampling fix was in. The model now trains on a properly licensed Mendeley dataset.

## Results

Trained on 1,695 images (of 1,856) that pass the quality gate. Validation uses folds
grouped by image resolution, so the model can't score well by recognising a photo session
instead of a tyre.

| | |
|---|---|
| Balanced accuracy | 0.912 ± 0.014 |
| Brier score (calibration) | 0.069 |
| Abstains on | 16.6% of images |
| Balanced accuracy when it does answer | 0.960 |

I compared a linear SVM, logistic regression and a random forest against the RBF-SVM
before picking it. Full numbers, the experiments behind each decision — including the
ones that didn't work — are in [`experiments/LOG.md`](experiments/LOG.md), and the
original audit that started all this is in [`docs/AUDIT.md`](docs/AUDIT.md).

## Limitations

- **It cannot measure tread depth.** No dataset I could find has depth measurements, so
  any millimetre figure would be invented.
- **It can't reliably tell tread from sidewall.** A lot of the training images are
  sidewall close-ups. The app reports which surface it thinks it saw, but doesn't enforce
  it.
- **It doesn't transfer to other datasets.** I tested this — a model trained on one
  dataset drops to 0.59–0.66 balanced accuracy on the other, against 0.77–0.88 within its
  own. It's only validated on the kind of images it was trained on.
- **The region of interest is a fixed centre band**, not a real tyre detector.
- **Camera capture hasn't been tested against a real tyre yet.** It works in the deployed
  build and is covered by unit tests, but I haven't done that end-to-end test.
- Overexposed photos are the weakest case for the quality gate.

## Running it

```bash
pip install -e ".[dev,api,research]"
pytest
```

Backend and frontend, in two terminals:

```bash
python -m uvicorn tyretread.api.app:create_app --factory --port 8010
```

```bash
cp frontend/.env.example frontend/.env.local && npm install --prefix frontend && npm run dev --prefix frontend
```

Check a single photo from the command line:

```bash
python -m tyretread inspect path/to/tyre.jpg
```

## Project layout

```
tyretread/      the Python package - imaging, features, model, FastAPI app
frontend/       React + TypeScript app
experiments/    16 numbered experiments, each with a JSON record and a log entry
tests/          138 backend tests (45 more in frontend/)
docs/           audit, dataset notes, deployment and frontend documentation
src/            my original pipeline, kept unchanged for comparison
```

I left `src/` alone on purpose — it's what the original result was measured with, so
keeping it next to the corrected version makes the comparison checkable.

## Dataset

Not included in the repository. See [`docs/DATA.md`](docs/DATA.md) for provenance and
download instructions.

P, PATHMANABAN; C, Abishek; Sai, Kousik muthayala; S, Karthick; S, Aakash (2023),
"Digital images of defective and good condition tyres", Mendeley Data, V1,
doi: 10.17632/bn7ch8tvyp.1 (CC BY 4.0)

---

**Please note:** this is an assistive screening tool, not a safety certification. It works
from a photo, can't measure tread depth, and can be wrong. If a tyre is flagged as
possibly defective — or if the result is inconclusive — have it checked by a qualified
professional.
