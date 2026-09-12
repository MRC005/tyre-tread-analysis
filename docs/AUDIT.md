# Phase 1–3 Audit — Tyre Tread Wear Severity Estimation

**Date:** 2026-09-11
**Scope:** Read-only audit of the repository at `main` (commit `d839469`). No functional code was
modified, no commits created. Everything below was measured by running the existing code and
analysis scripts against the existing data; nothing is estimated or assumed.

---

## 0. Git identity and history check (Phase 10)

| Check | Result |
|---|---|
| `git config user.name` | `Machum Roy Choudhury` |
| `git config user.email` | `machumroychoudhury05@gmail.com` |
| Remote `origin` | `https://github.com/MRC005/tyre-tread-analysis.git` (unchanged) |
| Branches | `main`, `claude/tyre-tread-audit-925c11` (local worktree branch), `origin/main` |
| Commits in history | 9 |
| Authors | `Machum Roy Choudhury <machumroychoudhury05@gmail.com>` and `Machum Roy Choudhury <142169575+MRC005@users.noreply.github.com>` |
| Committers | Same, plus `GitHub <noreply@github.com>` on the two commits made via the GitHub web UI |
| Claude / Anthropic / AI identity anywhere in history | **None.** No AI author, no AI committer, no `Co-authored-by` trailer, no AI-related string in any commit message or body. |

**Clean.** No AI identity is present in the repository.

⚠️ **One discrepancy that needs your decision before any commit.** You specified the expected
commit email as `roychoudhurymachum05@gmail.com`, but the configured identity and the entire
existing history use `machumroychoudhury05@gmail.com` (note the reversed name order). Both are
plausibly yours. I have **not** changed the Git config. See "Questions for you" at the end.

---

## 1. What the repository actually contains

```
tyre-tread-project/
├── README.md                    HTML-in-Markdown project description
├── requirements.txt             6 unpinned packages
├── src/                         11 Python scripts, no package structure
│   ├── stage1_preprocessing.py  grayscale + CLAHE + Gaussian blur
│   ├── stage2_roi.py            centre crop + Canny + morphological close
│   ├── stage3_tsci.py           2-D DFT high-frequency energy ratio (TSCI)
│   ├── stage3_tpdi.py           DEAD CODE — 173 lines, imported nowhere
│   ├── stage4_texture.py        GLCM (5 props) + LBP + adaptive-Canny edge density
│   ├── stage5_fusion.py         KNN trainer + TSCI/KNN fusion — train_knn() never called
│   ├── batch_process.py         feature extraction over data/good + data/bad → results.csv
│   ├── train_and_evaluate.py    RBF-SVM, 5-fold CV, ablation, baselines
│   ├── clean_dataset.py         quality filter → data/clean/  (never used downstream)
│   ├── augment_dataset.py       6× augmentation → data/augmented/  (never used downstream)
│   └── main.py                  single-image demo, hardcoded path
├── data/                        357 MB, 3,552 images committed to Git
└── outputs/                     figures + results.csv + results_final.csv
```

**Absent entirely** — there is no frontend, no backend, no API, no saved model artifact, no tests,
no `.gitignore`, no config system, no environment variables, no logging, no Dockerfile, no CI, and
no deployment configuration of any kind. Phases 5–9 of your brief are greenfield work, not
refactoring work.

`src/__pycache__/*.pyc` (6 files) is committed to Git.

---

## 2. Reproduction of the published results

I ran `python3 src/train_and_evaluate.py` unchanged against the committed `outputs/results.csv`.

| Metric | README claim | Reproduced | Match |
|---|---|---|---|
| Accuracy | 74.8% | **74.80%** | ✅ exact |
| Worn-tyre recall | ~72% | **0.72** | ✅ exact |
| TSCI only | ~63.7% | **63.69%** | ✅ exact |
| GLCM only | ~66.4% | **66.40%** | ✅ exact |
| Edge only | ~64.0% | **63.96%** | ✅ exact |
| GLCM + Edge | ~72.6% | **72.63%** | ✅ exact |
| Full | 74.8% | **74.80%** | ✅ exact |

**The reported numbers are genuine and bit-for-bit reproducible.** That is a real credit to the
project and worth saying plainly. The problems below are about what those numbers *mean*, not
about whether they were honestly obtained.

`python3 src/main.py` also runs end-to-end without error.

---

## 3. Critical findings

These are ordered by how much they matter. Findings 3.1–3.3 are the ones that change what the
project can honestly claim.

### 3.1 🔴 TSCI runs in the opposite direction to the paper's physical justification

`stage3_tsci.py` documents TSCI as *"DECREASES monotonically with tread wear — a physically
well-founded and lighting-invariant metric"*, and the hardcoded thresholds encode that
(`TSCI > 0.55 → Safe`). Measured on the actual dataset:

| | mean TSCI | n |
|---|---|---|
| `good` (unworn) | **0.4220** | 234 |
| `bad` (worn) | **0.4958** | 135 |

AUC = **0.308**, point-biserial r = **−0.340**, p = 8.4×10⁻¹⁰. Worn tyres have *higher* TSCI, highly
significantly. The relationship is real but **inverted relative to the stated physics**, so the
`main.py` demo path — which classifies purely on those hardcoded thresholds — is systematically
backwards. `main.py` on `data/images/test.jpg` returns TSCI = 0.6576 → "Safe" via
"Frequency-domain (TSCI) — high confidence", when by the dataset's own statistics a TSCI that high
is evidence of wear.

### 3.2 🔴 TSCI is largely a proxy for source image resolution, not tread depth

The correlation that explains 3.1:

| | correlation with TSCI |
|---|---|
| image width | **+0.681** |
| image height | +0.541 |
| pixel count | +0.415 |
| file size in bytes | +0.382 |

The mechanism is in the code: `compute_tsci()` resizes every ROI to a fixed 256×128 before the FFT.
A 148-px-wide source is upsampled (smooth, low high-frequency energy); a 5000-px-wide source is
aggressively downsampled (aliasing, high high-frequency energy). TSCI therefore measures *how much
the source image was rescaled*, which in this dataset correlates with which folder the image came
from.

I controlled for this by restricting to a resolution-matched band (width 225–320 px, n = 196,
mean width 276 good vs 272 bad):

| Feature set | Balanced accuracy, full data | Balanced accuracy, resolution-matched |
|---|---|---|
| TSCI only | 0.606 | **0.474 — below chance** |
| GLCM + Edge | 0.720 | 0.659 |
| Full (all 7) | 0.732 | 0.697 |

**Once resolution is controlled, TSCI alone carries no usable wear signal.** The GLCM and edge
features do retain real signal (0.697 vs 0.500 chance), so the texture half of the pipeline is
sound — but the "core novel contribution" is not currently doing the work the paper attributes to it.

### 3.3 🔴 A classifier using only image metadata — no pixels at all — gets 68.6%

| Input | Accuracy (5-fold) | Balanced accuracy |
|---|---|---|
| Majority-class dummy | 63.41% | 50.00% |
| Image width + height only | 66.94% | 60.30% |
| Pixel count only | 67.48% | 59.79% |
| File size in bytes only | 66.67% | 57.42% |
| **All metadata, zero pixel content** | **68.56%** | **60.95%** |
| Proposed TSCI+GLCM+edge SVM | 74.80% | 74.17% |

The honest reading: the pipeline's 74.8% sits only ~6 points above what is achievable by reading the
JPEG header. The *balanced*-accuracy gap is much healthier (74.2% vs 61.0%), so the pipeline is
genuinely learning image content — but the headline accuracy figure overstates the achievement,
and **balanced accuracy is the metric this project should be reporting.**

### 3.4 🔴 The 74.8% is a binary result presented as a 3-class system

`train_and_evaluate.py` trains and cross-validates a **binary** good/bad classifier. The 3-class
Safe/Warning/Dangerous output is a post-hoc rule (`assign_3class`) applied afterwards, and it is
built from **in-sample predictions** — `model.fit(X, y)` followed by `model.predict(X)` on the same
`X` at lines 134–135. The `predicted_label` column in `results_final.csv` is therefore a
training-set fit, never validated. **The 3-class severity output — the actual product output — has
never been evaluated at all.** Its distribution on the training data (Safe 218 / Warning 118 /
Dangerous 33) includes 21 `bad` images labelled Safe and 12 `good` images labelled Dangerous.

### 3.5 🟠 Five pairs of byte-identical images carry contradictory labels

MD5 hashing found 8 exact-duplicate groups (16 files), **5 of them cross-class** — the same file
present in both `data/good/` and `data/bad/`:

```
good/1666881312128.jpg                    ≡ bad/1666881312128.jpg
good/ghows-LK-6ba044f3-...-bc18eb14.jpeg  ≡ bad/ghows-LK-6ba044f3-...-bc18eb14.jpeg
good/images180.jpg                        ≡ bad/images62.jpg
good/images453.jpg                        ≡ bad/images249.jpg
good/images892.jpg                        ≡ bad/images462.jpg
```

These set a hard ceiling on achievable accuracy and are direct evidence of label noise.
Perceptual hashing (dHash, Hamming ≤ 6) found 45 near-duplicate pairs, 8 cross-class; 369 files
collapse to **331 distinct visual groups**.

**However — and this matters for honesty — the duplicates do *not* inflate the cross-validation
score.** I re-ran with `StratifiedGroupKFold` keyed on visual group so near-duplicates can never
straddle a fold:

| | Standard 5-fold (leaky) | Group-aware 5-fold |
|---|---|---|
| RBF-SVM | 0.732 ± 0.044 | **0.733 ± 0.052** |
| LogReg | 0.759 ± 0.051 | **0.758 ± 0.041** |

No measurable difference. The duplicates are a data-hygiene and label-noise problem, not a
leakage-inflation problem at this scale. I am reporting this explicitly because "leakage inflated
the results" would have been the convenient story, and it is not what the data shows.

### 3.6 🟠 The dataset does not match the deployment domain

The product premise is *smartphone photographs of tread*. The data is web-scraped stock and news
imagery:

- **266 of 369 filenames are `images<N>.jpg`** — the default name Google Images assigns on download
- Others: `AdobeStock_179232317.jpeg`, `stock-photo-bald-tyre-with-no-tread-left-1415399372.jpg`,
  `maxresdefault3.jpg` / `sddefault.jpg` (YouTube thumbnails), `ghows-LK-...` (Gannett news CDN),
  `081510web_a1ROADTIRES052_t600.jpg`, `Treading-too-lightly-OPP-charge-driver-for-dangerously-bald-tires.jpg`
- Median resolution **284 × 204 px**; range 148–5000 px wide. Most common single resolution is
  275×183 (31 images) — a thumbnail size.
- 13 of 369 images carry an EXIF orientation tag; 356 have none (stripped by web pipelines).

A model trained on 284×204 web thumbnails has no established validity on a 12-megapixel phone
photo taken at arm's length in a car park. This is the single largest gap between the current
science and the intended product.

### 3.7 🟠 Licensing risk: 357 MB of scraped imagery committed to a public repository

The dataset is [Kaggle: numberfive/tire-tread-photos](https://www.kaggle.com/datasets/numberfive/tire-tread-photos),
whose own description reads *"Actually just 369 images"* — an exact match for this project's data.
Its Kaggle license field is **`Unknown`** (usability rating 0.4375, 0 votes). The underlying images
are third-party stock and press photographs. Redistributing them in a public Git repository is a
genuine legal exposure, independent of the ML questions.

### 3.8 🟠 The paper's "369 usable images after quality filtering" is not what the code did

369 is the **raw** dataset size. `clean_dataset.py` — the quality filter — outputs 352 images
(227 good / 125 bad) into `data/clean/`, and `augment_dataset.py` produces 2,464 into
`data/augmented/`. **Neither is used by anything.** `batch_process.py` reads `data/good` and
`data/bad` directly, and `results.csv` has exactly 369 rows: 234 good, 135 bad. No filtering was
applied to the reported experiment. This needs correcting in the write-up.

### 3.9 🟡 The headline model is not the best model, and the TSCI gain is not significant

Repeated CV (5-fold × 10 repeats, identical folds across models, paired tests):

| Model | Accuracy | Balanced accuracy |
|---|---|---|
| Dummy (majority) | 0.634 ± 0.002 | 0.500 ± 0.000 |
| **Logistic Regression** | **0.763 ± 0.049** | **0.759 ± 0.051** |
| RBF-SVM, C=1 | 0.751 ± 0.049 | 0.745 ± 0.051 |
| RBF-SVM, C=10 *(current)* | 0.737 ± 0.046 | 0.732 ± 0.044 |
| Extra Trees (500) | 0.734 ± 0.046 | 0.692 ± 0.050 |
| Random Forest (500) | 0.732 ± 0.044 | 0.689 ± 0.048 |
| Gradient Boosting | 0.715 ± 0.052 | 0.677 ± 0.053 |

Two results:

1. **Plain logistic regression significantly beats the RBF-SVM** (+0.027 balanced accuracy, paired
   t-test p < 0.0001). With 7 features and 369 samples, the RBF kernel at C=10 is overfitting. No
   tree ensemble beats either. *XGBoost/CatBoost were not tested — not installed, and the tree
   models already underperform, so there is no reason to expect them to help.*
2. **The TSCI contribution is not statistically significant.** Full (7 features) vs GLCM+Edge
   (6 features) = +0.0116 balanced accuracy, paired t-test **p = 0.063**. The README's claim that
   "feature fusion significantly improves accuracy" is not supported.

Also note the single-seed 74.80% is optimistic: the honest figure is **73.7% ± 4.6%** accuracy /
**73.2% ± 4.4%** balanced accuracy for the current model. A ±4.6% standard deviation on 369 samples
means differences under ~4 points are not interpretable.

### 3.10 🟡 Engineering defects

| Issue | Location |
|---|---|
| LBP histogram computed on every image, then **discarded** — never enters the feature vector | `stage4_texture.py` → `batch_process.py:34` |
| `stage3_tpdi.py`, 173 lines, imported nowhere | `src/` |
| `train_knn()` defined and documented, never called | `stage5_fusion.py:45` |
| Docstring says feature vector is 31-dim; it is 6-dim | `stage5_fusion.py:52,144` |
| `plt.show()` in Stage 1 & 2 blocks on any interactive backend — fatal for a server | `stage1_preprocessing.py:40`, `stage2_roi.py:57` |
| Hardcoded relative paths — scripts only work if CWD is the repo root | `batch_process.py:10`, `train_and_evaluate.py:15` |
| GLCM at `levels=256` on the full ROI: slow, and a very sparse matrix at these image sizes | `stage4_texture.py:34` |
| GLCM uses a single angle (0°) and single distance (1) — no rotation invariance | `stage4_texture.py:34` |
| `except Exception` swallows all errors to a print; a systematic failure would look like success | `batch_process.py:55` |
| **No model is ever persisted.** The SVM is fit, used, and discarded — there is nothing to deploy | `train_and_evaluate.py` |
| No confidence or probability output anywhere (`SVC` built without `probability=True`) | `train_and_evaluate.py:36` |
| Quality-gate logic exists but is offline-only — inference accepts any image unconditionally | `clean_dataset.py` vs `main.py` |
| `main.py` (threshold rule) and `train_and_evaluate.py` (SVM) are two separate systems that disagree | — |
| Unpinned dependencies — not reproducible | `requirements.txt` |

---

## 4. Assessment against your A–G framing

**A. What is already good — preserve this**
- Results are exactly reproducible. That is rarer than it should be.
- The 5-stage decomposition is clean, readable and genuinely modular; it maps well onto a service.
- The ablation study and the two baselines show real evaluation instinct.
- `class_weight='balanced'` and `StratifiedKFold` were the right calls for a 63/37 split.
- `cross_val_predict` rather than a single split — correct for a dataset this small.
- `clean_dataset.py` already contains a working image-quality heuristic (Laplacian variance, mean
  intensity, aspect ratio, std-dev). It is in the wrong place, not wrong.
- The per-stage visualisations are already 80% of an explainability feature.
- The classical-CV framing is a legitimate, defensible engineering choice — not a limitation.

**B. What is technically weak** — 3.1, 3.2, 3.4, 3.9 (scientific); 3.10 (engineering). The core
issue is that the pipeline's headline feature doesn't do what it claims, and the product's actual
output has never been evaluated.

**C. What is missing** — Everything product-side: API, frontend, model persistence, tests,
`.gitignore`, config, logging, error handling, confidence output, inference-time quality gate.

**D. For real-world use** — Domain-matched training data (3.6) and an inference-time quality gate
are prerequisites, not enhancements. Without them a deployed system would produce confident
nonsense on real phone photos.

**E. What would impress an interviewer** — Findings 3.1–3.3 are the answer. "I audited my own
published result, discovered my headline feature was a resolution artefact, proved it with a
resolution-matched control, and corrected the claim" is a far stronger story than 74.8%. Most
candidates cannot describe a negative result about their own work. The group-aware CV check in
3.5 — where I tested for leakage and honestly found none — is the same kind of signal.

**F. Genuine novelty** — An inference-time quality gate with a calibrated "Unable to assess"
outcome; resolution-invariant feature design (fixing 3.2 properly); calibrated confidence with
abstention; visual evidence tied to the specific measurements that drove the decision.

**G. Cosmetic — avoid** — Swapping to a CNN to chase accuracy on 369 noisy web thumbnails (it will
memorise the same confounds, with less interpretability). XGBoost/CatBoost (tree models already
underperform). Heavy animation. A "dashboard" with invented metrics. Any tread-depth-in-millimetres
readout — there is no ground truth anywhere in the available data to support it.

---

## 5. Dataset research (Phase 3)

| Dataset | Size | Labels | Acquisition | License | Verdict |
|---|---|---|---|---|---|
| **Current** — [Kaggle numberfive/tire-tread-photos](https://www.kaggle.com/datasets/numberfive/tire-tread-photos) | 369 | good / bad | Web-scraped, 148–5000 px | **Unknown** | In use. Domain-mismatched, licence-unclear |
| **[Mendeley doi:10.17632/bn7ch8tvyp.1](https://data.mendeley.com/datasets/bn7ch8tvyp/1)** (Pathmanaban et al. 2023), mirrored as [Kaggle warcoder/tyre-quality-classification](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification) | **1,854** | defective / good | **Mobile phone camera, 3000×3000, good lighting, multiple orientations** | **CC BY 4.0** | **Strong candidate — see below** |
| [Roboflow tire-tread-djsz9](https://universe.roboflow.com/tire-yekl9/tire-tread-djsz9) | 6,316 | BALD / NORMAL / BAD (3-class) | Unverified — Roboflow blocked automated inspection (HTTP 403) | Unverified | Needs manual check before any consideration |
| [Kaggle samwelnjehia/simple-tire-wear...](https://www.kaggle.com/datasets/samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset) | — | Simulated telemetry | Not images | Apache 2.0 | **Not applicable** — no imagery |

**The Mendeley dataset is the highest-value change available to this project**, and it fixes three
findings at once:

- **5× more data** (1,854 vs 369)
- **Smartphone-captured** — closes the domain gap in 3.6, the project's biggest scientific weakness
- **Uniform 3000×3000** — makes the resolution confound in 3.2/3.3 *structurally impossible*
- **CC BY 4.0** — legally redistributable with attribution, resolving 3.7
- **Citable DOI** — a properly published dataset, not an anonymous upload

Its limits, stated plainly: labels are still **binary**, still **no tread-depth ground truth in
millimetres**, and **"defective" is not defined** by the authors — it may include sidewall damage,
cracks or punctures rather than tread wear specifically. I would want to inspect a sample before
committing to it. It is 2.9 GB.

**No dataset I found provides tread-depth measurements.** Tread-depth ground truth appears to exist
only inside commercial systems (the search surfaced several US patents on tread-depth measurement,
all proprietary). This is a hard constraint, not a gap to be closed by more searching — which
directly supports your instinct in Phase 3 that the output should be redesigned to be honest about
what the data can support, rather than claiming depth estimation.

---

## 6. Proposed roadmap

Ordered by value per unit of effort. Nothing here is started — this is for your approval.

### Priority 1 — Scientific correctness (no new data needed, ~1 session)
1. Fix the resolution confound: make features scale-invariant (fixed physical sampling density
   rather than fixed output size), then re-measure. Report before/after honestly.
2. Correct the TSCI direction claim, or demonstrate it is salvageable once resolution-invariant.
3. Switch reported metric to **balanced accuracy with repeated-CV error bars**, and publish the
   metadata-only control (3.3) as a stated baseline. Any honest paper needs that number in it.
4. Remove the in-sample 3-class assignment; either evaluate 3 classes properly or reduce the
   claim to binary + calibrated confidence.
5. Adopt logistic regression as the primary model on the measured evidence, keeping the SVM as a
   documented comparison.
6. Quarantine the 5 contradictory duplicate pairs and document the decision.

### Priority 2 — Make it deployable (~1–2 sessions)
7. Package `src/` properly; persist a versioned model artifact with its scaler and feature order.
8. Promote `clean_dataset.py`'s heuristics into an **inference-time quality gate** returning
   *Unable to assess — retake* with specific reasons (too blurry / too dark / too little tread).
9. Calibrated probabilities + an abstention threshold, so weak predictions are never shown as certain.
10. FastAPI backend (FastAPI 0.141 and uvicorn are already installed): validation, size limits,
    EXIF handling, timeouts, CORS, health endpoint, structured logging.
11. Pytest suite over preprocessing, features, the quality gate, and the API contract.
12. `.gitignore`; untrack `__pycache__`; decide what to do about 357 MB of scraped images.

### Priority 3 — Dataset upgrade (needs your decision)
13. Retrain on the Mendeley CC BY 4.0 smartphone dataset and report the domain-shift result
    honestly, whichever direction it goes.

### Priority 4 — Product (~2–3 sessions)
14. Mobile-first camera capture flow, then the inspection-report UI with visual evidence and a
    "Why this result?" section, then Vercel + Render deployment.

---

## 7. Questions for you

1. **Git email.** You specified `roychoudhurymachum05@gmail.com`; the repo and all 9 existing
   commits use `machumroychoudhury05@gmail.com`. Which should future commits use? I have changed
   nothing. (Using a different email than the history would split your contributor graph on GitHub.)
2. **Mendeley dataset (2.9 GB, CC BY 4.0).** Approve download and integration? I will not fetch it
   without your go-ahead.
3. **The 357 MB of scraped images currently in Git.** Leave as-is, or stop tracking them and
   document how to fetch the dataset? This is a licensing question as much as a repo-size one.
4. **Roadmap scope.** Approve Priority 1 + 2 to start? I would rather fix the science before
   building a product on top of it.

---

## 8. Honesty note

Every number in this document was produced by running code against the data in this repository
during this audit. The reproduction figures in §2 come from the unmodified
`train_and_evaluate.py`. The repeated-CV, group-aware-CV, resolution-matched and metadata-control
experiments in §3 are new analyses run for this audit; their scripts are not yet committed to the
repo. Dataset facts in §5 come from the Kaggle public API and the Mendeley dataset page as cited;
the Roboflow entry could not be verified and is marked as such. No performance claim for any
proposed improvement appears anywhere in this document, because none has been implemented or
measured yet.


---

# Revisions since this audit

This document records the state of the project as found. Later experiments revised two
of its conclusions, and both revisions are recorded here so the audit is not read as
still-current where it is not. The full records are in `experiments/`.

## §3.1–3.2 revised by `exp004` — TSCI is not worthless, it has a missing precondition

The audit concluded that TSCI should be withdrawn. That was too strong.

`exp004` measured TSCI's resolution sensitivity *as a function of oversampling factor*
— how many native ROI pixels are available per analysis pixel. Below roughly 3×
oversampling the artefact is **2.5× the good-versus-worn signal**, which is the failure
this audit detected. At or above 3× it falls to **0.6×**.

So the ratio E_high / E_total is not intrinsically invalid. It carries an unstated
precondition — the image must contain genuine detail up to the analysis grid's Nyquist
limit — which roughly 78% of the legacy dataset violated. The defensible criticisms of
the original work are narrower and more precise than "the feature does not work":

1. the fixed-size resize was invalid, since it upsampled small images;
2. the published physical interpretation (TSCI decreasing with wear) is contradicted by
   the data;
3. no resolution precondition was stated or enforced.

TSCI is therefore a **candidate** feature behind the quality gate's oversampling floor,
not a discarded one. Whether it earns a place in the production model is decided by
measurement, not by principle.

## §5 revised by `exp006` — the Mendeley dataset is adopted, but not for the advertised reason

The audit recommended the Mendeley dataset largely because its page advertised a
uniform 3000×3000 phone-camera acquisition, which would have made the resolution
confound structurally impossible.

That claim is false. There are 666 distinct resolutions and **no image is 3000×3000**;
17% are small squares consistent with web thumbnails. The dataset also carries its own
acquisition confound — metadata alone predicts its labels at 0.655 balanced accuracy —
and, most importantly, **its labels describe sidewall cracking and rubber damage rather
than tread wear**.

It is still the right dataset to adopt, for reasons the audit did not anticipate: the
quality gate passes 93.5% of it against 22% of the legacy data, and its image features
reach **0.847 balanced accuracy under resolution-grouped folds** that block session
fingerprinting, with trivial appearance features scoring near chance. The signal is
real micro-texture.

What changes is the claim, not the dataset. The system screens for **visible tyre
condition** — worn, cracked or degraded rubber — not tread depth. See `docs/DATA.md`.

## §3.9 revised by `exp007` — logistic regression's win did not survive more data

On the legacy dataset's 7 features and 369 samples, logistic regression significantly
beat the RBF-SVM. On the Mendeley dataset's 46 features and 1,735 samples the ordering
reverses: the **RBF-SVM at C=10 is significantly better** (0.877 against 0.852 balanced
accuracy under resolution-grouped folds, p < 0.0001 on identical folds).

This is a sample-size effect rather than a contradiction — a kernel method needs enough
data to pay for its flexibility, and at 369 samples it did not have it. It does carry a
real cost: the SVM is not linear, so predictions **cannot be decomposed exactly** into
per-feature contributions, and the explanation layer correctly refuses to invent a
substitute. That trade-off is an open decision, not a settled one.
