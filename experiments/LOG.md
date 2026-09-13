# Experiment log

Generated from the JSON records in this directory by
`tyretread.experiment.rebuild_log` - do not edit by hand. Each entry states a
hypothesis, what was run, what came out and what was decided as a result.
A negative result with a clear decision is a successful experiment.

Note on the feature cache. Most experiments read a cached feature table rather
than re-extracting features. `exp012` tightened the quality gate, which changed
how many images pass (1,735 before, 1,695 after). The cache was rebuilt then and
`exp007` re-run against it, so the served model and its metrics are the
post-tightening ones. `exp006`, `exp008`, `exp009` and `exp011` were not re-run,
so their records below describe the 1,735-image pipeline. See the README under
"Reproducing these figures" for what moves when they are.

---

## exp001_resolution_confound — TSCI responds to source resolution, not tread wear

*2026-09-11T11:12:04+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** If TSCI measures groove sharpness as documented, then downsampling a photograph without changing the tyre should leave TSCI roughly unchanged. If instead TSCI reflects how much the image was rescaled, it will move systematically with source resolution.

**Data.** Legacy scraped dataset, the 38 images with native width >= 1024 px, each anti-alias downsampled to widths [128, 192, 256, 384, 512, 768, 1024]. Within a family only the resolution differs.

**Method.** The original published pipeline, called through src/stage2_roi.py and src/stage3_tsci.py unmodified: BGR -> grey -> CLAHE(2.0, 8x8) -> Gaussian(5x5) -> centre-band ROI -> resize to 256x128 -> 2-D DFT -> E(r > 0.2*min_dim) / E_total.

**Validation.** Within-subject intervention. No model and no cross-validation: the comparison is between versions of the same photograph, so resolution is the only free variable.

**Result.** TSCI rises monotonically with source resolution. Across the sweep it moves +0.2657 on tyres that are pixel-for-pixel the same subject, while the entire good-versus-worn difference in the dataset is +0.0738 - an artefact 3.6 times larger than the signal it is supposed to be measuring. The mechanism is in the pipeline: resizing every ROI to a fixed 256x128 upsamples a thumbnail, whose upper octaves are already empty after web-scale JPEG compression, while packing a large photograph's genuine detail into the same grid. The documented direction is also wrong: worn tyres in this dataset have higher TSCI (0.4958) than serviceable ones (0.4220), because worn examples happen to come from slightly larger source images.

**Decision.** SUPERSEDED IN PART BY exp004. This experiment's own conclusion was that TSCI should be withdrawn from the production feature set. exp004 re-examined that and found the ratio is not intrinsically invalid: its sensitivity to resolution collapses from 2.5x the class signal to 0.6x once oversampling exceeds about 3x. The defensible statement is therefore narrower than 'TSCI does not work'. It is that TSCI has an unstated precondition - the image must carry genuine detail up to the analysis grid's Nyquist limit - which roughly 78% of the legacy dataset violated, and that its published physical interpretation (decreasing with wear) is contradicted by the data. TSCI therefore remains a candidate feature behind the oversampling floor rather than a discarded one, and earns its place only if the model-comparison experiment shows it contributes. What this experiment does establish unconditionally is that the original pipeline's fixed-size resize was invalid and that the published result was measured through it.

Full record: [`exp001_resolution_confound.json`](exp001_resolution_confound.json)

---

## exp002_scale_invariance — Resolution stability of every candidate descriptor

*2026-09-11T11:01:50+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Descriptors that are dimensionless or purely angular should be insensitive to a change of source resolution, whereas descriptors defined by an absolute spatial-frequency threshold should not be. Resampling above roughly 3x oversampling should suppress the artefact for all of them, because the analysis grid is then filled with genuine detail.

**Data.** Legacy scraped dataset, the 20 images with native width >= 1400 px. Each downsampled to widths [256, 384, 512, 768, 1024, 1400], so every family member shows the identical tyre.

**Method.** tyretread preprocess -> extract_roi -> normalise_scale(256x128, downsample-only) -> spectral + texture descriptors, plus legacy TSCI computed on the un-normalised ROI for comparison.

**Validation.** |drift| / SD, where drift is the change in a descriptor's mean across the sweep and SD is its between-tyre standard deviation at 768 px. Reported both for the full sweep and for the portion at or above 3x oversampling.

**Result.** Scale normalisation works. Once every ROI is resampled downwards onto a common analysis grid and only the part of the sweep at or above 3x oversampling is considered, 46 of 48 descriptors are stable, meaning their pure-resolution drift is under half of their between-tyre spread. The two exceptions are legacy TSCI (0.53) and one LBP bin (0.58), both marginal. Across the unrestricted sweep, which includes the 1x-2x region, many descriptors are not stable - legacy TSCI reaches 1.71 and several LBP bins exceed 1.0 - which is what makes the oversampling floor necessary rather than merely prudent. The theoretical prediction held for the spectral slope, whose drift falls from 1.02 over the full sweep to 0.29 above 3x, consistent with a power-law exponent being preserved under isotropic rescaling. It did not hold as cleanly for the angular statistics: orientation entropy and anisotropy are computed over a fixed band of normalised frequency, and rescaling moves physical content in and out of that band, so their invariance depends on normalisation rather than following from the mathematics alone.

**Decision.** Adopt the scale-invariant spectral descriptors and the multi-angle, multi-distance texture descriptors as the production feature set, with scale normalisation applied before any of them. Set the quality gate's oversampling floor from this evidence rather than by convention: the stability results do not support analysing images below roughly 3x oversampling. orientation_dominant_deg is retained as reported evidence but excluded from the model, because an absolute groove angle measures how the phone was held rather than the state of the tyre. The cost of the floor is quantified separately in exp003.

Full record: [`exp002_scale_invariance.json`](exp002_scale_invariance.json)

---

## exp003_oversampling_floor — Cost of enforcing resolution correctness on the legacy dataset

*2026-09-11T11:01:25+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Enforcing downsample-only scale normalisation requires rejecting images that cannot fill the analysis grid. If the legacy dataset's resolution distribution is adequate, a scientifically defensible floor should retain most of it. If not, the dataset cannot support the corrected pipeline at any grid size.

**Data.** Legacy scraped dataset, all 369 images (234 serviceable / 135 worn).

**Method.** Full tyretread pipeline at each candidate analysis grid, with the quality gate's oversampling floor set to each candidate value. An image is kept only if its ROI can be downsampled onto the grid without upsampling and clears every other quality check.

**Validation.** Descriptive. Counts surviving images and class balance per configuration; no model is fitted, because the question is whether enough usable data exists to fit one.

**Result.** The legacy dataset cannot support a resolution-correct pipeline. At the 3x floor that exp002 shows is needed for descriptor stability, every grid retains at most 80 of 369 images, and the class balance inverts from 63% serviceable to 34-40% - because the worn examples were scraped from systematically larger source images, which is the same confound exp001 identified, now visible as a survival bias. Relaxing the floor to 1x keeps 289 images but reinstates the artefact the correction exists to remove. There is no configuration that is both honest and adequately powered on this data.

**Decision.** The legacy dataset is retired from training. Its remaining roles are to reproduce the original published result and to serve as an out-of-domain robustness check where its resolution permits. Training requires a dataset whose images are large enough that the oversampling floor is never the binding constraint - which is the scientific argument for the Mendeley phone-camera dataset, independent of its size. The production grid is deferred until it can be chosen on data that can actually support the measurement.

Full record: [`exp003_oversampling_floor.json`](exp003_oversampling_floor.json)

---

## exp004_tsci_rehabilitation — TSCI is valid above the oversampling floor and degenerate below it

*2026-09-11T11:11:30+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** TSCI's instability is caused by computing it on images that cannot fill the analysis grid, not by the E_high/E_total ratio itself. If so, its sensitivity to resolution should collapse once oversampling exceeds roughly 3x, and persist below that.

**Data.** Legacy scraped dataset, the 13 images with native width >= 1600 px. Each rendered at source widths chosen to hit oversampling factors [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] against the 256 px analysis grid, so the tyre is constant and only oversampling varies.

**Method.** For each version: preprocess, extract the centre-band ROI, then compute legacy TSCI on the raw ROI (as originally published, including its own internal resize) alongside the scale-invariant descriptors on the normalised ROI.

**Validation.** Within-subject. Spread of the mean descriptor across oversampling factors, split at 3x, compared against the good-versus-worn signal of 0.0738 measured in exp001.

**Result.** The hypothesis holds. Below 3x oversampling, mean TSCI varies by +0.1842 on identical tyres - 2.5 times the good-versus-worn signal, which is the failure exp001 detected. At or above 3x it varies by only +0.0409, i.e. 0.6 times the signal. The ratio E_high/E_total is therefore not intrinsically invalid; it has an unstated precondition, namely that the image actually carries detail up to the analysis grid's Nyquist limit. The legacy dataset violated that precondition for roughly 78% of its images (exp003), which is why TSCI behaved as a resolution proxy there and why its apparent direction inverted.

**Decision.** Revise the decision recorded in exp001. TSCI is not withdrawn on principle; it is reinstated as a *candidate* feature, valid only behind the oversampling floor, and whether it earns a place in the production model is left to the model-comparison experiment on domain-matched data. The original work's error is more precisely stated as a missing precondition and an unvalidated physical interpretation, rather than a worthless feature. The scale-invariant descriptors are retained regardless, because they degrade more gracefully and because orientation statistics answer a question about tread structure that an isotropic energy ratio cannot.

Full record: [`exp004_tsci_rehabilitation.json`](exp004_tsci_rehabilitation.json)

---

## exp005_model_selection_legacy_scraped — Model selection on legacy_scraped

*2026-09-11T11:25:37+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Among lightweight classifiers on engineered texture and frequency features, a regularised linear model will be competitive with or better than an RBF-SVM and tree ensembles, and its probabilities will calibrate well enough to support a principled abstention band.

**Data.** legacy_scraped: 369 images, 82 passing the quality gate (22%). worn=49, serviceable=33.

**Method.** 46 features from tyretread.features (scale-normalised ROI). Each candidate is a scaler-plus-estimator pipeline wrapped in Platt scaling, so scaling and calibration are fitted inside every fold. orientation_dominant_deg is excluded as an artefact of camera pose.

**Validation.** 5-fold x 3 repeats, group-aware (StratifiedGroupKFold), seed 42. Model choice by paired t-test on identical folds. Threshold and abstention band chosen from averaged out-of-fold probabilities, never from in-sample predictions.

**Result.** 

**Decision.** 

Full record: [`exp005_model_selection_legacy_scraped.json`](exp005_model_selection_legacy_scraped.json)

---

## exp006_mendeley_audit — The Mendeley dataset is usable, but not for the reason advertised

*2026-09-11T11:53:47+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** If the Mendeley dataset is what its page describes - 1,854 phone-camera images at a uniform 3000x3000 - then image metadata cannot correlate with the label, the resolution confound found in exp001 becomes impossible, and it is a straightforward replacement for the legacy dataset.

**Data.** Mendeley doi:10.17632/bn7ch8tvyp.1, all 1856 images as distributed; 1735 pass the quality gate.

**Method.** Resolution and file metadata read directly from the images. Confound probes fit a balanced logistic regression on metadata alone, on image features alone, and on both. Label semantics assessed by visual inspection of 40 randomly sampled images, 20 per class.

**Validation.** Two cross-validation schemes on identical data: ordinary repeated stratified 5-fold, and 5-fold grouped by exact pixel resolution. An exact sensor resolution fingerprints one camera in one session, so grouping by it prevents a model from recognising the photo session instead of the tyre.

**Result.** Three of the dataset page's claims are wrong, and the most important one is right anyway. There are 666 distinct resolutions, not one, and not a single image is 3000x3000; 309 images (17%) are small squares between 224 and 600 px, which is the signature of web thumbnails rather than phone originals; and the count is 1856, not 1854. There is a real acquisition confound: metadata alone predicts the label at 0.655 balanced accuracy under ordinary folds, four resolution groups are 100% single-class, and the orientation split is stark - 42% of worn images are landscape against 19% of serviceable ones - so the two classes were photographed in separate sessions. Visual inspection answers the question that matters most: the 'defective' class is dominated by sidewall cracking, splits, perished rubber and bead damage, with many images showing the sidewall rather than the tread at all, and several showing deep, healthy tread. The 'good' class is largely new or nearly-new tyres in retail condition. These labels describe tyre damage, not tread depth. Against that, the signal is genuine. Grouping folds by exact resolution collapses the metadata-only model to 0.534 - chance - while the image features still reach 0.847, and metadata adds only +0.011 on top of them. Trivial appearance features reach only 0.540, so the model is not merely separating clean tyres from dirty ones; the discriminative power sits in micro-texture, with LBP alone at 0.819 and GLCM at 0.768. The quality gate passes 93% of this dataset against 22% of the legacy one.

**Decision.** Adopt the Mendeley dataset for training, and change what the system claims rather than overstating what the data supports. It cannot support a tread-depth or tread-wear claim, because its labels are not about tread depth. It can support a visible tyre-condition screener: worn, cracked or damaged rubber versus rubber in good condition. All reported metrics must come from resolution-grouped folds, which is the conservative estimate, because plain folds are inflated by roughly three points of session fingerprinting. docs/DATA.md is corrected to state the measured facts rather than the dataset page's claims.

Full record: [`exp006_mendeley_audit.json`](exp006_mendeley_audit.json)

---

## exp007_model_selection_mendeley — Model selection on mendeley_tyres

*2026-09-12T22:11:34+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Among lightweight classifiers on engineered texture and frequency features, a regularised linear model will be competitive with or better than an RBF-SVM and tree ensembles, and its probabilities will calibrate well enough to support a principled abstention band.

**Data.** mendeley_tyres: 1856 images, 1695 passing the quality gate (91%). worn=941, serviceable=754.

**Method.** 46 features from tyretread.features (scale-normalised ROI). Each candidate is a scaler-plus-estimator pipeline wrapped in Platt scaling, so scaling and calibration are fitted inside every fold. orientation_dominant_deg is excluded as an artefact of camera pose.

**Validation.** 5-fold x 5 repeats, group-aware (StratifiedGroupKFold), seed 42. Model choice by paired t-test on identical folds. Threshold and abstention band chosen from averaged out-of-fold probabilities, never from in-sample predictions.

**Result.** 

**Decision.** 

Full record: [`exp007_model_selection_mendeley.json`](exp007_model_selection_mendeley.json)

---

## exp008_cross_dataset — The two datasets do not measure the same thing

*2026-09-11T12:06:12+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** If the legacy dataset's 'worn' label and the Mendeley dataset's 'defective' label denote the same physical property, a model trained on one should transfer to the other with only a modest drop. If they denote different properties - tread wear versus rubber damage - transfer should collapse towards chance even though each dataset is separately learnable.

**Data.** Legacy scraped, 82 images passing the quality gate (49 worn). Mendeley, 1735 passing (957 worn). 46 shared features.

**Method.** RBF-SVM (C=10) with Platt scaling, the model exp007 selected, fitted on the whole of one dataset and evaluated on the whole of the other. Repeated with a calibrated logistic regression to check the gap is not specific to one model family. Within-dataset reference points use resolution-grouped cross-validation. A pooled model and a domain classifier are fitted as further checks.

**Validation.** Transfer is a true held-out evaluation: no image from the test dataset is seen during training. Balanced accuracy with 2000-sample percentile bootstrap 95% confidence intervals, which the small legacy test set makes necessary. Within-dataset baselines use folds grouped by exact pixel resolution, so they are the conservative comparison.

**Result.** Transfer degrades sharply in both directions. A model reaching 0.878 balanced accuracy within the Mendeley data scores 0.665 [0.559, 0.765] on the legacy data, and training on legacy gives only 0.595 [0.572, 0.617] on Mendeley against 0.878 within-dataset - a drop of -0.283. Both confidence intervals sit above 0.5, so transfer is weak but not absent: something shared is being learned, just far less than each dataset contains about itself. The calibrated logistic baseline shows the same pattern, so this is a property of the data rather than of one model family. Pooling changes the Mendeley-only result by -0.018 - no gain, while making the training target a blend of two concepts. The domain-separability check is the informative part and it did not go as expected: a classifier distinguishes which dataset an image came from at only 0.627 balanced accuracy. Had that been near 1.0, the transfer failure could be dismissed as ordinary domain shift - the model simply never seeing web thumbnails. It is not. The two image populations are only weakly distinguishable by these texture features, yet a model trained on one is close to useless on the other. That combination points away from appearance and towards the labels: the most consistent explanation is that 'worn tread' and 'defective tyre' name different properties, which is what exp006's visual inspection found directly. Acquisition differences remain a contributing factor and cannot be fully separated out with the data available, but they are no longer the leading explanation.

**Decision.** Do not pool the datasets. They are kept separate, with Mendeley as the training set and the legacy data retained only for reproducing the original published result. No combined label is created, because a pooled label would name a concept neither dataset measures. This result is also the project's honest answer on generalisation: the system is validated within the domain it was trained on, and there is direct evidence it does not yet transfer to a different imaging domain. That is a limitation to state plainly in the README, not one to leave for a reader to discover.

Full record: [`exp008_cross_dataset.json`](exp008_cross_dataset.json)

---

## exp009_tsci_contribution — TSCI is valid but adds no measurable predictive value

*2026-09-11T12:08:28+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Having been rehabilitated in exp004 as valid above the oversampling floor, TSCI should be readmitted to the production feature set only if it measurably improves the model. The null hypothesis is that it adds nothing beyond the scale-invariant spectral descriptors that replaced it.

**Data.** Mendeley, 1735 images passing the quality gate (957 worn / 778 serviceable). Every image is behind the oversampling floor, so TSCI is evaluated only inside the regime exp004 validated.

**Method.** RBF-SVM (C=10) with Platt scaling, the model exp007 selected. The 46-feature production set is compared against the same set plus legacy TSCI, on identical folds. TSCI alone and the scale-invariant spectral descriptors alone are also fitted for context.

**Validation.** 30 resolution-grouped folds, identical across both arms, compared with a paired t-test. Resolution grouping matters here specifically: TSCI's known failure mode is resolution sensitivity, so a fold scheme that let resolution leak could credit TSCI for the confound it is prone to.

**Result.** TSCI adds nothing, and the way it fails is more informative than the headline. Its univariate AUC of 0.627 looks like a real signal, but as a lone feature under resolution-grouped folds it scores 0.508 against a 0.500 floor - indistinguishable from chance. The univariate figure was measured across the whole dataset, where the acquisition confound identified in exp006 is free to contribute; blocking sessions removes it. That is the same failure mode exp001 found on the legacy dataset, reproduced here on independent data: what looks like tread information turns out to be information about the photograph. Added to the 46-feature production set, TSCI changes balanced accuracy by +0.0006 with a paired p of 0.6020 - no measurable improvement. The scale-invariant spectral descriptors that replaced it reach 0.669 on their own, and swapping them out for TSCI costs the full model 0.851 against 0.873, so the replacement was worth making. One further result deserves recording. Worn tyres average 0.6654 against 0.6235 for serviceable ones - again the opposite of the original paper's claim that TSCI decreases with wear. On the legacy dataset that inversion could be attributed to the resolution confound. Here it is reproduced on an independent dataset, entirely inside the oversampling regime exp004 validated, so that explanation no longer applies. The stated physical justification - that shallower grooves attenuate high-frequency energy - is not supported by either dataset.

**Decision.** TSCI is not included in the production classifier. It is retained as a reported diagnostic on the inspection report and as a research feature, where it is genuinely useful: it is cheap, interpretable, and its history in this project is the clearest available illustration of why a feature needs a stated validity regime. Keeping it out of the model while keeping it in the report is the honest resolution - it was never shown to be worthless, only redundant here, and the distinction is recorded rather than flattened. If a future dataset with real tread-depth labels changes that, this experiment is the one to re-run.

Full record: [`exp009_tsci_contribution.json`](exp009_tsci_contribution.json)

---

## exp010_robustness — Does the system stop answering when the photograph stops supporting an answer?

*2026-09-11T14:09:17+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** The quality gate should refuse degradations that destroy the texture the model relies on - blur, underexposure, insufficient resolution - and the abstention band should absorb milder degradations. Degradations that pass both guards while silently flipping the verdict are the dangerous case, because the user receives a confident answer with no signal that anything is wrong.

**Data.** 60 images sampled from the Mendeley set, all of which pass the quality gate undegraded, each put through nine degradations.

**Method.** The full production path: tyretread.imaging.io.load_image followed by tyretread.inspect_image with the served artifact, so EXIF handling and input downscaling match a real upload. Degradations: Gaussian blur at two strengths, under- and over-exposure, an eight-fold downscale standing in for photographing from too far away, heavy JPEG compression, a saturated glare patch, and sensor noise.

**Validation.** Descriptive. Reports the share refused by the gate, sent to inconclusive, and answered confidently, plus the silent-flip rate - the share of images answered confidently both before and after degradation whose verdict changed.

**Result.** The guards work where the degradation destroys texture outright and fail where it does not. Both blur strengths, the eight-fold downscale and the glare patch are refused on 100% of images - the gate's Laplacian-variance, oversampling and saturation checks each do exactly what they were added for. Undegraded images are refused 0% of the time and answered confidently 93% of the time, so the gate is not simply strict. Three real weaknesses show up. Underexposure is the most dangerous: 53% are refused, but 42% are still answered confidently and 20% of those silently change verdict - a user photographing a tyre in a dim garage can get a confident, different answer with nothing to indicate a problem. Overexposure behaves similarly at a lower rate, 12% answered confidently with 14% flipping. Heavy JPEG compression at quality 12 is refused only 7% of the time and answered confidently 77% of the time, and sensor noise is never refused at all - the gate has no check for either, because both preserve the global brightness, contrast and sharpness statistics the gate measures while corrupting the fine texture the model actually uses. The abstention band absorbs some of this - it sends 22% of noisy images to inconclusive against 7% of clean ones - but it was tuned for borderline tyres, not for corrupted images, and it is not sufficient on its own.

**Decision.** Record these as known failure modes in the README rather than quietly tightening thresholds to make the numbers look better. Two follow concretely and are proposed, not implemented: a noise or compression-quality check, since neither is currently detectable by any gate metric; and a re-examination of the exposure thresholds, which were inherited from the project's original cleaning script and are demonstrably too permissive at the dark end. Both need to be validated against the false-refusal rate before adoption - a gate that refuses good photographs is its own failure mode, and the current 0% baseline refusal is worth protecting. This experiment is the regression test for either change.

Full record: [`exp010_robustness.json`](exp010_robustness.json)

---

## exp011_tread_vs_sidewall — Half the accepted images are not tread, and the surface cannot be detected reliably

*2026-09-11T14:10:43+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** The ROI stage cannot distinguish the tread from the sidewall, so a significant share of accepted images are sidewall photographs being assessed as though they were tread. If the surface is detectable reliably, non-tread images can be refused; if it is only partly detectable, refusing would cost more than it gains.

**Data.** 116 images sampled at random from those passing the quality gate on the Mendeley dataset, hand-labelled tread / sidewall / mixed. Labels in data/annotations/surface_labels.json.

**Method.** Balanced logistic regression over the production feature set, and over the 8 interpretable structure features that tyretread.imaging.surface ships. Association between surface and condition label tested with chi-square.

**Validation.** Out-of-fold probabilities averaged over 10 runs of stratified 5-fold. ROC-AUC with a 3000-sample percentile bootstrap 95% confidence interval, which 120 labels make essential.

**Result.** 47% of the images the system accepts are not clean tread - 45 sidewall close-ups and 9 mixed out of 116. Many are photographs of moulded sidewall lettering, which contains no tread at all. The surface is not confounded with the condition label (chi-square p = 1.00); defect prevalence is similar across surfaces, so the model is not covertly learning 'this is a sidewall photograph'. That is the reassuring half. The detector reaches AUC 0.797 [0.698, 0.878] - real signal, but not reliable enough to reject on. At a threshold flagging three-quarters of non-tread images it also flags one genuine tread photograph in five. Given that the production model performs comparably on both surfaces, refusing sidewall images would discard working functionality to enforce a distinction the system cannot make confidently.

**Decision.** Report the surface, do not enforce it. The inspection states which surface it believes it assessed and says when it does not know, using the thresholds 0.35 and 0.6 - the lower one chosen at roughly 5% false flagging of genuine tread. The user-facing copy for a non-tread result states explicitly that the assessment says nothing about remaining tread. This is the honest reading of 'do not pretend the system can identify tread': it neither claims a tread assessment it cannot support, nor throws away a sidewall assessment it can. The detector is fitted on 120 labels and should be refitted on a larger annotated sample before being relied on more heavily. Refusing non-tread images remains a reasonable future option if tread-specific labels ever become available.

Full record: [`exp011_tread_vs_sidewall.json`](exp011_tread_vs_sidewall.json)

---

## exp012_gate_improvements — Closing the quality gate's blind spots at a 1% false-reject cost

*2026-09-11T14:14:30+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Underexposure, sensor noise and heavy compression evade the gate because exposure is measured after CLAHE - which is designed to hide it - and because noise and compression are not measured at all. Measuring exposure on the raw luma, adding dynamic range to separate a dark tyre from a dark photograph, and adding a median-residual noise statistic and a JPEG blockiness ratio should close all three without materially raising the false-reject rate, and specifically without rejecting ordinary phone JPEGs.

**Data.** 400 images sampled at random from those passing the quality gate on the Mendeley dataset, each also evaluated under seven degradations including two ordinary phone-grade JPEG qualities as negative controls.

**Method.** Five statistics computed on the raw luma: mean intensity; 2nd-to-98th percentile dynamic range; fraction of pixels above 224; mean absolute residual after a 3x3 median filter; and the ratio of horizontal luma steps on the JPEG 8-pixel grid to those off it. Thresholds chosen from the measured distribution over real images, then checked against the degradations - not tuned against the degradations.

**Validation.** False-reject rate measured on real images that currently pass. Catch rate measured per degradation. Quality-60 and quality-85 JPEGs act as negative controls: a gate that refuses these would be unusable on phone photographs.

**Result.** All three blind spots close. End to end (exp010), underexposed images refused rise from 53% to 100%, heavy JPEG from 7% to 93%, and sensor noise from 0% to 100%. The silent-flip rate - a confident verdict that changed under degradation with nothing to warn the user - falls from a worst case of 20% to **zero across every degradation tested**. The cost is measured as a fall in the dataset pass rate from 93.5% to 90.7% - about 2.8% of previously-accepted images are now refused, chiefly for being too dark. Measuring false rejection on images that already pass the new gate would be circular and returns nearly zero by construction; the pass-rate change is the figure that means something. The share of clean images answered confidently is unchanged at 93%. The negative controls behave: ordinary quality-85 and quality-60 phone JPEGs are not rejected, because blockiness measures about 1.19 and 1.52 for them against 3.40 at quality 12, and the threshold sits at 2.2. Dynamic range earns its place as a separate check. A well-exposed photograph of a black tyre and an underexposed photograph both have a low mean, but only the underexposed one has a compressed histogram; the 15 darkest real images span a dynamic range of 36-151 against 19-52 for underexposed versions. Tightening the bright-fraction threshold from 0.40 to 0.20 was free: the maximum over 400 real images is 0.197.

**Decision.** Adopt all five checks. The remaining weakness is overexposure, unchanged at 78% refused with 17% still answered confidently - though its silent-flip rate is now zero, so the residual risk is a wrong answer being offered rather than a changed one. Tightening further starts refusing real images and is not justified on this evidence. One consequence worth recording: quality thresholds are stored in the model artifact, so a served model keeps the thresholds it was validated under and changing the code alone does not change inference. The artifact must be rebuilt for a threshold change to take effect - which is the correct behaviour, and was discovered by a threshold change appearing to do nothing.

Full record: [`exp012_gate_improvements.json`](exp012_gate_improvements.json)

---

## exp013_roi_diagnosis — Is the wide-shot failure localisation, or lost information?

*2026-09-12T19:20:35+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** A photograph of a whole wheel degrades the assessment. If the cause is ROI localisation, cropping back to the known tyre rectangle should restore the close-up behaviour. If the cause is lost resolution or domain shift, an oracle crop will not help and no detector would either.

**Data.** 40 Mendeley images that the pipeline handles as close-ups, each composited into a 2.5x larger synthetic background so the tyre occupies roughly a sixth of the frame. The true tyre rectangle is known by construction, so perfect localisation needs no annotation.

**Method.** Three conditions per image: the original close photograph; the wide composite through the production pipeline; and the wide composite cropped by an oracle back to the true tyre rectangle. Compares quality-gate outcome, ROI intersection-over-union with the true box, calibrated probability, and per-feature drift in units of the close-up standard deviation.

**Validation.** Within-subject: every comparison is between conditions derived from the same photograph, so resolution and content are the only variables. The oracle condition is an upper bound on what any localiser could achieve.

**Result.** The failure is dominantly localisation, not lost information. Standing back drives the calibrated probability 0.282 away from the close-up value and drops verdict agreement to 63.6%. Cropping back to the known tyre rectangle - at the reduced resolution that standing back imposes, not the original pixels - restores agreement to 100% and leaves a residual of only 0.055. Perfect localisation therefore recovers about 80% of the damage, and the remaining 20% is the genuine resolution penalty that no detector could undo. The mechanism is visible in the ROI itself: median intersection-over-union between the pipeline's region and the true tyre box is 0.267, and 32 of 40 wide shots fall back to the centre band because no contour passes the acceptance test. The features move accordingly - gradient_mean by -1.07 standard deviations and glcm_contrast_d1 by -0.96 - and both roughly halve under the oracle crop, which is what dilution by background predicts. The quality gate is not a sufficient backstop here: it still accepts 72.5% of wide shots, because a wide shot is sharp, well exposed and high-resolution. It fails none of the checks the gate performs. The other four candidate causes are ruled out. Domain shift and model behaviour cannot explain a failure that an oracle crop repairs completely. Feature extraction is working as specified - the descriptors faithfully describe the region they are given, which is the problem, because that region is mostly not tyre.

**Decision.** Localisation is worth improving, and the improvement has a measurable ceiling: recovering at most the 0.226 of probability drift that the oracle recovers. That is a real target rather than a guess. Two things follow. First, candidate localisers can be ranked on this synthetic set, where ground truth is exact, before anything is built into the product (exp014). Second, whatever the outcome, the quality gate needs a check for how much of the frame is actually tyre - the current gate is blind to a technically excellent photograph of mostly ground. Note the limit of this evidence: composites have a hard rectangular boundary and a uniform synthetic surround, which is easier than a real wheel arch. Ranking here must be confirmed on real photographs before adoption.

Full record: [`exp013_roi_diagnosis.json`](exp013_roi_diagnosis.json)

---

## exp014_localiser_ranking — Ranking simple tread localisers against an exact ground truth

*2026-09-12T19:58:10+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Tread carries dense, dark, directionally coherent texture that its surroundings do not, so a texture-energy search should localise it better than a fixed centre-band crop - and that improvement should show up downstream as a calibrated probability closer to the close-up value, not merely as a better overlap number.

**Data.** 29 Mendeley images composited into constant-size frames with the tyre shrunk 2.5x, so the true tyre rectangle is known exactly. Identical images for every method.

**Method.** Five localisers: the production centre band; the production contour refinement; densest edge energy; edge energy weighted by darkness; and edge energy weighted by darkness and structure-tensor coherence. Region search is exhaustive over a coarse score map via an integral image.

**Validation.** Intersection-over-union and purity against the known box, plus the downstream absolute probability drift from the close-up value. An oracle crop supplies the floor. Ranking only - the synthetic surround is more forgiving than a real wheel arch.

**Result.** 

**Decision.** 

Full record: [`exp014_localiser_ranking.json`](exp014_localiser_ranking.json)

---

## exp015_localiser_gating — Can texture localisation be adopted without harming well-framed photos?

*2026-09-12T21:43:40+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** A texture-energy localiser repairs wide shots but damages close-ups. If its own confidence separates the two framings, it can gate itself and capture the benefit without the cost. If no threshold achieves that, localisation should not be adopted and the correct response to a wide shot is to ask for a closer photograph.

**Data.** 37 Mendeley images, each evaluated twice: as the original close-up and composited into a constant-size frame with the tyre shrunk 2.5x. The close-up probability is the reference for both.

**Method.** texture_energy (the exp014 winner) applied conditionally on its own confidence, falling back to the production centre band below the threshold.

**Validation.** Within-subject across both framings, identical images at every threshold. A threshold is viable only if it cuts the wide-shot flip rate by more than 5 points while leaving the close-up flip rate no more than 2 points worse than doing nothing.

**Result.** No viable threshold exists. The trade is real and it does not go away. Never cropping gives 11.8% verdict flips on close-ups and 54.5% on wide shots; always cropping reverses it almost exactly, to 25.0% and 12.1%. Every threshold between 0.02 and 0.35 behaves identically to always cropping, and nothing in between helps. The reason is instructive, and it is not that the confidence signal is weak. Separability is high - AUC 0.927 - but the statistic *saturates*: wide shots sit at 1.000 with a 10th percentile of 1.000, while close-ups have a median of 0.605 and a 90th percentile of also 1.000. The two distributions are well separated in rank and overlap completely at the ceiling, so there is no operating point that admits wide shots without admitting most close-ups too. A high AUC does not imply a usable threshold when the score is bounded and both classes pile up against the bound. Adopting the localiser would therefore more than double the flip rate for users who framed their photograph correctly - the majority - in order to help those who did not.

**Decision.** Do not adopt texture localisation as a localiser. The measured cost to the common case exceeds the benefit to the uncommon one, and the confidence signal cannot separate them at a usable operating point. The useful finding is what the same signal is good *for*. Detecting that the tyre occupies little of the frame is a far easier problem than localising it precisely, and AUC 0.927 says that detection is reliable. So the texture statistic should become a **quality-gate check** - 'the tyre does not fill enough of this photograph, move closer' - rather than a crop. That is option D: abstain and ask for a better photograph, using the localisation work as the detector that makes the abstention possible. exp016 measures that check directly before anything ships.

Full record: [`exp015_localiser_gating.json`](exp015_localiser_gating.json)

---

## exp016_coverage_check — Detecting a too-distant photograph, instead of trying to crop one

*2026-09-12T21:46:08+00:00 · commit `d839469` · working tree dirty*

**Hypothesis.** Texture search is a poor localiser but the underlying signal separates framings well. Reformulated as a proportion - how much of the frame carries tyre-like texture - it should support a quality-gate check that refuses wide shots with 'move closer' at an acceptable false-refusal rate on well-framed photographs.

**Data.** 60 Mendeley images as close-ups, each also composited at [1.6, 2.0, 2.5, 3.2]x shrink into a constant-size frame, so the check is characterised across a range of distances rather than one.

**Method.** subject_coverage: fraction of coarse cells whose edge energy, weighted towards dark regions, reaches 40% of the frame's 95th-percentile score. A high percentile rather than the maximum, so one highlight cannot set the scale.

**Validation.** Within-subject: every wide composite derives from a close-up in the same set. ROC-AUC per zoom level, plus a threshold sweep reporting false refusal on close-ups against catch rate at each distance.

**Result.** The reformulation works where the localiser did not. As a proportion the signal uses its whole range instead of saturating: close-ups have a median coverage of 0.421 with a 10th percentile of 0.307, while wide shots sit between 0.12 and 0.22. ROC-AUC against close-ups is 0.95 to 0.99 across 1.6x to 2.5x. That translates into a usable operating point, which is what exp015 could not find. Refusing below 0.20 rejects **none** of the close-ups in this set while catching 90% of 2.5x wide shots and 82% at 2.0x; moving to 0.25 costs 3.3% false refusal and catches 97%. The contrast with the localiser is the whole point - the same underlying measurement, asked a question it can answer. One irregularity is worth recording rather than smoothing over: catch rate falls at 3.2x (70%) relative to 2.5x (90%), and the 90th percentile of coverage rises to 0.366. At extreme shrink the tyre is small enough that the synthetic background begins to dominate the 95th-percentile normalisation, so the statistic is measured against a different scale. The check is therefore characterised for moderate framing errors, which is the realistic case, and is not claimed to degrade monotonically forever.

**Decision.** Adopt subject_coverage as a quality-gate check with a refusal threshold of 0.20, not as a localiser and not as a crop. This is option D: when the tyre does not fill enough of the frame, say so and ask for a closer photograph, rather than cropping badly or analysing mostly background. It also completes the live capture guidance: the same statistic can drive a 'move closer' hint before the shutter, so most users never reach the refusal. Not yet implemented in production - reported for approval first. Two caveats bound the claim. All of this is measured on synthetic composites with a hard rectangular boundary and a uniform surround, which is easier than a real wheel arch; the threshold needs confirming on real wide-shot photographs before it is trusted. And the check detects framing, not content: it cannot tell a distant tyre from a close-up of something that is not a tyre at all.

Full record: [`exp016_coverage_check.json`](exp016_coverage_check.json)
