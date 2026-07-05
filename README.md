# JPEG AI as a Threat to Deepfake Detection

**Computer Vision** — Sapienza University of Rome

Studying how JPEG AI Compression degrades Deepfake Detectors, and how to make Detectors more robust to compression.

## Group

- Enrico Bosco — 1956727 — bosco.1956727@studenti.uniroma1.it
- Gabriel Di Fazio — 1941485 — difazio.1941485@studenti.uniroma1.it

**Dataset and Detectors**: https://github.com/PeterWang512/CNNDetection. Notice that for both test dataset and fine-tuning train dataset we decided to use the "test set" of ProGAN, obviously taking disjoint images to not have any leakage. We did it because the real train dataset size was around 70GB and it was simpler to use the same source that contained enough images to split it again in the new train and the new test datasets.

**Mitigation Strategies**:
1. Fine-Tuning (Augmentation): We retrained the detector on a mix of 50% original and 50% JPEG-AI-compressed images with specific couples of bpp ([0.12, 0.25], [0.50, 0.75]), in order to have a specialized detector in lower and higher bpp.
2. Fine-Tuning (Consistency): We retrained on original-compressed pairs with an added invariance loss MSE(σ(orig), σ(comp)) that explicitly forces the detector to output the same score for an image and its compressed version.
3. Sharpening: A test-time, training-free preprocessing step that applies an unsharp mask to each image before detection, aiming to restore the high-frequency cues that JPEG AI smooths away.


## Notebook structure and pipeline execution

The notebook follows the required structure (**Imports · Globals · Utils · Data · Network · Train · Evaluation**). Here is what each section does, in execution order.

### Imports & Globals
- **Imports** — PyTorch / torchvision, scikit-learn (metrics), PIL, matplotlib, and mounts Google Drive.
- **Globals** — all paths and constants in one place: `DATASET_PATH` (test set), `TRAIN_DATASET` (fine-tuning set), `PROGAN_ROOT` (path of the downloaded ProGAN source), `JPEGAI_DIR` (path of JPEG AI Reference Software), `WEIGHTS_DIR` (path of Baseline Detectors), the target bitrates `BPP_LIST = [0.12, 0.25, 0.50, 0.75]`, and the random seed.

### Utils
Support functions reused later:
- `compute_metrics` — AUC, AP, Accuracy, Balanced Acc, FPR, FNR, F1, EER from labels and P(fake) scores.
- Frequency helpers (`load_gray`, `residual`, `avg_fft_residual`, `radial_psd`) — noise-residual spectra for the forensic analysis.
- `scores_with(predict_fn, suffix)` — runs any predictor over one condition (real + fake folders) and returns labels + scores.
- `sharpen` — unsharp-mask preprocessing (mitigation strategy 3).

### Data — reproducible compression pipeline
These cells are only needed to (re)generate the compressed dataset. If it already exists on Drive, evaluation needs only Imports / Globals / Network / Evaluation.

1. **Download the ProGAN test set** from CNNDetection (`https://github.com/PeterWang512/CNNDetection`): real/fake 256×256 images.
2. **Build the training split** (`build_disjoint_train`) — selects images *disjoint* from the test set (no train/test leakage) into `TRAIN_DATASET`.
3. **JPEG AI setup** — installs Miniconda and clones/builds the official JPEG AI reference software.
4. **Compression functions** — `build_split_by_marker` (builds the test split) and `compress_dataset_jpegai` (encodes each image at every target bitrate with the JPEG AI encoder).
5. **Compress test set** and **compress training set** — create the `{real,fake}_bpp{b}` folders next to `{real,fake}_original`.
6. **Paired datasets** — `PairedBPPDataset` (50% original / 50% compressed, for Fine-Tuning (Augmentation)) and `PairedConsistencyDataset` (returns aligned original↔compressed pairs, for the Fine-Tuning (Consistency)).
7. **`manifest.csv`** — one row per image (`class, label, condition, bpp, path`) for controlled experiments.

### Network — the deepfake detector
- **Download CNNDetection weights** (Wang et al. 2020): two variants, `prob0.1` (more vulnerable to compression) and `prob0.5` (more robust).
- **Load the detector** into `cnn` (ResNet50, single logit → P(fake)); `cnndet_predict(path)` applies the exact CNNDetection preprocessing (center-crop 224, no resize) and returns P(fake). Weights are loaded **once** into `cnn` and reused for every prediction.
- **`load_base`** — makes fresh copies of the base detector for the mitigation experts (a *low* and a *high* bitrate expert).

### Train — mitigation experts
Adapts the detector to JPEG-AI-compressed content with two supervised strategies (each producing a low + high expert):
- **Fine-Tuning (Augmentation)** — trained on `PairedBPPDataset` (compression augmentation).
- **Fine-Tuning (Consistency)** — same setup plus an invariance loss `BCE(orig) + BCE(comp) + λ·MSE(σ(orig), σ(comp))` on the paired data.

### Evaluation
- **Degradation curve** — `compute_metrics` on `original` and every bitrate, then a plot of the metrics vs bitrate. To evaluate a mitigated model, point `cnn` at the expert and re-run.
- **Sharpening comparison** — base vs sharpened predictions, per bitrate.
- **Frequency-domain forensic analysis** — average FFT spectra, radial power spectra, DCT coefficient distributions, and difference spectra between original/compressed and real/fake, explaining *why* the accuracy drops.


## How to run

The notebook runs on **Google Colab** with a **GPU** runtime.

1. Run **Imports** and **Globals** and **Utils**.
2. **Data** (see above)
3. **Network** - Run first cell to load baseline models. In the second you can select the detector variant with `DETECTOR_VARIANT` and load the model you want to evaluate. In the third you can select the baseline variant with `BASE` and load the model you want to train.
4. **Train** - First cell (Fine-Tuning (Augmentation)), Second cell (Fine-Tuning (Consistency)).
5. **Evaluation** — reproduces the degradation curve, the sharpening comparison, and the frequency-domain analysis. To evaluate a model, set DETECTOR_VARIANT. Run the first cell to have standard metrics result. Run the second for sharpening. The others are for forensic analysis.


