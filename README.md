# Audio Deepfake Detection with Half-Truth Localization

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)

Detection and temporal localization of audio deepfakes - including **half-truth audio** (real recordings with a spliced synthetic segment) - using two architectures evaluated on the [MLADDC](https://openreview.net/forum?id=ic3HvoOTeU) corpus across 20 languages.

**Semester 6:** Re-implemented MFAAN as a binary detection baseline, pretrained on FoR + In-the-Wild, fine-tuned on MLADDC T2 - 91.76% accuracy, 6.89% EER.

**Semester 8:** Designed CAFNet (Cross-Attentive Feature Network) - a unified three-class model (real / fully-fake / half-truth) with temporal regression of splice boundaries. Re-implemented MFAAN fresh on MLADDC T2 directly for a clean baseline comparison - 96.37% accuracy, 2.21% EER. CAFNet achieves 92.71% three-class accuracy, macro AUC 0.9910, and localizes splice boundaries with 0.075s MAE.

---

## Results Summary

### Binary Detection - MLADDC T2 (14 languages)

| Model | Accuracy (%) | EER (%) |
|---|---|---|
| MLADDC Baseline (LFCC + CNN) | 68.44 | 40.90 |
| MFAAN - S6 (pretrained + fine-tuned) | 91.76 | 6.89 |
| MFAAN - S8 (trained directly on T2) | 96.37 | 2.21 |
| CAFNet T2 binary | 96.76 | 3.20 |

### Three-Class Detection - MLADDC T2 + T3

| Metric | Value |
|---|---|
| Overall Accuracy | 92.71% |
| Macro AUC (OvR) | 0.9910 |
| EER (real vs. non-real) | 6.07% |
| Real F1 | 0.842 |
| Fake F1 | 0.971 |
| Half-Truth F1 | 0.930 |

### Temporal Localization - Half-Truth Test Set (16,000 samples)

| Boundary | MAE (s) | Median (s) | p90 (s) |
|---|---|---|---|
| Start | 0.083 | 0.060 | 0.153 |
| End | 0.068 | 0.040 | 0.135 |
| Overall | 0.075 | 0.052 | 0.131 |

84.7% of predictions fall within 0.1s of the true boundary. 96.6% within 0.25s.

### Cross-Dataset Generalization

| Dataset | MLADDC-only AUC | Pretrained AUC | Post Fine-Tune AUC |
|---|---|---|---|
| WaveFake | 0.4948 | 1.0000 | 0.5291 |
| ASVspoof 2019 LA | 0.5042 | 0.9289 | 0.3136 |
| FoR | 0.9289 | 0.9908 | 0.0503 |

Pretraining on FoR-norm + WaveFake + ASVspoof 2019 LA substantially improves cross-dataset generalization. Fine-tuning on MLADDC recovers task accuracy (90.67%) at the cost of catastrophic forgetting of pretrained representations.

---

## Architectures

### MFAAN - Multi-Feature Audio Authenticity Network

Re-implementation of [Krishnan & Krishnan (2023)](https://doi.org/10.1109/ICSC60394.2023.10441405) as a binary detection baseline.

Three parallel 2D-CNN paths - one each for MFCC, LFCC, and Chroma-STFT - each applying two Conv2D → ReLU → Dropout → MaxPool2D blocks. Outputs are concatenated and passed through a dense classification head. 322,562 parameters.

### CAFNet - Cross-Attentive Feature Network

Primary contribution of Semester 8. Unified architecture for three-class detection and temporal localization. 576,414 parameters, CPU-deployable.

**EnhancedPath (×3):** Each feature processed as a 1D temporal signal through two depthwise-separable conv blocks (→64, →128 channels), a lightweight self-attention module (residual-scaled, init 0), and MaxPool1d(2). Output: (128, T/2) per path.

**CrossAttnFusion:** MFCC sequence as query; concatenated LFCC + Chroma as key-value. 8-head MultiheadAttention (dim 128) with a learned gating mechanism that weights the three mean-pooled path representations before final Linear(128, 128) projection.

**Output heads:**
- Main classification head: Linear → BN → ReLU → Dropout (×2) → Linear(128, 3)
- Auxiliary head: Linear(128, 3) for deep supervision (weighted 0.4 in loss)
- Temporal head: pre-pooling concatenation (384, T/2) → 2-layer BiLSTM (64 units/direction) → Linear(128, 2) → Sigmoid → normalized [start, end] in [0, 1]

**Training loss:** `L = Lcls + 0.4·Laux + 0.3·Ltemp` where Ltemp (MSE on boundary predictions) is computed only for half-truth samples.

---

## Datasets

| Dataset | Samples | Description |
|---|---|---|
| MLADDC T2 | 168,000 | 14 international languages, binary real vs. fully-fake (HiFi-GAN + BigVGAN) |
| MLADDC T3 | 160,000 | Half-truth audio across 20 languages - ~1s synthetic splice at random offset |
| FoR-norm | ~69,000 | English TTS, used for pretraining |
| WaveFake | ~92,000 | 6 neural vocoders on LJSpeech, used for pretraining |
| ASVspoof 2019 LA | ~121,000 | 19 TTS/VC systems, used for pretraining |
| In-the-Wild | 31,779 | 58 public figures, used for zero-shot evaluation only |

All audio resampled to 16kHz mono, padded/trimmed to 4 seconds.

---

## Features

Three handcrafted spectral features extracted in parallel from each audio segment:

- **MFCC** - 40 coefficients, FFT 512, hop 256 → (40, 251). Captures timbral texture on the Mel scale.
- **LFCC** - 40 coefficients, custom linear filterbank (128 triangular filters, 0–8kHz), DCT → (40, 251). Sensitive to high-frequency vocoder artifacts.
- **Chroma-STFT** - 12 pitch-class bins → (12, 251). Captures harmonic structure.

---

## Installation

### Prerequisites
- Python 3.11
- GPU recommended (trained on Kaggle Tesla T4); inference runs on CPU

### Dependencies
```bash
pip install numpy pandas librosa torchaudio torch torchvision scikit-learn tqdm matplotlib seaborn
```

### Setup
```bash
git clone https://github.com/ssutharya/Audio_Deepfake_Detection.git
cd Audio_Deepfake_Detection
```

Datasets available on Kaggle:
- [MLADDC](https://www.kaggle.com/datasets/6013837/mladdc-dataset)
- [Fake-or-Real](https://www.kaggle.com/datasets/4555568/the-fake-or-real-dataset)
- [In-the-Wild](https://www.kaggle.com/datasets/4836275/in-the-wild-audio-deepfake)

Place datasets in `/kaggle/input/` or update dataset paths in the notebooks.

---

## Project Structure

```
Audio_Deepfake_Detection/
├── sem6/
│   └── audio-deepfake-detection.ipynb   # MFAAN: pretraining + fine-tuning on MLADDC T2
├── sem8/
│   ├── add_clean.ipynb                  # MFAAN re-implementation, direct T2 training
│   └── add_finetune.ipynb               # CAFNet: three-class + temporal localization
├── models/
│   ├── mfaan_t2.pth           
│   └── cafnet_unified.pth                  # CAFNet best checkpoint (epoch 2 by val loss)
└── README.md
```

---

## Limitations and Future Work

- **Fixed 4-second window:** Half-truth samples are 75% real audio by duration. 1,426 of 16,000 half-truth test samples are misclassified as real due to the splice being acoustically dominated by surrounding genuine speech. A sliding window or two-stage detect-then-localize approach would reduce this.
- **Fine-tuning erases generalization:** Despite a reduced backbone learning rate (1e-5), fine-tuning on MLADDC overwrites pretrained representations. Freezing the backbone during fine-tuning is a direction for future work.
- **English-centric pretraining corpora:** FoR, WaveFake, and ASVspoof are English-only. The synthesis models in MLADDC (HiFi-GAN, BigVGAN) were also pretrained on English data, meaning non-English deepfakes carry English acoustic priors - a known limitation of the benchmark itself.
- **Out of scope:** End-to-end waveform models, self-supervised features (wav2vec 2.0, HuBERT), real-time streaming inference.

---

## Acknowledgments

Datasets: MLADDC (Shah et al., NeurIPS 2024 Workshop), FoR (Reimao & Tzerpos, 2019), WaveFake (Frank & Schönherr, 2021), ASVspoof 2019 (Yamagishi et al., 2021), In-the-Wild (Müller et al., 2022).

Base architecture: MFAAN (Krishnan & Krishnan, ICSC 2023).

Hardware: Kaggle GPU (Tesla T4).
