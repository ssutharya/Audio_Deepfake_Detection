# MFAAN-Deepfake-Detection

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange) 

A Multilingual Audio Deepfake Detection system using the **Multi-Feature Audio Authenticity Network (MFAAN)**, achieving **91.76% accuracy** and **6.89% EER** on the MLADDC T2 dataset. MFAAN leverages MFCC, LFCC, and Chroma-STFT features to detect fake audio across 14 languages, outperforming the baseline (68.44% accuracy, 40.9% EER).

## Project Overview

This repository implements MFAAN, a convolutional neural network (CNN) designed for audio deepfake detection. It processes **Mel-Frequency Cepstral Coefficients (MFCC)**, **Linear-Frequency Cepstral Coefficients (LFCC)**, and **Chroma Short-Time Fourier Transform (Chroma-STFT)** features through parallel CNN paths, fusing them for robust classification. The model is trained and fine-tuned on the **Fake-or-Real (FoR)**, **In-the-Wild**, and **MLADDC T2** datasets, addressing multilingual and modern GAN-generated fakes (e.g., HiFi-GAN, BigVGAN).

### Key Features
- **Multi-Feature Fusion**: Combines 32-channel MFCC, 32-channel LFCC, and 12-channel Chroma-STFT for comprehensive audio analysis.
- **Multilingual Detection**: Optimized for MLADDC’s 14 languages, capturing diverse phonetic and tonal patterns.
- **High Performance**: Achieves 98.74% accuracy on FoR/In-the-Wild and 91.76% on MLADDC T2, with a 6.89% EER.
- **Efficient Design**: Lightweight model with 206,066 parameters (0.79 MB).

### Datasets
- **Fake-or-Real (FoR)**: 141,349 samples (84,756 real, 56,593 fake), English-centric, using older TTS/VC methods.
- **In-the-Wild**: 31,779 samples (19,963 real, 11,816 fake), 58 speakers, with advanced TTS/VC (e.g., Tacotron 2).
- **MLADDC T2**: 168,000 samples, 14 languages, featuring modern GAN-based fakes.

## Installation

### Prerequisites
- Python 3.11
- NVIDIA GPU (e.g., Tesla T4) with CUDA support
- Kaggle account for dataset access

### Dependencies
Install required packages using:
```bash
pip install kaggle numpy pandas librosa torchaudio torch torchvision scikit-learn tqdm matplotlib seaborn
```

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ssutharya/Audio-Deepfake-Detection.git
   cd Audio-Deepfake-Detection
   ```

2. **Download Datasets**:
   - Obtain FoR, In-the-Wild, and MLADDC datasets from Kaggle:
     - [Fake-or-Real](https://www.kaggle.com/datasets/4555568/the-fake-or-real-dataset)
     - [In-the-Wild](https://www.kaggle.com/datasets/4836275/in-the-wild-audio-deepfake)
     - [MLADDC](https://www.kaggle.com/datasets/6013837/mladdc-dataset)
   - Place datasets in `/kaggle/input/` or update `DATASET_PATHS` in the notebook.

3. **Prepare Environment**:
   - Ensure Kaggle API is configured for dataset access (see [Kaggle API docs](https://www.kaggle.com/docs/api)).
   - Run on a GPU-enabled environment (e.g., Kaggle, Google Colab).

## Usage

### Running the Notebook
The main implementation is in `audio-deepfake-detection.ipynb`. Follow these steps:

1. **Preprocess Data**:
   - Run the preprocessing cells to extract MFCC, LFCC, and Chroma-STFT features from audio files.
   - Features are saved in chunks to `/kaggle/working/preprocessed_chunks_268` and `/kaggle/working/preprocessed_t2_{train/val/test_chunks}`.

2. **Train/Fine-Tune Model**:
   - Fine-tune MFAAN on MLADDC T2 using the provided cell.
   - The model is trained for 20 epochs, saving the best weights (`mfaan_t2_finetuned.pth`) based on validation EER.

3. **Test Model**:
   - Evaluate on the MLADDC T2 test set (16,800 samples).
   - Outputs accuracy, EER, and per-class accuracy (Real: 95.86%, Fake: 89.71%).

### Example Output
```plaintext
Final T2 Test Accuracy: 91.76%
Final T2 Test EER: 6.89%
Per-class accuracy (Real, Fake): [95.86 89.71]
Baseline Accuracy (T2): 68.44%, Baseline EER: 40.90%
Model vs. Baseline:
Accuracy: Outperforms (Model: 91.76%, Baseline: 68.44%)
EER: Outperforms (Model: 6.89%, Baseline: 40.90%)
```

## Project Structure
- `deepfake-outputs.ipynb`: Main notebook for preprocessing, training, and testing.
- `/kaggle/working/preprocessed_chunks_268/`: Preprocessed FoR/In-the-Wild features.
- `/kaggle/working/preprocessed_t2_{train/val/test_chunks}/`: Preprocessed MLADDC T2 features.
- `mfaan_t2_finetuned.pth`: Best fine-tuned model weights.

## Novelty and Contributions
- **Multilingual Optimization**: Tailored for MLADDC’s 14 languages, addressing phonetic and tonal diversity.
- **Feature Engineering**: Uses 32-channel MFCC/LFCC (vs. 40 in literature) for reduced noise and faster training.
- **Robustness to Modern Fakes**: Handles advanced GAN-based fakes, unlike baselines trained on older TTS/VC.
- **Future Scope**: Proposes attention mechanisms for half-truth detection (T3 task).

## Results
| Dataset            | Accuracy (%) | EER (%) |
|--------------------|--------------|---------|
| FoR + In-the-Wild  | 98.74        | 1.26    |
| MLADDC T2 (Test)   | 91.76        | 6.89    |
| Baseline (ResNet-50)| 68.44        | 40.90   |

## Limitations
- Requires fine-tuning for MLADDC (33.83% accuracy without).
- Attention mechanisms for T3 half-truth detection are future work.
- English-centric pre-training (FoR/In-the-Wild) may limit generalization.

## Future Work
- Implement attention mechanisms for T3 half-truth detection.
- Explore transformer-based architectures for improved temporal modeling.
- Expand to more languages and newer generative models.

## Acknowledgments
- Datasets: FoR, In-the-Wild, MLADDC
- Libraries: PyTorch, torchaudio, librosa, scikit-learn
- Hardware: Kaggle GPU (Tesla T4)

For questions or contributions, open an issue or contact [sutharya8@gmail.com].
