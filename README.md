# SVD-Guided Signal Denoising (Noise2Semi_Clean Framework)

A PyTorch-based signal denoising framework using SVD (Singular Value Decomposition) to generate semi-clean training targets. This approach eliminates the need for ground truth clean signals during training.

The framework supports **two time-frequency representations**:
- **STFT** (Short-Time Fourier Transform) — fixed-resolution spectrogram
- **DWT** (Discrete Wavelet Transform) — multi-resolution wavelet decomposition

Both pipelines use the same U-Net architecture and SVD-based semi-clean target generation, demonstrating that the approach generalizes across different signal representations.

---

## Quick Start

### 1. Create Environment
```bash
conda create -n noisy_semi_clean python==3.9
conda activate noisy_semi_clean
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install PyWavelets  # Required for wavelet experiment
```

---

## Experiments

### Experiment 1: STFT + Impulsive-Gaussian Noise

#### Training
```bash
python noise_semi_clean_impulsive_gaussian.py
```

This script:
1. Loads signals from `Data/Signalstandford.mat`
2. Adds mixed impulsive + Gaussian noise at **-5, 0, and +5 dB** SNR
3. Applies STFT to convert to time-frequency domain (complex 2-channel)
4. Generates semi-clean targets using SVD subspace projection
5. Trains a separate U-Net model for each SNR level
6. Saves checkpoints to `checkpoints/best_model_impulsive_gaussian_{snr}dB.pth`

#### Inference
```bash
python predictions_impulsive_gaussian.py
```

This script:
1. Loads trained models for all three SNR levels
2. Runs denoising on the test set via ISTFT reconstruction
3. Calculates SNR improvement metrics across all methods
4. Generates time-domain and STFT spectrogram plots (IEEE journal style)

---

### Experiment 2: Wavelet (DWT) + Impulsive-Gaussian Noise

#### Training
```bash
python noise_semi_clean_wavelet.py
```

This script:
1. Loads signals from `Data/Signalstandford.mat`
2. Adds mixed impulsive + Gaussian noise at **-5, 0, and +5 dB** SNR
3. Applies DWT (Daubechies wavelet) and reshapes coefficients into square 2D matrices
4. Generates semi-clean targets using SVD subspace projection
5. Trains a U-Net model (1-channel input) for each SNR level
6. Saves checkpoints to `checkpoints/best_model_wavelet_{snr}dB.pth`

#### Inference
```bash
python predictions_wavelet.py
```

This script:
1. Loads trained wavelet models for all three SNR levels
2. Runs denoising and reconstructs via inverse DWT
3. Calculates SNR improvement metrics
4. Generates time-domain, STFT spectrogram, and CWT scalogram plots

---

### Experiment 0: Gaussian Noise (Baseline)

#### Training
```bash
python noise_semi_clean.py
```

#### Inference
```bash
python predictions.py
```

---

## Noise Model

The mixed noise model combines two components:

- **Gaussian component**: Additive white Gaussian noise (AWGN)
- **Impulsive component**: Sparse, high-amplitude spikes modeled as Bernoulli-Gaussian noise

The total noise power is calibrated to the target SNR and split equally (50/50) between the two components. See `noise_semi_clean_impulsive_gaussian.py` for implementation details.

---

## Hyperparameters

Key hyperparameters are defined in each pipeline's `CONFIG` dictionary. Per-SNR tuning is used for optimal performance:

### STFT Pipeline (`noise_semi_clean_impulsive_gaussian.py`)

| Target SNR | `energy_threshold` | `l1_weight` | `mse_weight` |
|---|---|---|---|
| -5 dB | 0.7 | 0.15 | 0.85 |
| 0 dB | 0.7 | 0.10 | 0.90 |
| +5 dB | 0.9 | 0.01 | 0.99 |

### Wavelet Pipeline (`noise_semi_clean_wavelet.py`)

| Target SNR | `energy_threshold` | `l1_weight` | `mse_weight` |
|---|---|---|---|
| -5 dB | 0.45 | 0.30 | 0.70 |
| 0 dB | 0.60 | 0.25 | 0.75 |
| +5 dB | 0.80 | 0.10 | 0.90 |

### Loss Function

The composite loss function combines MSE and L1 regularization:
```
loss = mse_weight × MSE + l1_weight × L1_regularization
```

---

## IDDM Benchmark

For benchmarking against IDDM (Iterative Diffusion-based Denoising Model):

### Training
```bash
python train_iddm_torch.py
```

### Inference
```bash
python inference_iddm_torch.py --checkpoint checkpoints/iddm_torch/best.pth --steps 3
```

**Arguments:**
- `--checkpoint`: Path to model weights
- `--steps`: Number of backward denoising steps (iterative refinement)
- `--alpha`: Blending factor (0.0=no change, 1.0=full denoising)
- `--batch_size`: Inference batch size

**Note:** The original IDDM paper uses a self-supervised approach. In our implementation, we train the IDDM architecture using SVD-generated semi-clean targets for stable training (see `train_iddm_torch.py` docstring for details).

---

## Project Structure

```
├── noise_semi_clean.py                    # Gaussian noise training pipeline
├── noise_semi_clean_impulsive_gaussian.py # Impulsive+Gaussian noise (STFT) pipeline
├── noise_semi_clean_wavelet.py            # Impulsive+Gaussian noise (Wavelet) pipeline
├── predictions.py                         # Gaussian noise inference
├── predictions_impulsive_gaussian.py      # Impulsive+Gaussian (STFT) inference + plots
├── predictions_wavelet.py                 # Wavelet inference + plots + CWT scalograms
├── model_torch.py                         # U-Net architecture (PyTorch, any input size)
├── iddm_model_torch.py                    # IDDM architecture (PyTorch)
├── iddm_losses_torch.py                   # IDDM loss functions
├── train_iddm_torch.py                    # IDDM benchmark training
├── inference_iddm_torch.py                # IDDM benchmark inference
├── requirements.txt                       # Dependencies
├── Data/                                  # Signal data (Signalstandford.mat)
├── checkpoints/                           # Saved model weights
├── plots_impulsive_gaussian/              # STFT experiment plots
└── plots_wavelet/                         # Wavelet experiment plots
```

---

## Model Architecture

The U-Net model follows the architecture described in our paper:

**Encoder:** 4 levels (64 → 128 → 256 → 256 filters) + Bottleneck (512)  
**Decoder:** 4 levels with skip connections + Dropout  
**Activation:** LeakyReLU (0.1)  
**Output:** Linear activation for regression  
**Input flexibility:** Accepts any square input size via dynamic padding (supports both 78×78 STFT and 79×79 wavelet matrices)

---

## Citation

If you use this code, please cite our paper.
