# SVD-Guided Signal Denoising (Noise2Semi_clean Framework)

A PyTorch-based signal denoising framework using SVD (Singular Value Decomposition) to generate semi-clean training targets. This approach eliminates the need for ground truth clean signals during training.

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
```

---

## Training & Inference

### Training
```bash
python noise_semi_clean.py
```

This script:
1. Loads signals from `Data/Signalstandford.mat`
2. Adds Gaussian noise at the specified SNR level
3. Applies STFT to convert to time-frequency domain
4. Generates semi-clean targets using SVD (no ground truth needed)
5. Trains a U-Net model to denoise STFT coefficients
6. Saves the best model to `checkpoints/best_model.pth`

### Inference
```bash
python predictions.py
```

This script:
1. Loads the trained model from `checkpoints/best_model.pth`
2. Runs denoising on the test set
3. Calculates SNR improvement metrics

---

## Hyperparameters

Key hyperparameters are defined in `noise_semi_clean.py` under `CONFIG`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_snr_db` | 0 | Target SNR for adding noise (dB). Lower = more noise. |
| `energy_threshold` | 0.7 | SVD energy threshold for semi-clean generation. Lower = more aggressive denoising. |

**Recommended settings by noise level:**

| Input SNR | `energy_threshold` | Notes |
|-----------|-------------------|-------|
| 5 dB | 0.9 | Light noise - keep more signal components |
| 0 dB | 0.7-0.8 | Medium noise |
| -5 dB | 0.6-0.7 | Heavy noise - more aggressive filtering |

### Loss Function

The loss function in `model_torch.py` uses a weighted combination:
```python
loss = mse_weight * MSE + l1_weight * L1_regularization
```

Default: `mse_weight=0.9`, `l1_weight=0.1`. Adjust these based on your dataset characteristics.

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
├── noise_semi_clean.py      # Main training pipeline
├── predictions.py           # Inference and evaluation
├── model_torch.py           # U-Net architecture (PyTorch)
├── iddm_model_torch.py      # IDDM architecture (PyTorch)
├── iddm_losses_torch.py     # IDDM loss functions
├── train_iddm_torch.py      # IDDM benchmark training
├── inference_iddm_torch.py  # IDDM benchmark inference
├── requirements.txt         # Dependencies
├── Data/                    # Signal data
└── checkpoints/             # Saved models
```

---

## Model Architecture

The U-Net model follows the architecture described in our paper:

**Encoder:** 4 levels (64 → 128 → 256 → 256 filters) + Bottleneck (512)  
**Decoder:** 4 levels with skip connections + Dropout  
**Activation:** LeakyReLU (0.1)  
**Output:** Linear activation for regression

---

## Citation

If you use this code, please cite our paper.
