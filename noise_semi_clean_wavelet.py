"""
Signal Denoising Pipeline using SVD-based Semi-Clean Generation
(Impulsive + Gaussian Mixed Noise — WAVELET TRANSFORM Experiment)

Pipeline Overview:
1. Load raw signals from Stanford dataset
2. Scale signals using StandardScaler
3. Add MIXED noise (Impulsive + Gaussian) at target SNR level
4. Apply Discrete Wavelet Transform (DWT) to both clean and noisy signals
5. Reshape wavelet coefficients into square 2D matrices for SVD
6. Split data into train/val/test sets
7. Generate Semi-Clean versions using SVD algorithm (projection-based denoising)
8. Train U-Net model (1-channel input) to learn noisy -> semi-clean mapping
9. Predict denoised signals

Key Difference from STFT Version:
- DWT replaces STFT as the time-frequency representation
- DWT coefficients are REAL (not complex) -> U-Net uses 1 input/output channel
- Inverse DWT (IDWT) is used for reconstruction instead of ISTFT
- Coefficients are reshaped into a square 2D matrix for SVD compatibility

Noise Model:
- Mixed noise = Gaussian component + Impulsive component
- Total noise power is calibrated to achieve the target SNR
- Gaussian component: additive white Gaussian noise
- Impulsive component: sparse, high-amplitude spikes modeled as
  Bernoulli-Gaussian (random positions with large amplitude bursts)
- The noise power is split equally (50/50) between the two components
"""

import gc
import math
import numpy as np
import pywt
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import PyTorch model functions
from model_torch import build_unet, train_model, get_device, custom_loss


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Data paths
    'data_path': r'Data\Signalstandford.mat',

    # Signal processing
    'snr_levels': [-5, 0, 5],        # SNR levels to train on (dB)
    'wavelet': 'db6',                # Wavelet family for DWT
    'dwt_level': None,               # DWT decomposition level (None = max level)

    # Noise mixing
    'impulse_probability': 0.05,     # Probability of impulse at each sample
    'noise_split_ratio': 0.5,        # Fraction of total noise power allocated to Gaussian
                                     # (remainder goes to impulsive)

    # Data splitting
    'test_size': 0.15,               # Fraction for test set
    'val_size': 0.2,                 # Fraction of remaining for validation
    'random_state': 42,              # Random seed for reproducibility

    # Per-SNR hyperparameters
    # L1 weight in loss: loss = mse_weight * MSE + l1_weight * L1
    'l1_weight': {-5: 0.3, 0: 0.25, 5: 0.1},
    'mse_weight': {-5: 0.7, 0: 0.75, 5: 0.90},
    # SVD energy threshold per SNR level
    'energy_threshold': {-5: 0.45, 0: 0.6, 5: 0.8},

    # Model training  (input_shape is computed dynamically after DWT)
    'learning_rate': 0.001,          # Initial learning rate
    'epochs': 50,                    # Maximum training epochs
    'batch_size': 128,               # Training batch size
    'patience': 10,                  # Early stopping patience
    # Checkpoint path template — {snr} is replaced with the SNR level
    'checkpoint_template': 'checkpoints/best_model_wavelet_{snr}dB.pth',
}


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def load_and_scale_signals(data_path):
    """
    Load raw signals from .mat file and apply standard scaling.

    Args:
        data_path: Path to the .mat file containing signal data

    Returns:
        scaled_signals: Scaled signals array of shape (num_signals, signal_length)
        scaler: Fitted StandardScaler object for inverse transform if needed
    """
    data = loadmat(data_path)
    raw_data = data['arr']
    raw_signals = raw_data.T  # Transpose to (num_signals, signal_length)

    print(f"Loaded raw signals shape: {raw_signals.shape}")

    # Scale each signal feature-wise
    scaler = StandardScaler()
    scaled_signals = scaler.fit_transform(raw_signals.T).T

    print(f"Scaled signals shape: {scaled_signals.shape}")

    return scaled_signals, scaler


# =============================================================================
# Impulsive + Gaussian Noise Generation
# =============================================================================
def add_impulsive_gaussian_noise(signals, target_snr_db, impulse_prob=0.05,
                                  noise_split_ratio=0.5):
    """
    Add mixed impulsive + Gaussian noise to signals at a target SNR.

    The noise model:
        noise = gaussian_component + impulsive_component

    - Gaussian component ~ N(0, sigma_g^2)
    - Impulsive component = bernoulli(p) * N(0, sigma_i^2)

    Total noise power is split according to noise_split_ratio:
        sigma_g^2 = ratio * total_noise_power
        sigma_i^2 = (1 - ratio) * total_noise_power / p

    Args:
        signals: Clean signals array (num_signals, signal_length)
        target_snr_db: Target Signal-to-Noise Ratio in decibels
        impulse_prob: Probability of impulse at each sample (default 0.05)
        noise_split_ratio: Fraction of noise power for Gaussian (default 0.5)

    Returns:
        noisy_signals: Signals with added mixed noise
    """
    noisy_signals = np.zeros_like(signals)
    actual_snrs = []

    for i, signal in enumerate(signals):
        sig_power = np.mean(signal ** 2)

        if sig_power < 1e-10:
            noisy_signals[i] = signal
            continue

        sig_avg_db = 10 * np.log10(sig_power)
        noise_avg_db = sig_avg_db - target_snr_db
        total_noise_power = 10 ** (noise_avg_db / 10)

        gauss_power = noise_split_ratio * total_noise_power
        impulse_total_power = (1 - noise_split_ratio) * total_noise_power

        gauss_noise = np.random.normal(0, np.sqrt(gauss_power), signal.shape)

        bernoulli_mask = np.random.binomial(1, impulse_prob, signal.shape)
        impulse_sigma = np.sqrt(impulse_total_power / max(impulse_prob, 1e-10))
        impulse_amplitudes = np.random.normal(0, impulse_sigma, signal.shape)
        impulse_noise = bernoulli_mask * impulse_amplitudes

        mixed_noise = gauss_noise + impulse_noise
        noisy_signals[i] = signal + mixed_noise

        noise_power = np.mean(mixed_noise ** 2)
        if noise_power > 1e-10:
            actual_snr = 10 * np.log10(sig_power / noise_power)
            actual_snrs.append(actual_snr)

    mean_actual_snr = np.mean(actual_snrs) if actual_snrs else float('nan')

    print(f"Added impulsive+Gaussian mixed noise at {target_snr_db} dB target SNR.")
    print(f"  Impulse probability: {impulse_prob}")
    print(f"  Noise split (Gaussian/Impulsive): {noise_split_ratio:.0%}/{1-noise_split_ratio:.0%}")
    print(f"  Mean actual SNR achieved: {mean_actual_snr:.2f} dB")
    print(f"  Noisy signals shape: {noisy_signals.shape}")

    return noisy_signals


# =============================================================================
# Discrete Wavelet Transform (DWT) — Forward and Inverse
# =============================================================================
def compute_dwt_metadata(signal_length, wavelet='db4', level=None):
    """
    Compute DWT metadata (coefficient lengths, square side) for a given
    signal length. This is called once to determine the 2D reshape dimensions.

    Args:
        signal_length: Length of each 1D signal
        wavelet: Wavelet family name (e.g. 'db4')
        level: Decomposition level (None = max)

    Returns:
        Dictionary with:
            - 'coeff_lengths': list of lengths per level [cA_L, cD_L, ..., cD_1]
            - 'total_coeffs': total number of wavelet coefficients
            - 'square_side': side of the square 2D matrix (ceil(sqrt(total)))
            - 'level': actual decomposition level used
            - 'wavelet': wavelet name
    """
    dummy = np.zeros(signal_length)
    coeffs = pywt.wavedec(dummy, wavelet, level=level)
    coeff_lengths = [len(c) for c in coeffs]
    total = sum(coeff_lengths)
    side = int(math.ceil(math.sqrt(total)))

    return {
        'coeff_lengths': coeff_lengths,
        'total_coeffs': total,
        'square_side': side,
        'level': len(coeffs) - 1,  # number of detail levels
        'wavelet': wavelet,
    }


def apply_dwt(signals, wavelet='db4', level=None, dwt_meta=None):
    """
    Apply multi-level 1D DWT to each signal and reshape into square 2D matrices.

    Args:
        signals: Input signals array of shape (num_signals, signal_length)
        wavelet: Wavelet family name
        level: Decomposition level (None = max)
        dwt_meta: Pre-computed metadata (if None, computed from first signal)

    Returns:
        matrices: 2D coefficient matrices of shape (N, side, side)
        dwt_meta: Metadata dict needed for inverse transform
    """
    if dwt_meta is None:
        dwt_meta = compute_dwt_metadata(signals.shape[1], wavelet, level)

    side = dwt_meta['square_side']
    total = dwt_meta['total_coeffs']
    N = len(signals)

    matrices = np.zeros((N, side, side), dtype=np.float32)

    for i, sig in enumerate(signals):
        coeffs = pywt.wavedec(sig, wavelet, level=dwt_meta['level'])
        # Flatten all coefficient arrays into 1D
        flat = np.concatenate(coeffs)
        # Pad to square size and reshape
        padded = np.zeros(side * side, dtype=np.float32)
        padded[:total] = flat.astype(np.float32)
        matrices[i] = padded.reshape(side, side)

    print(f"DWT coefficients shape: {matrices.shape}, "
          f"wavelet={wavelet}, level={dwt_meta['level']}, "
          f"total_coeffs={total}, square_side={side}")

    return matrices, dwt_meta


def apply_idwt(matrices, dwt_meta):
    """
    Apply inverse DWT to recover 1D signals from 2D coefficient matrices.

    Args:
        matrices: 2D coefficient matrices of shape (N, side, side)
        dwt_meta: Metadata dict from apply_dwt

    Returns:
        signals: Reconstructed 1D signals of shape (N, signal_length)
    """
    total = dwt_meta['total_coeffs']
    coeff_lengths = dwt_meta['coeff_lengths']
    wavelet = dwt_meta['wavelet']
    N = matrices.shape[0]

    # Reconstruct first signal to get output length
    flat0 = matrices[0].reshape(-1)[:total]
    coeffs0 = _split_coefficients(flat0, coeff_lengths)
    rec0 = pywt.waverec(coeffs0, wavelet)
    sig_len = len(rec0)

    signals = np.zeros((N, sig_len), dtype=np.float64)
    signals[0] = rec0

    for i in range(1, N):
        flat = matrices[i].reshape(-1)[:total]
        coeffs = _split_coefficients(flat, coeff_lengths)
        signals[i] = pywt.waverec(coeffs, wavelet)

    return signals


def _split_coefficients(flat_array, coeff_lengths):
    """
    Split a flattened coefficient array back into the list format
    expected by pywt.waverec.

    Args:
        flat_array: 1D array of concatenated wavelet coefficients
        coeff_lengths: List of lengths for each coefficient array

    Returns:
        List of coefficient arrays [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    """
    coeffs = []
    start = 0
    for length in coeff_lengths:
        coeffs.append(flat_array[start:start + length])
        start += length
    return coeffs


# =============================================================================
# Data Splitting
# =============================================================================
def split_data(noisy_dwt, clean_dwt, test_size=0.15, val_size=0.2, random_state=42):
    """
    Split DWT coefficient matrices into train, validation, and test sets.

    Args:
        noisy_dwt: Noisy DWT coefficient matrices (N, side, side)
        clean_dwt: Clean DWT coefficient matrices (N, side, side)
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing train/val/test splits
    """
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        noisy_dwt, clean_dwt, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    print(f"Data split sizes:")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def stack_for_unet(matrices):
    """
    Add a channel dimension for the U-Net (1-channel input).

    Args:
        matrices: DWT coefficient matrices of shape (N, H, W)

    Returns:
        Stacked array of shape (N, H, W, 1) — channels-last
    """
    return matrices[..., np.newaxis]


# =============================================================================
# SVD-based Semi-Clean Generation
# =============================================================================
def auto_energy_index(singular_values, threshold=0.90):
    """
    Find the index where cumulative energy exceeds threshold.

    Args:
        singular_values: Array of singular values from SVD
        threshold: Cumulative energy threshold (default 0.90 = 90%)

    Returns:
        Index where cumulative energy first exceeds threshold
    """
    energy = np.cumsum(singular_values) / np.sum(singular_values)
    return np.argmax(energy >= threshold)


def generate_semi_clean_svd(noisy_matrices, energy_threshold=0.90):
    """
    Generate semi-clean signals using SVD-based projection denoising.

    Algorithm:
    1. Compute autocorrelation matrix: R = X @ X^T
    2. Perform SVD on R to find eigenvectors
    3. Identify noise subspace using energy threshold
    4. Create projection matrix to remove noise subspace
    5. Apply projection to get semi-clean signal

    Args:
        noisy_matrices: Noisy DWT coefficient matrices (N, side, side)
        energy_threshold: Threshold for signal/noise subspace separation

    Returns:
        semi_clean_matrices: Denoised coefficient matrices of same shape
    """
    semi_clean_list = []

    for i, matrix in enumerate(noisy_matrices):
        rows, cols = matrix.shape
        assert rows == cols, f"Expected square matrix, got {matrix.shape}"

        # Compute autocorrelation matrix (real coefficients → use .T, no conj)
        autocorr = np.dot(matrix, matrix.T)

        # SVD of autocorrelation matrix
        U, S, _ = np.linalg.svd(autocorr, full_matrices=True)

        # Find signal/noise boundary using energy threshold
        signal_dim = auto_energy_index(S, threshold=energy_threshold)

        # Extract noise subspace eigenvectors
        U_noise = U[:, signal_dim:]

        # Projection matrix to remove noise subspace
        I = np.eye(rows)
        noise_projection = np.dot(U_noise, U_noise.T)
        signal_projection = I - noise_projection

        # Apply projection
        semi_clean = np.dot(signal_projection, matrix)
        semi_clean_list.append(semi_clean)

    semi_clean_matrices = np.array(semi_clean_list)
    print(f"Generated semi-clean matrices shape: {semi_clean_matrices.shape}")

    return semi_clean_matrices


# =============================================================================
# Wavelet-domain Prediction (1-channel, real output)
# =============================================================================
def predict_wavelet(model, X_test, device, batch_size=64):
    """
    Run U-Net inference on wavelet coefficient matrices (1-channel).

    Args:
        model: Trained PyTorch model
        X_test: Test input of shape (N, H, W, 1) — channels-last
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        Denoised 2D coefficient matrices of shape (N, H, W)
    """
    model.eval()

    # Convert to channels-first: (N, H, W, 1) -> (N, 1, H, W)
    X_test_cf = np.transpose(X_test, (0, 3, 1, 2))

    decoded_list = []

    with torch.no_grad():
        for i in range(0, len(X_test_cf), batch_size):
            batch = torch.tensor(
                X_test_cf[i:i + batch_size], dtype=torch.float32
            ).to(device)
            output = model(batch)
            decoded_list.append(output.cpu().numpy())

    decoded_data = np.concatenate(decoded_list, axis=0)

    # Convert back to channels-last: (N, 1, H, W) -> (N, H, W, 1) -> (N, H, W)
    decoded_data = np.transpose(decoded_data, (0, 2, 3, 1))
    decoded_matrices = decoded_data[:, :, :, 0]  # squeeze channel dim

    print(f"Predicted denoised matrices shape: {decoded_matrices.shape}")
    return decoded_matrices


# =============================================================================
# Single-SNR Pipeline
# =============================================================================
def run_single_snr_pipeline(target_snr_db):
    """
    Execute the complete wavelet-based denoising pipeline for one SNR level.

    Args:
        target_snr_db: Target Signal-to-Noise Ratio in decibels

    Returns:
        Dictionary with all processed data and results for this SNR level
    """
    checkpoint_path = CONFIG['checkpoint_template'].format(snr=target_snr_db)
    energy_threshold = CONFIG['energy_threshold'][target_snr_db]
    l1_weight = CONFIG['l1_weight'][target_snr_db]
    mse_weight = CONFIG['mse_weight'][target_snr_db]

    print("\n" + "#" * 60)
    print(f"  Wavelet Training Pipeline for SNR = {target_snr_db} dB")
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Energy threshold: {energy_threshold}")
    print(f"  Loss weights:     MSE={mse_weight}, L1={l1_weight}")
    print(f"  Wavelet:          {CONFIG['wavelet']}")
    print("#" * 60)

    # Step 1: Load and scale signals
    print("\n[Step 1] Loading and scaling signals...")
    scaled_signals, scaler = load_and_scale_signals(CONFIG['data_path'])

    # Step 2: Add Impulsive + Gaussian mixed noise
    print(f"\n[Step 2] Adding impulsive + Gaussian mixed noise at {target_snr_db} dB...")
    noisy_signals = add_impulsive_gaussian_noise(
        scaled_signals,
        target_snr_db,
        impulse_prob=CONFIG['impulse_probability'],
        noise_split_ratio=CONFIG['noise_split_ratio'],
    )

    # Step 3: Apply DWT to both clean and noisy signals
    print("\n[Step 3] Applying Discrete Wavelet Transform...")
    clean_dwt, dwt_meta = apply_dwt(
        scaled_signals, CONFIG['wavelet'], CONFIG['dwt_level']
    )

    # Free memory: delete scaled_signals after DWT
    del scaled_signals
    gc.collect()
    print("  (freed scaled_signals from memory)")

    noisy_dwt, _ = apply_dwt(
        noisy_signals, CONFIG['wavelet'], CONFIG['dwt_level'], dwt_meta=dwt_meta
    )

    del noisy_signals
    gc.collect()
    print("  (freed noisy_signals from memory)")

    # Determine input_shape from the DWT output
    side = dwt_meta['square_side']
    input_shape = (side, side, 1)
    print(f"  U-Net input_shape: {input_shape}")

    # Step 4: Split data
    print("\n[Step 4] Splitting data...")
    splits = split_data(
        noisy_dwt, clean_dwt,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )

    del noisy_dwt, clean_dwt
    gc.collect()
    print("  (freed full DWT arrays from memory)")

    # Step 5: Stack for U-Net (add channel dimension)
    print("\n[Step 5] Preparing data format (adding channel dim)...")
    train_noisy_stacked = stack_for_unet(splits['X_train'])
    train_clean_stacked = stack_for_unet(splits['y_train'])
    val_noisy_stacked = stack_for_unet(splits['X_val'])
    val_clean_stacked = stack_for_unet(splits['y_val'])
    test_noisy_stacked = stack_for_unet(splits['X_test'])
    test_clean_stacked = stack_for_unet(splits['y_test'])

    print(f"Stacked train noisy: {train_noisy_stacked.shape}")
    print(f"Stacked train clean: {train_clean_stacked.shape}")

    # Step 6: Generate semi-clean signals using SVD
    print("\n[Step 6] Generating semi-clean signals using SVD...")
    S_clean_train = generate_semi_clean_svd(splits['X_train'], energy_threshold)
    S_clean_val = generate_semi_clean_svd(splits['X_val'], energy_threshold)
    S_clean_test = generate_semi_clean_svd(splits['X_test'], energy_threshold)

    # Stack semi-clean for training targets
    S_clean_train_stacked = stack_for_unet(S_clean_train)
    S_clean_val_stacked = stack_for_unet(S_clean_val)
    S_clean_test_stacked = stack_for_unet(S_clean_test)

    print(f"Semi-clean train stacked: {S_clean_train_stacked.shape}")

    # Step 7: Create and train U-Net model
    print("\n[Step 7] Training U-Net model (PyTorch, 1-channel wavelet)...")

    device = get_device()
    model = build_unet(input_shape)
    model = model.to(device)

    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(
        model, optimizer,
        X_train=train_noisy_stacked,
        y_train=S_clean_train_stacked,
        X_val=val_noisy_stacked,
        y_val=S_clean_val_stacked,
        device=device,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        patience=CONFIG['patience'],
        checkpoint_path=checkpoint_path,
        mse_weight=mse_weight,
        l1_weight=l1_weight
    )

    # Step 8: Predict denoised signals (wavelet domain)
    print("\n[Step 8] Predicting denoised test signals...")
    decoded_matrices = predict_wavelet(model, test_noisy_stacked, device)

    print(f"\nPipeline Complete for SNR = {target_snr_db} dB!")
    print(f"Checkpoint saved: {checkpoint_path}")

    # Return all processed data and results
    return {
        'target_snr_db': target_snr_db,
        'checkpoint_path': checkpoint_path,
        'dwt_meta': dwt_meta,
        'input_shape': input_shape,
        'scaler': scaler,
        'model': model,
        'history': history,
        'splits': splits,
        'stacked': {
            'train_noisy': train_noisy_stacked,
            'train_clean': train_clean_stacked,
            'val_noisy': val_noisy_stacked,
            'val_clean': val_clean_stacked,
            'test_noisy': test_noisy_stacked,
            'test_clean': test_clean_stacked,
        },
        'semi_clean': {
            'train': S_clean_train,
            'val': S_clean_val,
            'test': S_clean_test,
        },
        'predictions': {
            'decoded_matrices': decoded_matrices,
        }
    }


# =============================================================================
# Main — Train all SNR levels
# =============================================================================
def main():
    """Train a separate model for each SNR level in CONFIG['snr_levels']."""

    print("=" * 60)
    print("Signal Denoising Pipeline — Wavelet Transform")
    print("  ** Impulsive + Gaussian Mixed Noise Experiment **")
    print(f"  SNR levels: {CONFIG['snr_levels']}")
    print(f"  Wavelet:    {CONFIG['wavelet']}")
    print("=" * 60)

    all_results = {}
    for snr_db in CONFIG['snr_levels']:
        results = run_single_snr_pipeline(snr_db)
        all_results[snr_db] = results

        # Free heavy arrays between runs to save memory
        del results['stacked']
        del results['semi_clean']
        gc.collect()

    print("\n" + "=" * 60)
    print("All SNR Levels Complete!")
    for snr_db in CONFIG['snr_levels']:
        ckpt = CONFIG['checkpoint_template'].format(snr=snr_db)
        print(f"  {snr_db:+d} dB  ->  {ckpt}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    results = main()
