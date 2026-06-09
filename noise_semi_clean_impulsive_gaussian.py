"""
Signal Denoising Pipeline using SVD-based Semi-Clean Generation
(Impulsive + Gaussian Mixed Noise Experiment)

Pipeline Overview:
1. Load raw signals from Stanford dataset
2. Scale signals using StandardScaler
3. Add MIXED noise (Impulsive + Gaussian) at target SNR level
4. Apply STFT to both clean and noisy signals
5. Split data into train/val/test sets
6. Generate Semi-Clean versions using SVD algorithm (projection-based denoising)
7. Train U-Net model to learn noisy -> semi-clean mapping
8. Predict denoised signals

The SVD algorithm works by:
- Computing autocorrelation matrix of noisy STFT coefficients
- Performing SVD on autocorrelation to find signal/noise subspaces
- Using energy threshold to determine signal subspace dimension
- Projecting out noise subspace to get semi-clean signal

Noise Model:
- Mixed noise = Gaussian component + Impulsive component
- Total noise power is calibrated to achieve the target SNR
- Gaussian component: additive white Gaussian noise
- Impulsive component: sparse, high-amplitude spikes modeled as
  Bernoulli-Gaussian (random positions with large amplitude bursts)
- The noise power is split equally (50/50) between the two components
"""

import gc
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import PyTorch model functions
from model_torch import build_unet, train_model, predict_and_reconstruct, get_device, custom_loss


# =============================================================================
# Configuration
# =============================================================================
CONFIG = {
    # Data paths
    'data_path': r'Data\Signalstandford.mat',
    
    # Signal processing
    'snr_levels': [-5, 0, 5],        # SNR levels to train on (dB)
    'stft_nperseg': 155,             # STFT window size (samples)
    'stft_window': 'hann',           # STFT window type
    
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
    'l1_weight': {-5: 0.15, 0: 0.1, 5: 0.01},
    'mse_weight': {-5: 0.85, 0: 0.9, 5: 0.99},
    # SVD energy threshold per SNR level
    'energy_threshold': {-5: 0.7, 0: 0.7, 5: 0.8},
    
    # Model training
    'input_shape': (78, 78, 2),      # Input shape for U-Net (freq, time, channels)
    'learning_rate': 0.001,          # Initial learning rate
    'epochs': 30,                    # Maximum training epochs
    'batch_size': 128,               # Training batch size
    'patience': 10,                  # Early stopping patience
    # Checkpoint path template — {snr} is replaced with the SNR level
    'checkpoint_template': 'checkpoints/best_model_impulsive_gaussian_{snr}dB.pth',
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
# Noise Addition — Impulsive + Gaussian Mixed Noise
# =============================================================================
def add_impulsive_gaussian_noise(signals, target_snr_db, impulse_prob=0.05,
                                  noise_split_ratio=0.5):
    """
    Add mixed Impulsive + Gaussian noise to signals at a specified SNR level.
    
    The total noise power is determined by the target SNR. It is then split
    between a Gaussian component and an impulsive (Bernoulli-Gaussian) component
    according to noise_split_ratio.
    
    Impulsive noise model (Bernoulli-Gaussian):
        n_imp[k] = b[k] * g[k]
    where b[k] ~ Bernoulli(impulse_prob) and g[k] ~ N(0, sigma_imp^2).
    The variance of this product is impulse_prob * sigma_imp^2, which is set
    to (1 - noise_split_ratio) * total_noise_power.
    
    Args:
        signals: Clean signals array of shape (num_signals, signal_length)
        target_snr_db: Target Signal-to-Noise Ratio in decibels
        impulse_prob: Probability of an impulse occurring at each sample
        noise_split_ratio: Fraction of total noise power allocated to Gaussian.
                           The remaining (1 - noise_split_ratio) goes to impulsive.
        
    Returns:
        noisy_signals: Noisy signals array of same shape as input
    """
    noisy_signals = np.zeros_like(signals)
    
    for i, signal in enumerate(signals):
        # Calculate signal power in dB
        signal_power = np.square(signal)
        sig_avg_watts = np.mean(signal_power)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        
        # Calculate required total noise power based on target SNR
        noise_avg_db = sig_avg_db - target_snr_db
        total_noise_power = 10 ** (noise_avg_db / 10)
        
        # Split noise power between Gaussian and impulsive components
        gaussian_power = noise_split_ratio * total_noise_power
        impulsive_power = (1.0 - noise_split_ratio) * total_noise_power
        
        # --- Gaussian component ---
        gaussian_noise = np.random.normal(0, np.sqrt(gaussian_power), len(signal))
        
        # --- Impulsive component (Bernoulli-Gaussian) ---
        # b[k] ~ Bernoulli(impulse_prob): sparse mask
        impulse_mask = np.random.binomial(1, impulse_prob, len(signal)).astype(float)
        
        # Variance of Bernoulli-Gaussian: impulse_prob * sigma_imp^2 = impulsive_power
        # => sigma_imp^2 = impulsive_power / impulse_prob
        sigma_imp = np.sqrt(impulsive_power / impulse_prob)
        impulse_values = np.random.normal(0, sigma_imp, len(signal))
        
        impulsive_noise = impulse_mask * impulse_values
        
        # --- Combined noise ---
        mixed_noise = gaussian_noise + impulsive_noise
        noisy_signals[i] = signal + mixed_noise
    
    # Compute and report the actual achieved SNR
    actual_snrs = []
    for i in range(len(signals)):
        s_power = np.mean(np.square(signals[i]))
        n_power = np.mean(np.square(noisy_signals[i] - signals[i]))
        if n_power > 0:
            actual_snrs.append(10 * np.log10(s_power / n_power))
    mean_actual_snr = np.mean(actual_snrs) if actual_snrs else float('nan')
    
    print(f"Added impulsive+Gaussian mixed noise at {target_snr_db} dB target SNR.")
    print(f"  Impulse probability: {impulse_prob}")
    print(f"  Noise split (Gaussian/Impulsive): {noise_split_ratio:.0%}/{1-noise_split_ratio:.0%}")
    print(f"  Mean actual SNR achieved: {mean_actual_snr:.2f} dB")
    print(f"  Noisy signals shape: {noisy_signals.shape}")
    
    return noisy_signals


# =============================================================================
# STFT Transformation
# =============================================================================
def apply_stft(signals, window='hann', nperseg=155, dtype=np.complex64):
    """
    Apply Short-Time Fourier Transform to signals.
    
    Args:
        signals: Input signals array of shape (num_signals, signal_length)
        window: Window function for STFT
        nperseg: Number of samples per segment
        dtype: Data type for output (complex64 uses half the memory of complex128)
        
    Returns:
        stft_coeffs: Complex STFT coefficients of shape (num_signals, freq_bins, time_frames)
    """
    # Get shape from first signal to preallocate array
    _, _, first_stft = stft(signals[0], window=window, nperseg=nperseg)
    freq_bins, time_frames = first_stft.shape
    
    # Preallocate with specified dtype (complex64 = half memory of default complex128)
    stft_coeffs = np.zeros((len(signals), freq_bins, time_frames), dtype=dtype)
    stft_coeffs[0] = first_stft.astype(dtype)
    
    for i in range(1, len(signals)):
        _, _, stft_coeff = stft(signals[i], window=window, nperseg=nperseg)
        stft_coeffs[i] = stft_coeff.astype(dtype)
    
    print(f"STFT coefficients shape: {stft_coeffs.shape}, dtype: {stft_coeffs.dtype}")
    
    return stft_coeffs


# =============================================================================
# Data Splitting
# =============================================================================
def split_data(noisy_stft, clean_stft, test_size=0.15, val_size=0.2, random_state=42):
    """
    Split STFT data into train, validation, and test sets.
    
    Args:
        noisy_stft: Noisy STFT coefficients (input features)
        clean_stft: Clean STFT coefficients (targets)
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing train/val/test splits for both noisy and clean data
    """
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        noisy_stft, clean_stft, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }
    
    print(f"Train: noisy {X_train.shape}, clean {y_train.shape}")
    print(f"Val:   noisy {X_val.shape}, clean {y_val.shape}")
    print(f"Test:  noisy {X_test.shape}, clean {y_test.shape}")
    
    return splits


def stack_real_imag(stft_coeffs):
    """
    Stack real and imaginary parts of complex STFT coefficients.
    
    Args:
        stft_coeffs: Complex STFT coefficients
        
    Returns:
        Stacked array with shape (..., 2) where last dim is [real, imag]
    """
    return np.stack([np.real(stft_coeffs), np.imag(stft_coeffs)], axis=-1)


# =============================================================================
# SVD-based Semi-Clean Generation
# =============================================================================
def auto_energy_index(singular_values, threshold=0.90):
    """
    Find the index where cumulative energy exceeds threshold.
    
    This determines the boundary between signal and noise subspaces
    based on the energy (sum of squared singular values) distribution.
    
    Args:
        singular_values: Array of singular values from SVD
        threshold: Cumulative energy threshold (default 0.90 = 90%)
        
    Returns:
        Index where cumulative energy first exceeds threshold
    """
    energy = np.cumsum(singular_values) / np.sum(singular_values)
    return np.argmax(energy >= threshold)


def generate_semi_clean_svd(noisy_stft_signals, energy_threshold=0.90):
    """
    Generate semi-clean signals using SVD-based projection denoising.
    
    Algorithm:
    1. Compute autocorrelation matrix: R = X @ X^H
    2. Perform SVD on R to find eigenvectors
    3. Identify noise subspace using energy threshold
    4. Create projection matrix to remove noise subspace
    5. Apply projection to get semi-clean signal
    
    Args:
        noisy_stft_signals: Noisy STFT coefficients of shape (N, freq, time)
        energy_threshold: Threshold for signal/noise subspace separation
        
    Returns:
        semi_clean_signals: Denoised STFT coefficients of same shape
    """
    semi_clean_list = []
    
    for i, signal_stft in enumerate(noisy_stft_signals):
        # Verify expected shape (square matrix for this implementation)
        freq_bins, time_frames = signal_stft.shape
        assert freq_bins == time_frames, f"Expected square matrix, got {signal_stft.shape}"
        
        # Compute autocorrelation matrix
        autocorr = np.dot(signal_stft, np.conj(signal_stft.T))
        
        # SVD of autocorrelation matrix
        U, S, _ = np.linalg.svd(autocorr, full_matrices=True)
        
        # Find signal/noise boundary using energy threshold
        signal_dim = auto_energy_index(S, threshold=energy_threshold)
        
        # Extract noise subspace eigenvectors
        U_noise = U[:, signal_dim:]
        
        # Create projection matrix to remove noise subspace
        # P = I - U_noise @ U_noise^H projects onto signal subspace
        I = np.eye(freq_bins)
        noise_projection = np.dot(U_noise, np.conj(U_noise.T))
        signal_projection = I - noise_projection
        
        # Apply projection to get semi-clean signal
        semi_clean = np.dot(signal_projection, signal_stft)
        semi_clean_list.append(semi_clean)
    
    semi_clean_signals = np.array(semi_clean_list)
    print(f"Generated semi-clean signals shape: {semi_clean_signals.shape}")
    
    return semi_clean_signals


# =============================================================================
# Single-SNR Pipeline
# =============================================================================
def run_single_snr_pipeline(target_snr_db):
    """
    Execute the complete signal denoising pipeline for one SNR level.

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
    print(f"  Training pipeline for SNR = {target_snr_db} dB")
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Energy threshold: {energy_threshold}")
    print(f"  Loss weights:     MSE={mse_weight}, L1={l1_weight}")
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

    # Step 3: Apply STFT to both clean and noisy signals
    print("\n[Step 3] Applying STFT...")
    clean_stft = apply_stft(scaled_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])

    # Free memory: delete scaled_signals after STFT is computed
    del scaled_signals
    gc.collect()
    print("  (freed scaled_signals from memory)")

    noisy_stft = apply_stft(noisy_signals, CONFIG['stft_window'], CONFIG['stft_nperseg'])

    # Free memory: delete noisy_signals after STFT is computed
    del noisy_signals
    gc.collect()
    print("  (freed noisy_signals from memory)")

    # Step 4: Split data into train/val/test
    print("\n[Step 4] Splitting data...")
    splits = split_data(
        noisy_stft, clean_stft,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )

    # Free memory: delete full STFT arrays after splitting
    del noisy_stft, clean_stft
    gc.collect()
    print("  (freed full STFT arrays from memory)")

    # Step 5: Stack real and imaginary parts for neural network input
    print("\n[Step 5] Preparing data format (stacking real/imag)...")
    train_noisy_stacked = stack_real_imag(splits['X_train'])
    train_clean_stacked = stack_real_imag(splits['y_train'])
    val_noisy_stacked = stack_real_imag(splits['X_val'])
    val_clean_stacked = stack_real_imag(splits['y_val'])
    test_noisy_stacked = stack_real_imag(splits['X_test'])
    test_clean_stacked = stack_real_imag(splits['y_test'])

    print(f"Stacked train noisy: {train_noisy_stacked.shape}")
    print(f"Stacked train clean: {train_clean_stacked.shape}")

    # Step 6: Generate semi-clean signals using SVD
    print("\n[Step 6] Generating semi-clean signals using SVD...")
    S_clean_train = generate_semi_clean_svd(splits['X_train'], energy_threshold)
    S_clean_val = generate_semi_clean_svd(splits['X_val'], energy_threshold)
    S_clean_test = generate_semi_clean_svd(splits['X_test'], energy_threshold)

    # Stack semi-clean signals (real/imag) for training targets
    S_clean_train_stacked = stack_real_imag(S_clean_train)
    S_clean_val_stacked = stack_real_imag(S_clean_val)
    S_clean_test_stacked = stack_real_imag(S_clean_test)

    print(f"Semi-clean train stacked: {S_clean_train_stacked.shape}")

    # Step 7: Create and train U-Net model (PyTorch)
    print("\n[Step 7] Training U-Net model (PyTorch)...")

    device = get_device()
    model = build_unet(CONFIG['input_shape'])
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

    # Step 8: Predict denoised signals
    print("\n[Step 8] Predicting denoised test signals...")
    decoded_complex = predict_and_reconstruct(model, test_noisy_stacked, device)

    print(f"\nPipeline Complete for SNR = {target_snr_db} dB!")
    print(f"Checkpoint saved: {checkpoint_path}")

    # Return all processed data and results
    return {
        'target_snr_db': target_snr_db,
        'checkpoint_path': checkpoint_path,
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
            'decoded_complex': decoded_complex,
        }
    }


# =============================================================================
# Main — Train all SNR levels
# =============================================================================
def main():
    """Train a separate model for each SNR level in CONFIG['snr_levels']."""

    print("=" * 60)
    print("Signal Denoising Pipeline - SVD Semi-Clean Generation")
    print("  ** Impulsive + Gaussian Mixed Noise Experiment **")
    print(f"  SNR levels: {CONFIG['snr_levels']}")
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
